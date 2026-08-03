import logging
import os
import pickle
import time
from typing import Optional, Dict, Any, List

logger = logging.getLogger(__name__)

import numpy as np
from openai import OpenAI
import pandas as pd
import plotly.graph_objects as go
import sqlalchemy as sa
import streamlit as st
from pgvector.sqlalchemy import VECTOR

# Import existing authentication and configuration from your base module
from toolkit.base import get_auth_token, get_databricks_host, get_db_engine

# ─── Configuration ───────────────────────────────────────────────
# Standard Databricks hosted embedding model
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "system.ai.gte-large-en")
# Cosine similarity threshold (0.90 to 0.95 is ideal for semantic matching)
SIMILARITY_THRESHOLD = float(os.environ.get("SIMILARITY_THRESHOLD", "0.92"))

# Embedding dimension for system.ai.gte-large-en
EMBEDDING_DIM = 1024


class SemanticCache:
    """
    A persistent semantic cache backed by Postgres + pgvector.

    Stores embeddings in a vector(1024) column and uses the native <=>
    cosine distance operator for ANN lookup — no Python-side similarity
    loop and no full-table scan.
    """

    def __init__(self):
        self._init_db()

    def _init_db(self):
        """
        Verifies that star_semantic_cache exists and is reachable.

        The table and IVFFlat index must be created once by an admin from the
        Lakebase SQL Editor before the app starts.  The app user only needs
        SELECT / INSERT / UPDATE / DELETE on the table — no DDL privileges.

        Setup SQL (run once as admin):
        -----------------------------------------------------------------------
        CREATE EXTENSION IF NOT EXISTS vector;

        CREATE TABLE IF NOT EXISTS star_semantic_cache (
            id             SERIAL PRIMARY KEY,
            prompt         TEXT UNIQUE,
            embedding      vector(1024),
            final_text     TEXT,
            dfs_pickle     BYTEA,
            figures_pickle BYTEA,
            timestamp      FLOAT
        );

        CREATE INDEX IF NOT EXISTS cache_embedding_idx
            ON star_semantic_cache
            USING ivfflat (embedding vector_cosine_ops)
            WITH (lists = 10);

        GRANT SELECT, INSERT, UPDATE, DELETE
            ON star_semantic_cache TO <your_app_user>;
        -----------------------------------------------------------------------
        """
        engine = get_db_engine()
        with engine.connect() as conn:
            row = conn.execute(sa.text("""
                SELECT 1
                FROM   information_schema.tables
                WHERE  table_name = 'star_semantic_cache'
                LIMIT  1
            """)).fetchone()

            if row is None:
                raise RuntimeError(
                    "Cache table 'star_semantic_cache' not found. "
                    "Run the setup SQL in the Lakebase SQL Editor first "
                    "(see the docstring in SemanticCache._init_db for the full script)."
                )

        logger.info("SemanticCache: star_semantic_cache table verified.")

    def _get_embedding(self, text: str) -> np.ndarray:
        """
        Fetches a dense vector embedding for the given text from the Databricks
        embedding endpoint.  Always uses a freshly fetched auth token.
        Input is lowercased and stripped before embedding for consistent
        similarity scoring.
        """
        client = OpenAI(
            api_key=get_auth_token(),
            base_url=f"{get_databricks_host()}/ai-gateway/mlflow/v1"
        )
        response = client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=[text.strip().lower()]
        )
        return np.array(response.data[0].embedding, dtype=np.float32)

    @staticmethod
    def _vec_to_str(vec: np.ndarray) -> str:
        """Converts a numpy float32 array to the pgvector literal '[x,y,...]'."""
        return "[" + ",".join(map(str, vec.tolist())) + "]"

    def check_cache(self, user_prompt: str) -> Optional[Dict[str, Any]]:
        """
        Checks if a semantically equivalent prompt exists in the cache.

        Issues a single SQL query using pgvector's <=> cosine distance operator
        and the IVFFlat index — O(log n) instead of the former O(n) Python loop.

        Returns the cached dictionary if the top match has
        similarity >= SIMILARITY_THRESHOLD, else None.
        """
        try:
            query_vector = self._get_embedding(user_prompt)
        except Exception as e:
            st.warning(f"Cache embedding generation failed: {e}")
            return None

        vec_str = self._vec_to_str(query_vector)

        sql = sa.text("""
            SELECT prompt,
                   final_text,
                   dfs_pickle,
                   figures_pickle,
                   1 - (embedding <=> CAST(:vec AS vector)) AS similarity
            FROM   star_semantic_cache
            ORDER  BY embedding <=> CAST(:vec AS vector)
            LIMIT  1
        """)

        try:
            engine = get_db_engine()
            with engine.connect() as conn:
                row = conn.execute(sql, {"vec": vec_str}).fetchone()
        except Exception as e:
            logger.warning(f"Cache lookup failed ({type(e).__name__}): {e}", exc_info=True)
            return None

        if row is None:
            return None

        matched_prompt, final_text, dfs_blob, figs_blob, similarity = row

        if similarity < SIMILARITY_THRESHOLD:
            return None

        dfs = pickle.loads(bytes(dfs_blob)) if dfs_blob else []
        figures = pickle.loads(bytes(figs_blob)) if figs_blob else []

        logger.info(
            f"Cache hit — similarity={similarity:.3f} "
            f"for prompt: '{matched_prompt[:60]}...'"
        )
        return {
            "content": final_text,
            "dfs": dfs,
            "figures": figures,
            "matched_prompt": matched_prompt,
            "similarity": round(similarity, 3),
        }

    def save_to_cache(
        self,
        user_prompt: str,
        final_text: str,
        dfs: List[pd.DataFrame],
        figures: List[go.Figure],
    ):
        """
        Serializes and saves a successful agent execution to Postgres.

        Uses INSERT ... ON CONFLICT (prompt) DO UPDATE so re-running the same
        prompt always refreshes the cached result rather than failing.

        BYTEA columns are passed via the raw pg8000 connection using named
        parameters so the driver handles bytes natively without SQLAlchemy
        coercing them to strings.
        """
        try:
            embedding = self._get_embedding(user_prompt)
            vec_str = self._vec_to_str(embedding)

            dfs_blob = pickle.dumps(dfs)
            figs_blob = pickle.dumps(figures)

            engine = get_db_engine()
            with engine.begin() as conn:
                # Access the underlying pg8000 connection so BYTEA params are
                # passed as raw bytes rather than going through SQLAlchemy's
                # text() layer, which would stringify them.
                raw = conn.connection
                raw.run(
                    """
                    INSERT INTO star_semantic_cache
                        (prompt, embedding, final_text, dfs_pickle, figures_pickle, timestamp)
                    VALUES
                        (:p, CAST(:v AS vector), :t, :d, :f, :ts)
                    ON CONFLICT (prompt) DO UPDATE SET
                        embedding      = EXCLUDED.embedding,
                        final_text     = EXCLUDED.final_text,
                        dfs_pickle     = EXCLUDED.dfs_pickle,
                        figures_pickle = EXCLUDED.figures_pickle,
                        timestamp      = EXCLUDED.timestamp
                    """,
                    p=user_prompt.strip().lower(),
                    v=vec_str,
                    t=final_text,
                    d=dfs_blob,
                    f=figs_blob,
                    ts=time.time(),
                )

        except Exception as e:
            # Log but never crash the primary UI loop
            logger.warning(
                f"Failed to save execution to semantic cache ({type(e).__name__}): {e}",
                exc_info=True,
            )

    def delete_from_cache(self, user_prompt: str):
        """
        Permanently removes any cache entry whose prompt is a semantic match for
        user_prompt (above SIMILARITY_THRESHOLD), ensuring the next run is always
        a fresh execution rather than a cache hit.

        Uses the same pgvector cosine distance query to identify matching IDs,
        then deletes them in a single WHERE id = ANY(...) statement.
        """
        try:
            query_vector = self._get_embedding(user_prompt)
        except Exception as e:
            logger.warning(f"Cache eviction embedding failed ({type(e).__name__}): {e}")
            return

        vec_str = self._vec_to_str(query_vector)

        find_sql = sa.text("""
            SELECT id,
                   1 - (embedding <=> CAST(:vec AS vector)) AS similarity
            FROM   star_semantic_cache
            WHERE  1 - (embedding <=> CAST(:vec AS vector)) >= :threshold
        """)

        try:
            engine = get_db_engine()
            with engine.begin() as conn:
                rows = conn.execute(
                    find_sql,
                    {"vec": vec_str, "threshold": SIMILARITY_THRESHOLD},
                ).fetchall()

                if not rows:
                    return

                ids_to_delete = [row[0] for row in rows]

                conn.execute(
                    sa.text(
                        "DELETE FROM star_semantic_cache WHERE id = ANY(:ids)"
                    ),
                    {"ids": ids_to_delete},
                )

                logger.info(
                    f"Cache eviction: removed {len(ids_to_delete)} "
                    f"entries matching '{user_prompt[:60]}...'"
                )

        except Exception as e:
            logger.warning(
                f"Cache eviction query failed ({type(e).__name__}): {e}"
            )


# Instantiate a global cache object to be imported into app.py
agent_cache = SemanticCache()
