import contextlib
import io
import traceback
from typing import Any, Dict, List, Optional, Union

import mlflow
import numpy as np
import pandas as pd

from agent.memory import DataFrameMemory
from .base import run_sql_query, get_join_clause, TABLE_DIMENSIONS
from .validators import validate_safe_python_code, SecurityViolationError


__all__ = [
    "link_tables",
    "execute_sql_query_tool",
    "calculate_unit_economics_tool",
    "calculate_ratio_tool",
    "join_dataframes_tool",
    "pivot_dataframe_tool",
    "execute_python_tool",
]


# ─── Helper Functions ───────────────────────────────────────────
@mlflow.trace(name="link_tables")
def _link_tables(
    tables: Union[str, List[str]], 
    columns: Optional[List[str]] = None, 
    where_clause: Optional[str] = None, 
    group_by: Optional[List[str]] = None,
    order_by: Optional[str] = None,
    limit: Optional[int] = 100000, 
    random_order: bool = False
) -> pd.DataFrame:
    """
    Centralized data-fetching helper. Dynamically builds SQL queries and joins multiple 
    tables based on shared conformed dimensions defined in TABLE_DIMENSIONS in base.py.
    """
    if isinstance(tables, str):
        if "," in tables:
            table_list = [t.strip() for t in tables.split(",")]
        else:
            table_list = [tables.strip()]
    else:
        table_list = [str(t).strip() for t in tables]
        
    table_list = list(dict.fromkeys(table_list))
    
    if columns:
        safe_cols = []
        for col in columns:
            clean_col = col.replace('"', '').replace("'", "").strip()
            if "." in clean_col or any(func in clean_col.upper() for func in ["SUM(", "AVG(", "COUNT(", "MIN(", "MAX("]):
                safe_cols.append(col)
            else:
                safe_cols.append(f'"{clean_col}"')
        columns_str = ", ".join(safe_cols)
    else:
        columns_str = "*"
        
    base_table = table_list[0]
    from_clause = f"FROM {base_table}"
    
    if len(table_list) > 1:
        joined_tables = [base_table]
        for next_table in table_list[1:]:
            join_condition = None
            for joined_t in joined_tables:
                try:
                    cond = get_join_clause(joined_t, next_table)
                    if cond:
                        join_condition = cond
                        break
                except ValueError:
                    continue
            
            if not join_condition:
                raise ValueError(
                    f"No shared dimensions found in TABLE_DIMENSIONS between '{next_table}' "
                    f"and currently joined tables ({joined_tables}). Please update base.py."
                )
            
            from_clause += f" INNER JOIN {next_table} ON {join_condition}"
            joined_tables.append(next_table)
            
    sql_query = f"SELECT {columns_str} {from_clause}"
    
    if where_clause:
        sql_query += f" WHERE {where_clause}"
        
    if group_by:
        safe_group = []
        for col in group_by:
            if "." not in col:
                clean_col = col.replace('"', '').replace("'", "").strip()
                safe_group.append(f'"{clean_col}"')
            else:
                safe_group.append(col)
        sql_query += f" GROUP BY {', '.join(safe_group)}"
        
    if random_order:
        sql_query += " ORDER BY RANDOM()"
    elif order_by:
        sql_query += f" ORDER BY {order_by}"
        
    if limit:
        sql_query += f" LIMIT {limit}"
        
    df = run_sql_query(sql_query)
    df.columns = [str(col).replace('"', '').replace("'", "").strip() for col in df.columns]
    return df


# ─── SQL Retreival Tool ───────────────────────────────────────────
@mlflow.trace(name="execute_sql_query")
def execute_sql_query_tool(sql_query: str) -> dict:
    """
    Executes an arbitrary PostgreSQL query and returns up to 100 preview rows as CSV text.
    Returns a dict with 'text' (summary + CSV preview) and 'data' (full DataFrame).
    """
    try:
        df = run_sql_query(sql_query)
        if df.empty:
            return {"text": "Error: Query executed successfully, but returned 0 rows.", "data": None}
        
        csv_text = df.head(100).to_csv(index=False)
        return {"text": f"Success. Showing top 100 rows:\n{csv_text}", "data": df}
        
    except Exception as e:
        error_msg = str(e)
        if "does not exist" in error_msg.lower() or "missingcolumn" in error_msg.lower():
             return {"text": f"Error executing SQL: {error_msg}\n\nSYSTEM HINT: Do not query information_schema. Look at the JSON schema in your system prompt for the correct exact column names (e.g., Activation_Year instead of Year).", "data": None}
        return {"text": f"Error executing SQL: {error_msg}", "data": None}


# ─── Data Transformation Tools ───────────────────────────────────────────
@mlflow.trace(name="join_dataframes_tool")
def join_dataframes_tool(
    left_dataframe_id: str, 
    right_dataframe_id: str, 
    how: str, 
    left_on: list, 
    right_on: list, 
    df_memory: DataFrameMemory = None
) -> dict:
    """
    Merges two in-memory dataframes and returns the resulting dataframe.
    """
    try:
        if not df_memory:
            return {"text": "Error: df_memory is not initialized.", "data": None}
            
        df_left = df_memory.get_df(left_dataframe_id)
        df_right = df_memory.get_df(right_dataframe_id)
        
        if df_left is None:
            return {"text": f"Error: Left DataFrame '{left_dataframe_id}' not found.", "data": None}
        if df_right is None:
            return {"text": f"Error: Right DataFrame '{right_dataframe_id}' not found.", "data": None}
            
        # Perform the merge
        merged_df = pd.merge(
            df_left, 
            df_right, 
            how=how, 
            left_on=left_on, 
            right_on=right_on,
            suffixes=('_left', '_right')
        )
        
        if merged_df.empty:
            return {"text": "Warning: The join executed successfully but resulted in 0 rows (no overlapping keys).", "data": merged_df}
            
        return {
            "text": f"Successfully joined DataFrames '{left_dataframe_id}' and '{right_dataframe_id}' ({how} join). Resulting shape: {merged_df.shape}.", 
            "data": merged_df
        }
        
    except Exception as e:
        return {"text": f"DataFrame Join Error: {str(e)}", "data": None}
    

@mlflow.trace(name="pivot_dataframe_tool")
def pivot_dataframe_tool(
    dataframe_id: str,
    index_columns: list,
    pivot_column: str,
    value_column: str,
    aggregation: str = "SUM",
    df_memory: DataFrameMemory = None
) -> dict:
    """
    Pivots an in-memory DataFrame from long to wide format.
    """
    try:
        if not df_memory:
            return {"text": "Error: df_memory is not initialized.", "data": None}

        df = df_memory.get_df(dataframe_id)
        if df is None:
            return {"text": f"Error: DataFrame '{dataframe_id}' not found in memory.", "data": None}

        agg_map = {"SUM": "sum", "AVG": "mean", "COUNT": "count", "MAX": "max", "MIN": "min"}
        agg_func = agg_map.get(aggregation.upper(), "sum")

        # Perform pivot table
        pivoted_df = pd.pivot_table(
            df,
            index=index_columns,
            columns=pivot_column,
            values=value_column,
            aggfunc=agg_func
        ).reset_index()

        # Flatten multi-index column names if created
        pivoted_df.columns.name = None

        if pivoted_df.empty:
            return {"text": "Warning: Pivot table created but resulted in an empty DataFrame.", "data": pivoted_df}

        return {
            "text": f"Successfully pivoted DataFrame '{dataframe_id}'. New shape: {pivoted_df.shape}.",
            "data": pivoted_df
        }

    except Exception as e:
        return {"text": f"Pivot Execution Error: {str(e)}", "data": None}


# ─── Ratio Analysis Tools ───────────────────────────────────────────
@mlflow.trace(name="calculate_unit_economics_tool")
def calculate_cpa_tool(marketing_where_clause: str = None, subscriber_where_clause: str = None) -> dict:
    """
    Joins monthly marketing spend against monthly activation counts to compute
    Total Marketing CPA, Residential CPA, and Residential Non-Caliber CPA.
    """
    try:
        # 1. Marketing Data
        df_mkt = _link_tables(
            tables='"sandbox"."dbs_marketing_sync"',
            columns=['"Year"', '"Month"', 'SUM("Amount") AS total_spend'], 
            where_clause=marketing_where_clause,
            group_by=['"Year"', '"Month"'],
            limit=None
        )
        
        if not df_mkt.empty:
            df_mkt.columns = [col.replace('"', '') for col in df_mkt.columns]

        # 2. Acquisition Data (Pivot long format to wide via conditional SQL aggregation)
        df_acq = _link_tables(
            tables='"sandbox"."subcount_data_synced"',
            columns=[
                '"Year"', 
                '"Month"', 
                'SUM(CASE WHEN "Metric" = \'Gross Adds\' THEN "Amount" ELSE 0 END) AS gross_adds',
                'SUM(CASE WHEN "Metric" = \'Commercial Activations\' THEN "Amount" ELSE 0 END) AS commercial_activations',
                'SUM(CASE WHEN "Metric" = \'Direct Activations\' THEN "Amount" ELSE 0 END) AS direct_activations',
                'SUM(CASE WHEN "Metric" = \'National Retail Activations\' THEN "Amount" ELSE 0 END) AS national_retail',
                'SUM(CASE WHEN "Metric" = \'Local Retail Activations\' THEN "Amount" ELSE 0 END) AS local_retail',
                'SUM(CASE WHEN "Metric" = \'Sales Partner Activations\' THEN "Amount" ELSE 0 END) AS sales_partner',
                'SUM(CASE WHEN "Metric" = \'Telco Activations\' THEN "Amount" ELSE 0 END) AS telco'
            ],
            where_clause=subscriber_where_clause,
            group_by=['"Year"', '"Month"'],
            limit=None
        )

        # 3. Clean and Merge
        for df_tmp in [df_mkt, df_acq]:
            df_tmp['Year'] = pd.to_numeric(df_tmp['Year'], errors='coerce')
            df_tmp['Month'] = pd.to_numeric(df_tmp['Month'], errors='coerce')

        df_merged = pd.merge(df_mkt, df_acq, on=['Year', 'Month'], how='inner')

        if df_merged.empty:
            return {"text": "Error: Could not calculate cpas. No overlapping months found.", "data": None}

        # 4. Calculate Defined Denominators
        df_merged['residential_activations'] = df_merged['gross_adds'] - df_merged['commercial_activations']
        df_merged['residential_non_caliber_activations'] = (
            df_merged['direct_activations'] + 
            df_merged['national_retail'] + 
            df_merged['local_retail'] + 
            df_merged['sales_partner'] + 
            df_merged['telco']
        )

        # 5. Calculate CPAs
        df_merged['total_marketing_cpa'] = df_merged['total_spend'] / df_merged['gross_adds']
        df_merged['residential_cpa'] = df_merged['total_spend'] / df_merged['residential_activations']
        df_merged['residential_non_caliber_cpa'] = df_merged['total_spend'] / df_merged['residential_non_caliber_activations']

        # 6. Formatting and Cleanup
        df_merged.replace([np.inf, -np.inf], np.nan, inplace=True)
        df_merged = df_merged.sort_values(by=['Year', 'Month'])

        df_merged['Date'] = pd.to_datetime(
            df_merged['Year'].astype(int).astype(str) + '-' + 
            df_merged['Month'].astype(int).astype(str) + '-01', 
            errors='coerce'
        )

        # Overall blended numbers for the text output
        overall_spend = df_merged['total_spend'].sum()
        overall_gross = df_merged['gross_adds'].sum()
        overall_res = df_merged['residential_activations'].sum()
        overall_res_nc = df_merged['residential_non_caliber_activations'].sum()

        blended_total = overall_spend / overall_gross if overall_gross > 0 else 0
        blended_res = overall_spend / overall_res if overall_res > 0 else 0
        blended_res_nc = overall_spend / overall_res_nc if overall_res_nc > 0 else 0
        
        text_output = (
            f"Unit Economics Summary:\n"
            f"  • Total Marketing Spend Analyzed: \\${overall_spend:,.2f}\n"
            f"  • Total Marketing CPA: \\${blended_total:,.2f}\n"
            f"  • Residential CPA: \\${blended_res:,.2f}\n"
            f"  • Residential Non-Caliber CPA: \\${blended_res_nc:,.2f}\n"
        )

        return {"text": text_output, "data": df_merged}

    except Exception as e:
        return {"text": f"Unit Economics Calculation Error: {e}", "data": None}


@mlflow.trace(name="calculate_ratio_tool")
def calculate_ratio_tool(
    numerator_column: str,
    denominator_column: str,
    numerator_table: str = None,
    numerator_dataframe_id: str = None,
    denominator_table: str = None,
    denominator_dataframe_id: str = None,
    where_clause: str = None,
    numerator_aggregation: str = "SUM",
    denominator_aggregation: str = "SUM",
    df_memory: DataFrameMemory = None
) -> dict:
    
    VALID_AGGS = {"SUM": "sum", "AVG": "mean", "COUNT": "count"}
    num_agg_sql = numerator_aggregation.upper() if numerator_aggregation.upper() in VALID_AGGS else "SUM"
    den_agg_sql = denominator_aggregation.upper() if denominator_aggregation.upper() in VALID_AGGS else "SUM"
    
    num_agg_pd = VALID_AGGS.get(num_agg_sql, "sum")
    den_agg_pd = VALID_AGGS.get(den_agg_sql, "sum")

    def _fetch_side(table_name, df_id, col_name, sql_agg, pd_agg):
        if df_id:
            if not df_memory:
                raise ValueError("df_memory is not initialized.")
            df = df_memory.get_df(df_id)
            if df is None:
                raise ValueError(f"DataFrame '{df_id}' not found in memory.")
            
            # Detect year/month columns across known standard names
            known_year_cols = {dims["year"] for dims in TABLE_DIMENSIONS.values()} | {"year", "Year"}
            known_month_cols = {dims["month"] for dims in TABLE_DIMENSIONS.values()} | {"month", "Month"}
            
            year_col = next((c for c in known_year_cols if c in df.columns), None)
            month_col = next((c for c in known_month_cols if c in df.columns), None)
            
            if not (year_col and month_col and col_name in df.columns):
                raise ValueError(f"Missing year/month dimensions or column '{col_name}' in DataFrame '{df_id}'.")
                
            # Perform Pandas GroupBy Aggregation
            df_agg = df.groupby([year_col, month_col], as_index=False)[col_name].agg(pd_agg)
            df_agg.rename(columns={year_col: "year", month_col: "month", col_name: "val"}, inplace=True)
            return df_agg
            
        elif table_name:
            dims = TABLE_DIMENSIONS.get(table_name.strip())
            if not dims:
                raise ValueError(f"Table '{table_name}' not found in TABLE_DIMENSIONS.")
                
            y_col, m_col = dims["year"], dims["month"]
            
            # Perform SQL Pushdown Aggregation
            df = _link_tables(
                tables=table_name,
                columns=[f'"{y_col}"', f'"{m_col}"', f'{sql_agg}("{col_name}") AS val'],
                where_clause=where_clause,
                group_by=[y_col, m_col],
                order_by=f'"{y_col}" ASC, "{m_col}" ASC',
                limit=None,
            )
            df.columns = [c.replace('"', '') for c in df.columns]
            df.rename(columns={y_col: "year", m_col: "month"}, inplace=True)
            return df
            
        else:
            raise ValueError(f"Must provide either a TABLE_NAME or dataframe_id for column '{col_name}'.")

    try:
        # Fetch and aggregate each side independently 
        df_num = _fetch_side(numerator_table, numerator_dataframe_id, numerator_column, num_agg_sql, num_agg_pd)
        df_den = _fetch_side(denominator_table, denominator_dataframe_id, denominator_column, den_agg_sql, den_agg_pd)

        # Standardize join keys
        for df_tmp in [df_num, df_den]:
            df_tmp["year"] = pd.to_numeric(df_tmp["year"], errors="coerce")
            df_tmp["month"] = pd.to_numeric(df_tmp["month"], errors="coerce")

        # Inner join on time dimensions
        df = pd.merge(df_num, df_den, on=["year", "month"], how="inner", suffixes=('_num', '_den'))

        if df.empty:
            return {"text": "Error: No overlapping year/month periods found between the datasets.", "data": None}

        # Calculate the ratio
        ratio_col = f"ratio_{numerator_column}_per_{denominator_column}"
        df[ratio_col] = df["val_num"] / df["val_den"]
        df.replace([np.inf, -np.inf], np.nan, inplace=True)

        # Clean up column names for output
        df.rename(columns={"val_num": numerator_column, "val_den": denominator_column}, inplace=True)
        df = df.sort_values(by=["year", "month"]).reset_index(drop=True)

        avg_ratio = df[ratio_col].mean()
        min_ratio = df[ratio_col].min()
        max_ratio = df[ratio_col].max()

        text_output = (
            f"Monthly Ratio — {numerator_column} / {denominator_column}:\n"
            f"  • Periods computed: {len(df)}\n"
            f"  • Average ratio: {avg_ratio:,.4f}\n"
            f"  • Min: {min_ratio:,.4f}  |  Max: {max_ratio:,.4f}\n"
        )

        return {"text": text_output, "data": df}

    except Exception as e:
        return {"text": f"Ratio Calculation Error: {e}", "data": None}
    

# ─── Custom Python Tool ───────────────────────────────────────────
@mlflow.trace(name="execute_python_tool")
def execute_python_tool(
    code: str, 
    TABLE_NAME: Optional[Union[str, List[str]]] = None,
    dataframe_id: Optional[str] = None,
    df_memory: DataFrameMemory = None
) -> Dict[str, Any]:
    """
    Executes LLM-generated Python code in a restricted sandbox with access to
    pre-loaded pandas DataFrames (df), numpy (np), and pandas (pd).
    """
    try:
        # --- 1. Data Loading (Unchanged) ---
        if dataframe_id:
            df = df_memory.get_df(dataframe_id) if df_memory else None
            if df is None:
                return {"text": f"Error: No DataFrame found for ID '{dataframe_id}'.", "data": None}
        elif TABLE_NAME:
            if isinstance(TABLE_NAME, str):
                table_list = [t.strip() for t in TABLE_NAME.split(",")] if "," in TABLE_NAME else [TABLE_NAME]
            else:
                table_list = list(TABLE_NAME)

            if len(table_list) == 1:
                df = _link_tables(table_list[0], limit=100000)
            else:
                df = [_link_tables(t, limit=100000) for t in table_list]
        else:
            return {"text": "Error: Must provide either TABLE_NAME or dataframe_id.", "data": None}
            
        # --- 2. NEW AST Security Check ---
        try:
            validate_safe_python_code(code)
        except SecurityViolationError as e:
            return {"text": f"Error: {str(e)} Execution blocked for security.", "data": None}
                
        # --- 3. Execution Engine ---
        # Add __builtins__ directly to your environment dictionary
        execution_env = {
            '__builtins__': __builtins__,
            'df': df.copy() if isinstance(df, pd.DataFrame) else df,
            'pd': pd,
            'np': np,
            'result_df': None,
            'result_text': None
        }
        
        stdout_buffer = io.StringIO()
        with contextlib.redirect_stdout(stdout_buffer):
            # Pass the dictionary once; it acts as both globals and locals
            exec(code, execution_env)
            
        output = stdout_buffer.getvalue()
        
        final_text = "Python Execution Successful.\n"
        if output:
            final_text += f"Console Output:\n{output}\n"
            
        # Make sure to retrieve your results from the new execution_env dictionary
        if execution_env.get('result_text'):
            final_text += f"Model Result Text:\n{execution_env['result_text']}\n"
            
        return {
            "text": final_text, 
            "data": execution_env.get('result_df') if isinstance(execution_env.get('result_df'), pd.DataFrame) else (execution_env.get('df') if isinstance(execution_env.get('df'), pd.DataFrame) else None)
        }
        
    except Exception as e:
        return {"text": f"Python Execution Error: {e}", "data": None}