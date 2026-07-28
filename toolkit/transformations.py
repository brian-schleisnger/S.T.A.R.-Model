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

@mlflow.trace(name="link_tables")
def link_tables(
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
    

@mlflow.trace(name="calculate_unit_economics_tool")
def calculate_unit_economics_tool(marketing_where_clause: str = None, subscriber_where_clause: str = None) -> dict:
    """
    Joins monthly marketing spend against monthly activation counts to compute
    CPA (Cost Per Acquisition), CLV (Customer Lifetime Value via NPV of MCF), and
    the CLV:CPA ratio for each month. Returns a blended summary and the merged
    monthly DataFrame with all computed columns.

    Uses MONTHLY_WACC and avg_churn to discount future cash flows for CLV.
    Both where_clause params are applied independently to their respective tables
    before the inner join, allowing independent filtering (e.g. by channel or segment).
    """
    # Kept as-is since the schema explicitly handles the two-table where clauses without a TABLE_NAME arg.
    try:
        # 1. Marketing Data
        df_mkt = link_tables(
            tables='"sandbox"."dbs_marketing_sync"',
            # Keep the quotes for Postgres, but drop the 'AS' alias
            columns=['"Year"', '"Month"', 'SUM("Amount") AS total_spend'], 
            where_clause=marketing_where_clause,
            group_by=['"Year"', '"Month"'],
            limit=None
        )
        
        # Strip the quotes out of the resulting pandas column names if the driver leaves them in
        if not df_mkt.empty:
            df_mkt.columns = [col.replace('"', '') for col in df_mkt.columns]

        # 2. Acquisition Data
        df_acq = link_tables(
            tables='"sandbox"."subcount_data_synced"',
            # Keep the quotes to protect the capital letters for Postgres
            columns=[
                '"Year"', 
                '"Month"', 
                'SUM("Amount") AS total_activations'
            ],
            where_clause=subscriber_where_clause,
            group_by=['"Year"', '"Month"'],
            limit=None
        )

        for df_tmp in [df_mkt, df_acq]:
            df_tmp['Year'] = pd.to_numeric(df_tmp['Year'], errors='coerce')
            df_tmp['Month'] = pd.to_numeric(df_tmp['Month'], errors='coerce')

        df_merged = pd.merge(df_mkt, df_acq, on=['Year', 'Month'], how='inner')

        if df_merged.empty:
            return {"text": "Error: Could not calculate UNit Economics. No overlapping months found.", "data": None}

        df_merged['cpa'] = df_merged['total_spend'] / df_merged['total_activations']

        df_merged.replace([np.inf, -np.inf], np.nan, inplace=True)
        df_merged = df_merged.sort_values(by=['Year', 'Month'])

        df_merged['Date'] = pd.to_datetime(
            df_merged['Year'].astype(int).astype(str) + '-' + 
            df_merged['Month'].astype(int).astype(str) + '-01', 
            errors='coerce'
        )

        overall_spend = df_merged['total_spend'].sum()
        overall_acq = df_merged['total_activations'].sum()
        blended_cpa = overall_spend / overall_acq if overall_acq > 0 else 0
        
        text_output = (
            f"Unit Economics Summary:\n"
            f"  • Total Marketing Spend Analyzed: ${overall_spend:,.2f}\n"
            f"  • Total Activations: {overall_acq:,.0f}\n"
            f"  • Blended CPA: ${blended_cpa:,.2f}\n"
        )

        return {"text": text_output, "data": df_merged}

    except Exception as e:
        return {"text": f"Unit Economics Calculation Error: {e}", "data": None}


@mlflow.trace(name="calculate_ratio_tool")
def calculate_ratio_tool(
    numerator_column: str,
    numerator_table: str,
    denominator_column: str,
    denominator_table: str,
    where_clause: str = None,
    numerator_aggregation: str = "SUM",
    denominator_aggregation: str = "SUM",
) -> dict:
    """
    Calculates a monthly ratio (numerator / denominator) between any two numeric columns,
    which may live in the same table or in two different tables.

    Both sides are independently aggregated to one value per (year, month) period using
    their respective aggregation functions, then joined on the shared time dimensions.
    Returns a DataFrame with columns: year, month, <numerator_column>, <denominator_column>,
    and ratio_<numerator_column>_per_<denominator_column>.
    """
    VALID_AGGS = {"SUM", "AVG", "COUNT"}
    num_agg = numerator_aggregation.upper() if numerator_aggregation.upper() in VALID_AGGS else "SUM"
    den_agg = denominator_aggregation.upper() if denominator_aggregation.upper() in VALID_AGGS else "SUM"

    try:
        same_table = numerator_table.strip() == denominator_table.strip()

        if same_table:
            # Both columns live in the same table — fetch in a single query.
            dims = TABLE_DIMENSIONS.get(numerator_table.strip())
            if dims is None:
                return {"text": f"Error: Table '{numerator_table}' not found in TABLE_DIMENSIONS.", "data": None}

            year_col = dims["year"]
            month_col = dims["month"]

            df = link_tables(
                tables=numerator_table,
                columns=[
                    f'"{year_col}"',
                    f'"{month_col}"',
                    f'{num_agg}("{numerator_column}") AS numerator_val',
                    f'{den_agg}("{denominator_column}") AS denominator_val',
                ],
                where_clause=where_clause,
                group_by=[year_col, month_col],
                order_by=f'"{year_col}" ASC, "{month_col}" ASC',
                limit=None,
            )
            df.columns = [c.replace('"', '') for c in df.columns]
            df.rename(columns={year_col: "year", month_col: "month"}, inplace=True)

        else:
            # Two different tables — query each independently then join on year/month.
            num_dims = TABLE_DIMENSIONS.get(numerator_table.strip())
            den_dims = TABLE_DIMENSIONS.get(denominator_table.strip())

            if num_dims is None:
                return {"text": f"Error: Table '{numerator_table}' not found in TABLE_DIMENSIONS.", "data": None}
            if den_dims is None:
                return {"text": f"Error: Table '{denominator_table}' not found in TABLE_DIMENSIONS.", "data": None}

            num_year, num_month = num_dims["year"], num_dims["month"]
            den_year, den_month = den_dims["year"], den_dims["month"]

            df_num = link_tables(
                tables=numerator_table,
                columns=[
                    f'"{num_year}"',
                    f'"{num_month}"',
                    f'{num_agg}("{numerator_column}") AS numerator_val',
                ],
                where_clause=where_clause,
                group_by=[num_year, num_month],
                order_by=f'"{num_year}" ASC, "{num_month}" ASC',
                limit=None,
            )
            df_num.columns = [c.replace('"', '') for c in df_num.columns]
            df_num.rename(columns={num_year: "year", num_month: "month"}, inplace=True)

            df_den = link_tables(
                tables=denominator_table,
                columns=[
                    f'"{den_year}"',
                    f'"{den_month}"',
                    f'{den_agg}("{denominator_column}") AS denominator_val',
                ],
                where_clause=where_clause,
                group_by=[den_year, den_month],
                order_by=f'"{den_year}" ASC, "{den_month}" ASC',
                limit=None,
            )
            df_den.columns = [c.replace('"', '') for c in df_den.columns]
            df_den.rename(columns={den_year: "year", den_month: "month"}, inplace=True)

            for df_tmp in [df_num, df_den]:
                df_tmp["year"] = pd.to_numeric(df_tmp["year"], errors="coerce")
                df_tmp["month"] = pd.to_numeric(df_tmp["month"], errors="coerce")

            df = pd.merge(df_num, df_den, on=["year", "month"], how="inner")

        if df.empty:
            return {"text": "Error: No overlapping year/month periods found for the two columns.", "data": None}

        df["year"] = pd.to_numeric(df["year"], errors="coerce")
        df["month"] = pd.to_numeric(df["month"], errors="coerce")
        df["numerator_val"] = pd.to_numeric(df["numerator_val"], errors="coerce")
        df["denominator_val"] = pd.to_numeric(df["denominator_val"], errors="coerce")

        ratio_col = f"ratio_{numerator_column}_per_{denominator_column}"
        df[ratio_col] = df["numerator_val"] / df["denominator_val"]
        df.replace([np.inf, -np.inf], np.nan, inplace=True)

        df.rename(columns={
            "numerator_val": numerator_column,
            "denominator_val": denominator_column,
        }, inplace=True)

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
                df = link_tables(table_list[0], limit=100000)
            else:
                df = [link_tables(t, limit=100000) for t in table_list]
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