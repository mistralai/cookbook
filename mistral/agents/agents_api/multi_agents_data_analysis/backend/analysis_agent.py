"""
Data Analysis Agent Module

This module contains the AnalysisAgent class which handles natural language to SQL conversion
and query execution for data analysis tasks.
"""

import os
import sqlite3
from typing import Tuple, List, AsyncGenerator
from mistralai import Mistral, MessageOutputEvent
from utils.db_manager import load_excel_to_sqlite


DATA_ANALYSIS_PROMPT = """
You are an advanced data analysis agent with a strong attention to detail. You are given a question and database schema in metadata. Your task is to write a SQL query to answer the question.

Instructions:
    Given an input question, output a syntactically correct SQLite query to run.
    When generating the query:
    1. Always limit your query to at most 10 results, unless the user specifies a specific number of examples they wish to obtain.
    2. You can order the results by a relevant column to return the most interesting examples in the database.
    3. Never query for all the columns from a specific table, only ask for the relevant columns given the question.
    4. If you get an error while executing a query, rewrite the query and try again.
    5. If you get an empty result set, you should try to rewrite the query to get a non-empty result set.
    6. When using strftime or other SQL functions with % symbols, escape them properly in JSON by using double backslashes: strftime('%%Y-%%m', d.date)
    7. NEVER NEVER make stuff up if you don't have enough information to answer the query... just say you don't have enough information.
    8. Really think about whether the exact field or table being asked actually exists and is given in the metadata. Only can access ones given with exact same wording.
    9. DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the database.

    Table Metadata:
    The table you can query is provided below. The table has the following columns and metadata:
    {metadata_description}

    Note:
    When ordering by a calculated column, repeat the calculation in the ORDER BY clause.

    Examples:
    Example 1:
    Query: Which customers have dormant accounts?
    SQL Query: SELECT customer_name, account_name, account_status
    FROM accounts
    WHERE is_dormant = 1
    ORDER BY customer_name
    LIMIT 10

    Example 2:
    Query: Show customer names with their account balances from the most recent date
    SQL Query: SELECT a.customer_name, d.available_balance, d.date
    FROM accounts a
    JOIN daily_metrics d ON a.account_number = d.account_number
    WHERE d.date = (SELECT MAX(date) FROM daily_metrics)
    ORDER BY d.available_balance DESC
    LIMIT 10

    Example 3:
    Query: Show accounts with the highest balance changes in January 2025
    SQL Query: SELECT a.customer_name, d.account_number, d.net_change, d.date
    FROM accounts a
    JOIN daily_metrics d ON a.account_number = d.account_number
    WHERE ABS(d.net_change) > 0
    AND d.date >= '2025-01-01' AND d.date < '2025-02-01'
    ORDER BY ABS(d.net_change) DESC
    LIMIT 10

    Natural Language Query: {input}

    Output:
    Return ONLY the JSON format as {{"SQL_query": "<SQL_QUERY>"}} with no markdown formatting or code blocks.
"""


class AnalysisAgent:
    def __init__(self, xlsx_path: str) -> None:
        self.conn = load_excel_to_sqlite(xlsx_path)
        self.client = Mistral(api_key=os.getenv("MISTRAL_API_KEY"))
        self.metadata = self._build_template()
        self.completion_args = {"response_format": {"type": "json_object"}}
        self.agent = self.client.beta.agents.create(
            name="data_analysis_agent",
            model="mistral-medium-latest",
            instructions=DATA_ANALYSIS_PROMPT.format(
                metadata_description=self.metadata, input="{input}"
            ),
            description="Data Analysis Agent",
            completion_args={"response_format": {"type": "json_object"}},
        )

    def _get_table_metadata(self, table_name: str) -> list[str]:
        """Get metadata for a table in a formatted list."""
        cursor = self.conn.cursor()
        cursor.execute(f"PRAGMA table_info({table_name})")
        columns = cursor.fetchall()
        metadata = []
        for col_id, col_name, col_type, notnull, dflt_value, pk in columns:
            col_def = f"{col_name} ({col_type or 'UNKNOWN'}"
            col_def += f", {'NOT NULL' if notnull else 'NULL'}"
            if pk:
                col_def += ", PRIMARY KEY"
            col_def += ")"
            metadata.append(col_def)
        return metadata

    def _build_template(self) -> str:
        """Build the metadata template for the LLM prompt."""
        tables = ["accounts", "daily_metrics"]
        metadata_blocks = []
        for table in tables:
            schema_lines = self._get_table_metadata(table)
            metadata_blocks.append(f"### Table: {table}\n" + "\n".join(schema_lines))
        return "\n\n".join(metadata_blocks)

    async def generate_sql(self, query: str) -> AsyncGenerator[str, None]:
        """Generate SQL based on natural language query."""
        response = self.client.beta.conversations.start_stream(
            agent_id=self.agent.id,
            inputs=query,
            stream=True,
        )

        with response as event_stream:
            for event in event_stream:
                if isinstance(event.data, MessageOutputEvent):
                    content = event.data.content
                    # Ensure we yield a string
                    if isinstance(content, list):
                        # If it's a list of chunks, extract the text content
                        yield "".join(
                            chunk.get("text", "")
                            if isinstance(chunk, dict)
                            else str(chunk)
                            for chunk in content
                        )
                    else:
                        yield str(content)

    def execute_sql_query(self, sql_query: str) -> Tuple[List[str], List[Tuple]]:
        """Execute a SQL query and return (columns, results)."""
        try:
            cursor = self.conn.cursor()
            cursor.execute(sql_query)
            results = cursor.fetchall()
            columns = [desc[0] for desc in cursor.description]
            return columns, results
        except sqlite3.OperationalError as e:
            print(f"OperationalError: {e}")
            # List tables for debugging
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            print("Available tables:", cursor.fetchall())
            raise
