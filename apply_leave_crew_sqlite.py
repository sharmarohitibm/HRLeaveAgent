import os
from dotenv import load_dotenv
from textwrap import dedent
from typing import List

from crewai import LLM, Task, Agent, Crew, Process
from crewai_tools import NL2SQLTool
from crewai.tools import tool
from langchain_community.tools.sql_database.tool import (
    InfoSQLDatabaseTool,
    ListSQLDatabaseTool,
    QuerySQLCheckerTool,
    QuerySQLDataBaseTool,
)
from langchain_community.utilities.sql_database import SQLDatabase

# === Load environment variables ===
load_dotenv()
import pandas as pd
import sqlite3

# === Step 1: Load CSV data ===
csv_file = "employees_structure.csv"  # your input file
df = pd.read_csv(csv_file)

# === Step 2: Connect to SQLite (creates sales.db) ===
conn = sqlite3.connect('sales.db')

# === Step 3: Save DataFrame to SQLite table ===
df.to_sql('leave_balance', conn, if_exists='replace', index=False)

# === Step 4: Done ===
# print("Successfully created sales.db with table 'leave_balance'!")

conn.close()

# === Database connection using SQLite ===
db_url = os.environ.get("DB_URI", "sqlite:///sales.db")
db = SQLDatabase.from_uri(db_url)

# === NL2SQL Tool setup ===
nl2sql = NL2SQLTool(db=db)

# === LLM Setup ===
api_key = os.environ["WATSONX_API_KEY"]
llm = LLM(
    api_key=api_key,
    model="watsonx/meta-llama/llama-3-70b-instruct",
    params={
        "decoding_method": "greedy",
        "max_new_tokens": 15000,
        "temperature": 0,
        "repetition_penalty": 1.05
    }
)

# === Tools ===
@tool("list_tables")
def list_tables() -> str:
    """List the available tables in the database."""
    return ListSQLDatabaseTool(db=db).invoke("")

@tool("tables_schema")
def tables_schema(tables: str) -> str:
    """Describe table schema and provide sample rows."""
    tool_instance = InfoSQLDatabaseTool(db=db)
    return tool_instance.invoke(tables)

@tool("execute_sql")
def execute_sql(sql_query: str) -> str:
    """Execute a SQL query against the database and return results."""
    return QuerySQLDataBaseTool(db=db).invoke(sql_query)

# Skip check_sql tool (because it depended on watsonx_llm, and SQLite does not need complex query checker)

# === Function to handle leave query ===
def queryleave(question: str) -> str:
    sql_dev = Agent(
        role="Senior Database Developer",
        goal="Construct and execute SQL queries based on a request",
        backstory=dedent(
            """
            You are an experienced database engineer who is master at creating efficient and complex SQL queries.
            Use `list_tables` to find available tables.
            Use `tables_schema` to understand the metadata.
            Use `execute_sql` to run the query.
            """
        ),
        llm=llm,
        tools=[list_tables, tables_schema, execute_sql],  # removed check_sql
        allow_delegation=False,
    )

    extract_data = Task(
        description=f"Extract data required for the query: {question}.",
        expected_output="Database result for the query",
        agent=sql_dev,
    )

    data_analyst = Agent(
        role="Senior Data Analyst",
        goal="Analyze the extracted database result and answer the question concisely.",
        backstory=dedent(
            """
            You are a skilled data analyst experienced in understanding and summarizing structured query outputs.
            """
        ),
        llm=llm,
        allow_delegation=False,
    )

    analyze_data = Task(
        description=f"Answer the question: {question}.",
        expected_output="Concise answer in summary text form.",
        agent=data_analyst,
        context=[extract_data],
    )

    crew = Crew(
        agents=[sql_dev, data_analyst],
        tasks=[extract_data, analyze_data],
        memory=False,
        verbose=False
    )

    result = crew.kickoff()

    if isinstance(result, dict) and "tasks_output" in result:
        return result["tasks_output"][-1].get("raw", result)
    return result.raw
