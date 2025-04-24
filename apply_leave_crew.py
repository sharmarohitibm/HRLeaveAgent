import os
from dotenv import load_dotenv
from crewai import LLM, Task, Agent, Crew, Process
from typing import List

from crewai_tools import NL2SQLTool
from crewai.tools import tool
from crewai import Agent, Crew, Task, Process
import os

from textwrap import dedent

from langchain_community.tools.sql_database.tool import (
    InfoSQLDatabaseTool,
    ListSQLDatabaseTool,
    QuerySQLCheckerTool,
    QuerySQLDataBaseTool,
)
from langchain_community.utilities.sql_database import SQLDatabase

load_dotenv()
# psycopg2 was installed to run this example with PostgreSQL
# nl2sql = NL2SQLTool(db_uri="sqlite:///sales.db")
api_key=os.environ["WATSONX_API_KEY"]
#db_url=os.environ["DB_URI"]
# db = SQLDatabase.from_uri("postgresql://rohit@localhost:5432/leavedb")

# DB_URL = "localhost"
DB_PORT = "5432"
DB_NAME = "leavedb"
DB_USER = "postgres"
DB_PASSWORD = "rohit"
DB_URL = os.getenv("DB_URL", "host.docker.internal")
nl2sql = NL2SQLTool(db_uri=f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_URL}:{DB_PORT}/{DB_NAME}")


db = SQLDatabase.from_uri(f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_URL}:{DB_PORT}/{DB_NAME}")
# print("HERE")
# tables = ListSQLDatabaseTool(db=db).invoke("")
# print(tables)

# print("HERE2")
# tool = InfoSQLDatabaseTool(db=db)
# print(tool.invoke(tables))
# #nl2sql = NL2SQLTool(db_uri= db_url)

from ibm_watsonx_ai.foundation_models.schema import TextChatParameters

parameters = TextChatParameters(
    max_tokens=1000,
)
url="https://us-south.ml.cloud.ibm.com",
project_id="cbc99383-62c7-4c52-8214-cb61cefbac15",
# from langchain_ibm import ChatWatsonx
# watsonx_llm = ChatWatsonx(
#     model_id="meta-llama/llama-3-3-70b-instruct",
#     url="https://us-south.ml.cloud.ibm.com",
#     apikey="UGZZNppvxMnbsXge01pUHyC5EtcwePoMqmrbUKLOE6gE",
#     project_id="cbc99383-62c7-4c52-8214-cb61cefbac15",
#     params=parameters,
# )

# sql_query = "SELECT annual_leave, sick_leave FROM leave_balance where employee_name = 'Khai Wei'"
# print("HERE3")
# print(QuerySQLCheckerTool(db=db, llm=watsonx_llm).invoke({"query": sql_query}))

load_dotenv()



# === LLM Setup ===
# Initialize LLM
llm = LLM(
    api_key=api_key,
    model="watsonx/meta-llama/llama-3-3-70b-instruct",
    params={
        "decoding_method": "greedy",
        "max_new_tokens": 15000,
        "temperature": 0,
        "repetition_penalty": 1.05
    }
)

@tool("list_tables")
def list_tables() -> str:
    """List the available tables in the database"""
    return ListSQLDatabaseTool(db=db).invoke("")

@tool("tables_schema")
def tables_schema(tables: str) -> str:
    """
    Input is a comma-separated list of tables, output is the schema and sample rows
    for those tables. Be sure that the tables actually exist by calling `list_tables` first!
    Example Input: table1, table2, table3
    """
    tool = InfoSQLDatabaseTool(db=db)
    return tool.invoke(tables)

@tool("execute_sql")
def execute_sql(sql_query: str) -> str:
    """Execute a SQL query against the database. Returns the result"""
    return QuerySQLDataBaseTool(db=db).invoke(sql_query)

@tool("check_sql")
def check_sql(sql_query: str) -> str:
    """
    Use this tool to double check if your query is correct before executing it. Always use this
    tool before executing a query with `execute_sql`.
    """
    return QuerySQLCheckerTool(db=db, llm=watsonx_llm).invoke({"query": sql_query})

# === Function to Generate  ===
def queryleave(question: str) -> str:

    sql_dev = Agent(
        role="Senior Database Developer",
        goal="Construct and execute SQL queries based on a request",
        backstory=dedent(
            """
            You are an experienced database engineer who is master at creating efficient and complex SQL queries.
            You have a deep understanding of how different databases work and how to optimize queries.
            Use the `list_tables` to find available tables.
            Use the `tables_schema` to understand the metadata for the tables.
            Use the `check_sql` to check your queries for correctness.
            Use the `execute_sql` to execute queries against the database.
        """
        ),
        llm=llm,
        tools=[list_tables, tables_schema, execute_sql, check_sql],
        allow_delegation=False,
    )

    extract_data = Task(
        description=(f"Extract data that is required for the query {question}."),
        expected_output="Database result for the query",
        agent=sql_dev,
    )

    data_analyst = Agent(
        role="Senior Data Analyst",
        goal="You receive data from the database developer and answer the question",
        backstory=dedent(
            """
            You have deep experience with analyzing datasets return from database.
        """
        ),
        llm=llm,
        allow_delegation=False,
    )


    analyze_data = Task(
        description=(f"Answer the question {question}."),
        expected_output="Concise answer in the form of summary text.",
        agent=data_analyst,
        context=[extract_data],
    )


    # === Assemble the Crew ===
    crew = Crew(
        agents=[sql_dev, data_analyst],
        tasks=[extract_data, analyze_data],
        memory=False,
        verbose=False
    )

    # === Run the Crew ===
    result = crew.kickoff()
    if isinstance(result, dict) and "tasks_output" in result:
        # Get the final task's raw output
        return result["tasks_output"][-1].get("raw", result)
    return result.raw
    
