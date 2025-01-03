from dataclasses import dataclass
from typing import Annotated
import pandas as pd
from pydantic import Field
from pydantic_ai import Agent, ModelRetry, RunContext
from pydantic_ai.models.ollama import OllamaModel


@dataclass
class Deps:
    """workflow dependencies"""

    df: pd.DataFrame


model = OllamaModel(
    "llama3.2",
)

dataframe_agent = Agent(
    model,
    system_prompt="""
    You are an AI assistant that helps extract information from a pandas DataFrame.
    ALWAYS CHECK THE COLUMN NAME FIRST.
    If asked about an specific column, be sure to check the column names first.
    Write 'unable to answer the question' when not able to answer.
    Think step by step. Do not use backtick.
    """,
    deps_type=Deps,
)


@dataframe_agent.tool(retries=5)
async def df_query(
    ctx: RunContext[Deps],
    query: Annotated[str, Field(description="query to use inside `pandas.eval")],
) -> str:
    """A tool for running queries on the `pandas.DataFrame` that is given on the dependencies.
    Use this tool to interact with the DataFrame.

    `query` will be executed using `pd.eval(query, target=df)`, so it must contain syntax compatible with
    `pandas.eval`.

    """
    print("using df_query tool")
    print(f"Running query: `{query}`")
    try:
        response = str(pd.eval(query, target=ctx.deps.df))
        return response

    except Exception as e:
        print(f"query: `{query}` is not a valid query. Reason: `{e}`")
        raise ModelRetry(f"query: `{query}` is not a valid query. Reason: `{e}`") from e
