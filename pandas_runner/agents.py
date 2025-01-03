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

    async def system_prompt_factory(self) -> str:
        return f"""
            You are an AI assistant that helps extract information from a pandas DataFrame.
            you have the following columns on the dataframe {self.df.columns.tolist()}.
            Write 'unable to answer the question' when not able to answer.
            Think step by step. Do not use backtick.
            """


model = OllamaModel(
    "llama3.2",
)

dataframe_agent = Agent(
    model,
    deps_type=Deps,
)


@dataframe_agent.system_prompt
async def get_system_prompt(ctx: RunContext[Deps]) -> str:
    return await ctx.deps.system_prompt_factory()


@dataframe_agent.tool(retries=5)
async def df_eval(ctx: RunContext[Deps], query: str) -> str:
    """A tool for running queries on the `pandas.DataFrame` that is given on the dependencies.
    Use this tool to interact with the DataFrame.

    `query` will be executed using `pd.eval(query, target=df)`, so it must contain syntax compatible with
    `pd.eval`.
    """
    print("using df_query tool")
    print(f"Running query: `{query}`")
    try:
        df = ctx.deps.df
        response = str(pd.eval(query, target=df))
        return response

    except Exception as e:
        print(f"query: `{query}` is not a valid query. Reason: `{e}`")
        raise ModelRetry(f"query: `{query}` is not a valid query. Reason: `{e}`") from e
