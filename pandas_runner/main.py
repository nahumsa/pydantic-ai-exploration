from dataclasses import dataclass
import pandas as pd
from pydantic_ai import Agent, ModelRetry, RunContext
from pydantic_ai.models.ollama import OllamaModel

from data import generate_df


@dataclass
class Deps:
    """The only dependency we need is the DataFrame we'll be working with."""

    df: pd.DataFrame


model = OllamaModel(
    "llama3.2",
)

agent = Agent(model, deps_type=Deps, retries=10)


@agent.system_prompt
async def system_prompt() -> str:
    return """
    You are an AI assistant that helps extract information from a pandas DataFrame.
    If asked about columns, be sure to check the column names first.
    Be concise in your answers.
    """


@agent.tool
async def df_query(ctx: RunContext[Deps], query: str) -> str:
    """A tool for running queries on the `pandas.DataFrame`. Use this tool to interact with the DataFrame.

    `query` will be executed using `pd.eval(query, target=df)`, so it must contain syntax compatible with
    `pandas.eval`.
    """

    # Print the query for debugging purposes and fun :)
    print(f"Running query: `{query}`")
    try:
        # Execute the query using `pd.eval` and return the result as a string (must be serializable).
        return str(pd.eval(query, target=ctx.deps.df))
    except Exception as e:
        #  On error, raise a `ModelRetry` exception with feedback for the agent.
        raise ModelRetry(f"query: `{query}` is not a valid query. Reason: `{e}`") from e


def ask_agent(question: str, input_df: pd.DataFrame):
    """Function to ask questions to the agent and display the response"""
    deps = Deps(df=input_df)
    print(f"Question: {question}")
    response = agent.run_sync(question, deps=deps)
    print(f"Answer: {response.new_messages()[-1]}")
    print("---")


# Example questions
df = generate_df()
ask_agent("What are the column names in this dataset?", input_df=df)
ask_agent("How many rows are in this dataset?", input_df=df)
ask_agent("What is the average price of cars sold?", input_df=df)
