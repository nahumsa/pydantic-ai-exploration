import pandas as pd
from pprint import pprint

from data import generate_df
from agents import Deps, dataframe_agent


def ask_agent(question: str, input_df: pd.DataFrame):
    """Function to ask questions to the agent and display the response"""
    deps = Deps(df=input_df)
    pprint(f"Question: {question}", indent=4)
    response = dataframe_agent.run_sync(
        user_prompt=question,
        deps=deps,
    )
    parsed_response = response.new_messages()[-1].parts[0].content
    print(f"Answer: {parsed_response}")
    print(response.new_messages())


if __name__ == "__main__":
    # Example questions
    df = generate_df()
    # ask_agent("What are the column names in this dataset?", input_df=df)
    # ask_agent("How many rows are in this dataset?", input_df=df)
    ask_agent("What is the average price of cars sold?", input_df=df)
