# Pandas Runner

This example is based on: <https://github.com/NirDiamant/GenAI_Agents/blob/main/all_agents_tutorials/simple_data_analysis_agent_notebook-pydanticai.ipynb>

Even though I followed the example, I wasn't able to reproduce it using Llama3.2.
Thus I added a tool to find the name of the columns deterministically, since Llama3.2 was
not able to do so by itself. After adding the columns, I was able to reproduce it.

Adding the columns was pretty straight forward following the pydantic-ai documentation,
and I really think it showcase the power of dynamic system prompts.
