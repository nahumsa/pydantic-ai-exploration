from pydantic_ai import Agent
from openai import AsyncOpenAI
from pydantic_ai.models.openai import OpenAIModel


client = AsyncOpenAI(
    base_url="http://localhost:11434/v1",
    api_key="your-api-key",
)

model = OpenAIModel(
    "qwen2.5-coder:3b",  # type: ignore
    openai_client=client,
)

agent = Agent(
    model,
    system_prompt="Be concise, reply with one sentence.",
)

result = agent.run_sync('Where does "hello world" come from?')
print(result.data)
print(result.cost())
