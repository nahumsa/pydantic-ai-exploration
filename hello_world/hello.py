from pydantic_ai import Agent
from pydantic_ai.models.ollama import OllamaModel


model = OllamaModel(
    "llama3.2",
)

agent = Agent(
    model,
    system_prompt="Be concise, reply with one sentence.",
)

result = agent.run_sync('Where does "hello world" come from?')
print(result.data)
