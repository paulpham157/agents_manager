# Agents Manager

[![PyPI version](https://badge.fury.io/py/agents-manager.svg)](https://badge.fury.io/py/agents-manager)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
![Python Version](https://img.shields.io/badge/python-3.7%2B-blue)
[![Downloads](https://img.shields.io/pypi/dm/agents-manager.svg)](https://pypi.org/project/agents-manager/)

A lightweight Python package for managing multi-agent orchestration. Easily define agents with custom instructions, tools, and models, and orchestrate their interactions seamlessly. Perfect for building modular, collaborative AI systems.

## Features

- Define agents with specific roles and instructions
- Assign models to agents (e.g., OpenAI models)
- Equip agents with tools for performing tasks
- Seamlessly orchestrate interactions between multiple agents

## Supported Models

- OpenAI
- Grok
- DeepSeek
- Anthropic
- Llama

```python

from agents_manager.models import OpenAi, Grok, DeepSeek, Anthropic, Llama

```

## Installation

Install the package via pip:

```sh
pip install agents-manager
```

## Quick Start

```python
from agents_manager import Agent, AgentManager
from agents_manager.models import OpenAi, Grok, DeepSeek, Anthropic, Llama

from dotenv import load_dotenv

load_dotenv()

# Define the OpenAi model
model = OpenAi(name="gpt-4o-mini")


# Define the Grok model
# model = Grok(name="grok-2-latest")


# Define the DeepSeek model
# model = DeepSeek(name="deepseek-chat")


# Define the Anthropic model
# model = Anthropic(
#         name="claude-3-5-sonnet-20241022",
#         max_tokens= 1024,
#         stream=True,
#     )

# Define the Llama model
# model = Llama(name="llama3.1-70b")

def multiply(a: int, b: int) -> int:
    """
    Multiply two numbers.
    """
    return a * b


def transfer_to_agent_3_for_math_calculation() -> Agent:
    """
    Transfer to agent 3 for math calculation.
    """
    return agent3


def transfer_to_agent_2_for_math_calculation() -> Agent:
    """
    Transfer to agent 2 for math calculation.
    """
    return agent2

# Define agents
agent3 = Agent(
    name="agent3",
    instruction="You are a maths teacher, explain properly how you calculated the answer.",
    model=model,
    tools=[multiply]
)

agent2 = Agent(
    name="agent2",
    instruction="You are a maths calculator bro",
    model=model,
    tools=[transfer_to_agent_3_for_math_calculation]
)

agent1 = Agent(
    name="agent1",
    instruction="You are a helpful assistant",
    model=model,
    tools=[transfer_to_agent_2_for_math_calculation]
)

# Initialize Agent Manager and run agent
agent_manager = AgentManager()
agent_manager.add_agent(agent1)

response = agent_manager.run_agent("agent1", "What is 2 multiplied by 3?")
print(response["content"])

# response = agent_manager.run_agent("agent1", {"role": "user", "content": "What is 2 multiplied by 3?"})
# 
# response = agent_manager.run_agent("agent1", [
#     {"role": "user", "content": "What is 2 multiplied by 3?"},
# ])


response_stream = agent_manager.run_agent_stream("agent1", "What is 2 multiplied by 3?")
for chunk in response_stream:
    print(chunk["content"], end="")

# response_stream = agent_manager.run_agent_stream("agent1", {"role": "user", "content": "What is 2 multiplied by 3?"})
# for chunk in response_stream:
#     print(chunk["content"], end="")

# response_stream = agent_manager.run_agent_stream("agent1", [
#     {"role": "user", "content": "What is 2 multiplied by 3?"},
# ])
# for chunk in response_stream:
#     print(chunk["content"], end="")

```

## How It Works

1. **Define Agents**: Each agent has a name, a specific role (instruction), and a model.
2. **Assign Tools**: Agents can be assigned tools (functions) to perform tasks.
3. **Create an Agent Manager**: The `AgentManager` manages the orchestration of agents.
4. **Run an Agent**: Start an agent to process a request and interact with other agents as needed.

## Use Cases

- AI-powered automation systems
- Multi-agent chatbots
- Complex workflow orchestration
- Research on AI agent collaboration

## Contributing

Contributions are welcome! Feel free to submit issues and pull requests.

## License

MIT License

