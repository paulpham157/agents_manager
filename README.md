# Agents Manager

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

```python

from agents_manager.models import OpenAi, Grok, DeepSeek

```

## Installation

Install the package via pip:

```sh
pip install agents-manager
```

## Quick Start

```python
from agents_manager import Agent, AgentManager
from agents_manager.models import OpenAi

from dotenv import load_dotenv

load_dotenv()

# Define the model
model = OpenAi(name="gpt-4o-mini")

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

response = agent_manager.run_agent("agent1", "What is 459 * 1?")

print(response)
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

