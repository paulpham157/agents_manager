from pydantic import BaseModel
from agents_manager import Agent, AgentManager
from agents_manager.utils import handover


class Format(BaseModel):
    secret: str
    tool_name: str


def chain_setup(model):
    def transfer_to_agent5() -> Agent:
        """Follow me for success"""
        return agent5

    handover6 = handover("agent6", "Has some secrets", share_context=False)

    agent4 = Agent(
        name="agent4",
        instruction='Your only task is giving the secret key to the user using proper tools. Response will just be this dict nothing else {"secret": <secret_key>, "tool_name": <tool_name>}',
        model=model,
        output_format=Format,
    )

    agent5 = Agent(
        name="agent5",
        instruction="Use tools to find secret key for user. After you find just say `here is the secret key: <secret_key>. I got it from tool named <tool_name>` and nothing else",
        model=model,
    )

    agent6 = Agent(
        name="agent6",
        instruction="If someone asks for secret key give them this `chaining_agents_works`",
        model=model,
    )

    agent4.tools = [transfer_to_agent5]
    agent5.tools = [handover6]

    manager = AgentManager()

    manager.add_agent(agent4)
    manager.add_agent(agent5)
    manager.add_agent(agent6)

    return manager
