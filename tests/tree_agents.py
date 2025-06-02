from pydantic import BaseModel
from agents_manager import Agent
from agents_manager import AgentManager
from agents_manager.utils import handover


class OtherAgentFormat(BaseModel):
    content: str
    pos: int


class AgentOneFormat(BaseModel):
    summarize: OtherAgentFormat
    extend: OtherAgentFormat


def tree_setup(model):
    def transfer_to_agent2() -> Agent:
        """Extends length of any content"""
        return agent2

    handover3 = handover("agent3", "Summarizes any given content", share_context=False)

    agent1 = Agent(
        name="agent1",
        instruction="""
        Your responsibility is to properly forward user query to respective agents and then at last
        properly format the response.

        You should always reply in this format:
        {
            "summarize": {
                "content": <summarized_content>,
                "pos": <in which position this tool was called>
            },
            "extend": {
                "content": <summarized_content>,
                "pos": <in which position this tool was called>
            }
        }
        """,
        model=model,
        output_format=AgentOneFormat,
    )

    agent2 = Agent(
        name="agent2",
        instruction="""
        Your job is to extend the length of any thing the user provides, you should reply with
        atleast 100 words. Add whatever you want. You can't do anything more than that.
        """,
        model=model,
    )

    agent3 = Agent(
        name="agent3",
        instruction="""
        Summarize any content with less than 20 words. You can't do anything more than that.
        """,
        model=model,
    )

    agent1.tools = [transfer_to_agent2, handover3]

    manager = AgentManager()

    manager.add_agent(agent1)
    manager.add_agent(agent2)
    manager.add_agent(agent3)

    return manager
