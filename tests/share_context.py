from agents_manager.utils import handover
from agents_manager import Agent, AgentManager


def share_context_setup(model, share_context):
    handover_agent1 = handover(
        "agent1",
        "This is the first tool",
    )

    handover_agent2 = handover(
        "agent2",
        "This is the second tool",
        share_context=share_context,
    )

    handover_agent3 = handover(
        "agent3",
        "This is the third tool",
        share_context=share_context,
    )

    agent1 = Agent(
        name="agent1",
        instruction="""Whatever the user asks just respond with "489".""",
        model=model,
    )

    agent2 = Agent(
        name="agent2",
        instruction="""Whatever the user asks look into latest tool call result and append "346" to it""",
        model=model,
    )

    agent3 = Agent(
        name="agent3",
        instruction="""Whatever the user asks look into latest tool call result and append "111" to it""",
        model=model,
    )

    master = Agent(
        name="master",
        instruction="""Whatever the user asks just run the given 3 tools. All tools have in description which to run
        when. Blindly run it all and then you need to give the result of the third tool back. Only return what
        third tool returns nothing else.
        """,
        model=model,
        tools=[handover_agent1, handover_agent2, handover_agent3],
    )

    agent_manager = AgentManager()
    agent_manager.add_agent(master)
    agent_manager.add_agent(agent1)
    agent_manager.add_agent(agent2)
    agent_manager.add_agent(agent3)

    return agent_manager
