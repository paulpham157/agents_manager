import json
from agents_manager.models import OpenAi, Anthropic, Genai

from tree_agents import tree_setup
from chain_agents import chain_setup
from share_context import share_context_setup

STORY = """
A quiet seed fell into rich soil.
Rain came gently, and the sun followed.
Days passed. A sprout emerged, green and hopeful.
It grew tall, touched by breeze and birdsong.
In time, it became a tree, offering shade and shelter.
Life continued, simple and still, beneath its patient branches.
"""

openai_model = OpenAi(name="gpt-4o-mini")
genai_model = Genai(name="gemini-2.0-flash")
anthropic_model = Anthropic(name="claude-sonnet-4-20250514", max_tokens=1024)


def test_tree_handover():
    manager = tree_setup(openai_model)

    resp = manager.run_agent(
        "agent1",
        [{"role": "user", "content": f"Summarize it and then extend it {STORY}"}],
    )

    resp = json.loads(resp["content"])

    assert resp["summarize"]["pos"] == 1
    assert resp["extend"]["pos"] == 2


def test_chain_handover():
    manager = chain_setup(openai_model)

    resp = manager.run_agent(
        "agent4",
        [{"role": "user", "content": "Give me the secret"}],
    )

    resp = json.loads(resp["content"])

    assert resp["secret"] == "chaining_agents_works"
    assert resp["tool_name"] == "handover_agent6"


def test_share_context():
    manager = share_context_setup(anthropic_model, True)

    resp = manager.run_agent(
        "master", {"role": "user", "content": "Do as the system prompt says"}
    )

    assert "489346111" in resp["content"]
