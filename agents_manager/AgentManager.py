import json
from typing import List, Optional, Any

from agents_manager.Agent import Agent
from agents_manager.utils import populate_template


class AgentManager:
    def __init__(self) -> None:
        """
        Initialize the AgentManager with an empty list of agents.
        """
        self.agents: List[Agent] = []

    def add_agent(self, agent: Agent) -> None:
        """
        Add an agent to the manager's list.

        Args:
            agent (Agent): The agent instance to add.
        """
        if not isinstance(agent, Agent):
            raise ValueError("Only Agent instances can be added")
        self.agents.append(agent)
        agent.set_messages([{"role": "assistant", "content": agent.instruction}])
        agent.set_tools(agent.tools)

    def get_agent(self, name: str) -> Optional[Agent]:
        """
        Retrieve an agent by name.

        Args:
            name (str): The name of the agent to find.

        Returns:
            Optional[Agent]: The agent if found, else None.
        """
        for agent in self.agents:
            if agent.name == name:
                return agent
        return None

    def run_agent(self, name: str, user_input: Optional[str] = None) -> Any:
        """
        Run a specific agent's non-streaming response.

        Args:
            name (str): The name of the agent to run.
            user_input (str, optional): Additional user input to append to messages.

        Returns:
            Any: The agent's response.
        """
        agent = self.get_agent(name)
        if agent is None:
            raise ValueError(f"No agent found with name: {name}")

        if user_input:
            current_messages = agent.get_messages() or []
            agent.set_messages(current_messages + [{"role": "user", "content": user_input}])

        response = agent.get_response()

        if not response['tool_calls']:
            return response["content"]

        tool_calls = response['tool_calls']
        current_messages = agent.get_messages()

        output_tool_calls = []
        for tool_call in tool_calls:
            output = agent.get_model().get_parsed_tool_call_data(tool_call)
            populated_data = populate_template(agent.get_model().get_tool_call_format(), output)
            output_tool_calls.append(populated_data)

        if output_tool_calls and len(output_tool_calls) > 0:
            assistant_message = {
                "role": "assistant",
                "content": response["content"] or "",
                "tool_calls": output_tool_calls,
            }
            current_messages.append(assistant_message)

        # executing the functions and getting the response
        for tool_call in tool_calls:
            output = agent.get_model().get_parsed_tool_call_data(tool_call)
            function_name = output["name"]
            arguments = json.loads(output["arguments"])
            tools = agent.tools
            for tool in tools:
                if tool.__name__ == function_name:
                    tool_result = tool(**arguments)
                    if isinstance(tool_result, Agent):
                        self.add_agent(tool_result)
                        nested_response = self.run_agent(tool_result.name,
                                                         user_input
                                                         )
                        tool_response_content = (
                            nested_response.content
                            if hasattr(nested_response, "content")
                            else str(nested_response)
                        )
                        return tool_response_content
                    else:
                        tool_response = {
                                            "role": "tool",
                                            "content": str(tool_result),
                                        } | populate_template(agent.get_model().get_tool_call_id_format(), output)
                    current_messages.append(tool_response)

        agent.set_messages(current_messages)
        response = agent.get_response()
        return response["content"]

    def run_agent_stream(self, name: str, user_input: Optional[str] = None) -> Any:
        """
        Run a specific agent's streaming response.

        Args:
            name (str): The name of the agent to run.
            user_input (str, optional): Additional user input to append to messages.

        Returns:
            Any: The agent's streaming response.
        """
        agent = self.get_agent(name)
        if agent is None:
            raise ValueError(f"No agent found with name: {name}")

        if user_input:
            current_messages = agent.model.messages or []
            agent.model.set_messages(current_messages + [{"role": "user", "content": user_input}])

        return agent.get_response_stream()
