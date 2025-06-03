from typing import List, Dict, Any, Union, Optional, Generator, Callable

from openai import OpenAI
from openai.types.chat import ChatCompletion

from agents_manager.Container import Container
from agents_manager.Model import Model
from agents_manager.utils import populate_template, function_to_json, container_to_json


class OpenAi(Model):
    def __init__(self, name: str, **kwargs: Any) -> None:
        """
        Initialize the OpenAi model with a name and optional keyword arguments.

        Args:
            name (str): The name of the OpenAI model (e.g., "gpt-3.5-turbo").
            **kwargs (Any): Additional arguments, including optional "api_key".
        """
        super().__init__(name, **kwargs)

        if name is None:
            raise ValueError("A valid  OpenAI model name is required")

        self.client = OpenAI(
            api_key=self.kwargs.pop("api_key", None),  # type: Optional[str]
        )

    def generate_response(self) -> Dict:
        """
        Generate a non-streaming response from the OpenAI model.

        Returns:
            Union[ChatCompletion, str]: The raw ChatCompletion object if stream=False,
                                        or a string if further processed.
        """
        kwargs = self.kwargs.copy()
        output_format = kwargs.pop("output_format")
        if not self.kwargs.get("output_format", None):
            response = self.client.chat.completions.create(
                model=self.name,  # type: str
                messages=self.get_messages(),  # type: List[Dict[str, str]]
                **kwargs,  # type: Dict[str, Any],
                stream=False,
            )
        else:
            response = self.client.beta.chat.completions.parse(
                model=self.name,  # type: str
                messages=self.get_messages(),  # type: List[Dict[str, str]]
                response_format=output_format,
                **kwargs,  # type: Dict[str, Any]
            )

        message = response.choices[0].message

        return {
            "tool_calls": message.tool_calls or [],
            "content": message.content,
        }

    def generate_stream_response(self) -> Generator[Dict, None, None]:
        """
        Generate a streaming response from the OpenAI model.

        Returns:
            Union[ChatCompletion, str]: The raw ChatCompletion object if stream=False,
                                        or a string if further processed.
        """

        kwargs = self.kwargs.copy()
        output_format = kwargs.pop("output_format")

        if not self.kwargs.get("output_format", None):
            response = self.client.chat.completions.create(
                model=self.name,  # type: str
                messages=self.get_messages(),  # type: List[Dict[str, str]]
                **kwargs,  # type: Dict[str, Any]
                stream=True,
            )
            final_tool_calls = {}
            result = {
                "tool_calls": [],
                "content": "",
            }
            for chunk in response:
                for tool_call in chunk.choices[0].delta.tool_calls or []:
                    index = tool_call.index
                    if index not in final_tool_calls:
                        final_tool_calls[index] = tool_call
                    final_tool_calls[
                        index
                    ].function.arguments += tool_call.function.arguments
                    result["tool_calls"] = final_tool_calls
                if chunk.choices[0].delta.content is not None:
                    result["content"] = chunk.choices[0].delta.content
                yield result
            return
        else:
            with self.client.beta.chat.completions.stream(
                model=self.name,  # type: str
                messages=self.get_messages(),  # type: List[Dict[str, str]]
                response_format=output_format,
                **kwargs,  # type: Dict[str, Any]
            ) as response:
                result = {
                    "tool_calls": [],
                    "content": "",
                }
                for event in response:
                    if event.type == "content.delta":
                        if event.parsed is not None:
                            # print("content.delta parsed:", event.parsed)
                            result["content"] = event.parsed
                    elif event.type == "content.done":
                        # print("content.done")
                        pass
                    elif event.type == "error":
                        # print("Error in stream:", event.error)
                        pass
                    yield result
                return

    def get_tool_format(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": "{name}",
                "description": "{description}",
                "parameters": {
                    "type": "object",
                    "properties": "{parameters}",
                    "required": "{required}",
                    "additionalProperties": False,
                },
                "strict": True,
            },
        }

    @staticmethod
    def _get_tool_call_format() -> Dict[str, Any]:
        return {
            "id": "{id}",
            "type": "function",
            "function": {"name": "{name}", "arguments": "{arguments}"},
        }

    def get_keys_in_tool_output(self, tool_call: Any) -> Dict[str, Any]:
        return {
            "id": tool_call.id,
            "name": tool_call.function.name,
            "arguments": tool_call.function.arguments,
        }

    def get_assistant_message(self, response: Any):

        tool_calls = response["tool_calls"]
        output_tool_calls = []
        for tool_call in tool_calls:
            output = self.get_keys_in_tool_output(tool_call)
            populated_data = populate_template(self._get_tool_call_format(), output)
            output_tool_calls.append(
                {
                    "role": "assistant",
                    "content": response["content"] or "",
                    "tool_calls": (
                        [populated_data]
                        if type(populated_data) != list
                        else populated_data
                    ),
                }
            )

        if tool_calls:
            return output_tool_calls
        else:
            [
                {
                    "role": "assistant",
                    "content": response["content"] or "",
                    "tool_calls": [],
                }
            ]

    def get_tool_message(self, tool_responses: List[Dict[str, Any]]) -> Any:
        tool_results = []
        for tool_response in tool_responses:
            tool_results.append(
                {
                    "role": "tool",
                    "content": tool_response["tool_result"],
                    "tool_call_id": tool_response["id"],
                }
            )

        return tool_results

    def set_system_message(self, message: str) -> None:
        self.set_messages(
            [
                {
                    "role": "system",
                    "content": message,
                }
            ]
        )

    def set_user_message(self, message: str) -> None:
        current_messages = self.get_messages() or []
        if isinstance(message, str):
            user_input = {"role": "user", "content": message}
            current_messages.append(user_input)
        if isinstance(message, dict):
            user_input = [message]
            current_messages.extend(user_input)
        if isinstance(message, list):
            current_messages.extend(message)
        self.set_messages(current_messages)

    def set_tools(self, tools: List[Callable]) -> None:

        json_tools: List[Dict[str, Any]] = []
        for tool in tools:
            if isinstance(tool, Callable):
                json_tools.append(function_to_json(tool, self.get_tool_format()))
            if isinstance(tool, Container):
                json_tools.append(container_to_json(tool, self.get_tool_format()))
        self.kwargs.update({"tools": json_tools})

    def set_output_format(self, output_format: Callable) -> None:
        self.kwargs.update({"output_format": output_format})
