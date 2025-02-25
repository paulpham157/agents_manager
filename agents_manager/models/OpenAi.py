from typing import List, Dict, Any, Union, Optional

from openai import OpenAI
from openai.types.chat import ChatCompletion

from agents_manager.Model import Model
from agents_manager.utils import populate_template


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
            api_key=kwargs.get("api_key"),  # type: Optional[str]
        )

    def generate_response(self) -> Dict:
        """
        Generate a non-streaming response from the OpenAI model.

        Returns:
            Union[ChatCompletion, str]: The raw ChatCompletion object if stream=False,
                                        or a string if further processed.
        """

        # remove api_key from kwargs
        if "api_key" in self.kwargs:
            self.kwargs.pop("api_key")

        response = self.client.chat.completions.create(
            model=self.name,  # type: str
            messages=self.get_messages(),  # type: List[Dict[str, str]]
            **self.kwargs  # type: Dict[str, Any]
        )
        stream = self.kwargs.get("stream", False)
        if stream:
            final_tool_calls = {}
            final_content = ""
            for chunk in response:
                for tool_call in chunk.choices[0].delta.tool_calls or []:
                    index = tool_call.index
                    if index not in final_tool_calls:
                        final_tool_calls[index] = tool_call
                    final_tool_calls[index].function.arguments += tool_call.function.arguments

                if chunk.choices[0].delta.content is not None:
                    final_content += chunk.choices[0].delta.content

            return {
                "tool_calls": list(final_tool_calls.values()),
                "content": final_content,
            }
        else:
            message = response.choices[0].message
            return {
                "tool_calls": message.tool_calls,
                "content": message.content,
            }

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
            }
        }

    @staticmethod
    def _get_tool_call_format() -> Dict[str, Any]:
        return {
            "id": "{id}",
            "type": "function",
            "function": {
                "name": "{name}",
                "arguments": "{arguments}"
            }
        }

    def get_keys_in_tool_output(self, tool_call: Any) -> Dict[str, Any]:
        return {
            "id": tool_call.id,
            "name": tool_call.function.name,
            "arguments": tool_call.function.arguments
        }

    def get_assistant_message(self, response: Any):

        tool_calls = response["tool_calls"]
        output_tool_calls = []
        for tool_call in tool_calls:
            output = self.get_keys_in_tool_output(tool_call)
            populated_data = populate_template(self._get_tool_call_format(), output)
            output_tool_calls.append(populated_data)

        return {
            "role": "assistant",
            "content": response["content"] or "",
            "tool_calls": output_tool_calls,
        }

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
