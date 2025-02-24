from typing import Iterator
from typing import List, Dict, Any, Union, Optional

from openai import OpenAI
from openai.types.chat import ChatCompletion, ChatCompletionMessageToolCall
from openai.types.chat.chat_completion_chunk import ChatCompletionChunk

from agents_manager.Model import Model


class Grok(Model):
    def __init__(self, name: str, **kwargs: Any) -> None:
        """
        Initialize the OpenAi model with a name and optional keyword arguments.

        Args:
            name (str): The name of the OpenAI model (e.g., "gpt-3.5-turbo").
            **kwargs (Any): Additional arguments, including optional "api_key".
        """
        if name is None:
            raise ValueError("A valid  OpenAI model name is required")

        super().__init__(name, **kwargs)

        self.client = OpenAI(
            api_key=kwargs.get("api_key"),  # type: Optional[str]
            base_url="https://api.x.ai/v1"
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
        message = response.choices[0].message
        return {
            "tool_calls": message.tool_calls,
            "content": message.content,
        }

    def generate_response_stream(self) -> Iterator[ChatCompletionChunk]:
        """
        Generate a streaming response from the OpenAI model.

        Returns:
            Iterator[ChatCompletionChunk]: An iterator over ChatCompletionChunk objects.
        """
        if "api_key" in self.kwargs:
            self.kwargs.pop("api_key")

        return self.client.chat.completions.create(
            model=self.name,  # type: str
            messages=self.get_messages(),  # type: List[Dict[str, str]]
            **self.kwargs,  # type: Dict[str, Any]
            stream=True,  # type: bool
        )

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

    def get_tool_call_format(self) -> Dict[str, Any]:
        return {
            "id": "{id}",
            "type": "function",
            "function": {
                "name": "{name}",
                "arguments": "{arguments}"
            }
        }

    def get_tool_call_id_format(self) -> Dict[str, Any]:
        return {
            "tool_call_id": "{id}",
        }

    def get_parsed_tool_call_data(self, tool_call: ChatCompletionMessageToolCall) -> Dict[str, Any]:
        return {
            "id": tool_call.id,
            "name": tool_call.function.name,
            "arguments": tool_call.function.arguments
        }
