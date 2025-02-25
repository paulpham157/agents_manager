import json
from typing import List, Dict, Any, Union, Optional

from anthropic import Anthropic as Ap

from agents_manager.Model import Model
from agents_manager.utils import populate_template


class Anthropic(Model):
    def __init__(self, name: str, **kwargs: Any) -> None:
        """
        Initialize the Anthropic model with a name and optional keyword arguments.

        Args:
            name (str): The name of the Anthropic model (e.g., "claude-3-5-sonnet-20241022").
            **kwargs (Any): Additional arguments, including optional "api_key".
        """
        super().__init__(name, **kwargs)

        if name is None:
            raise ValueError("A valid  OpenAI model name is required")

        self.client = Ap(
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

        needs_stream = self.kwargs.get("stream", False)
        kwargs = self.kwargs.copy()
        kwargs.pop("stream", None)
        if needs_stream:
            with self.client.messages.stream(
                    model=self.name,
                    messages=self.get_messages(),
                    **kwargs
            ) as stream:
                message = self.parse_stream(stream)
            return {
                "tool_calls": self.extract_content(message, "tool_use"),
                "content": self.extract_content(message, "text")[0].text,
            }
        else:
            message = self.client.messages.create(
                model=self.name,
                messages=self.get_messages(),
                **kwargs
            )
            return {
                "tool_calls": self.extract_content(message, "tool_use"),
                "content": self.extract_content(message, "text")[0].text,
            }

    @staticmethod
    def parse_stream(stream):
        current_content_blocks = {}
        accumulated_json = {}

        for event in stream:
            # Handle different event types
            if event.type == "message_start":
                pass

            elif event.type == "content_block_start":
                # Initialize a new content block
                index = event.index
                content_block = event.content_block
                current_content_blocks[index] = content_block

                if content_block.type == "tool_use":
                    accumulated_json[index] = ""

            elif event.type == "content_block_delta":
                index = event.index
                delta = event.delta

                # Handle text deltas
                if delta.type == "text_delta":
                    if index in current_content_blocks and current_content_blocks[index].type == "text":
                        if not hasattr(current_content_blocks[index], "text"):
                            current_content_blocks[index].text = ""
                        current_content_blocks[index].text += delta.text

                # Handle tool use input deltas
                elif delta.type == "input_json_delta":
                    if index in accumulated_json:
                        accumulated_json[index] += delta.partial_json
                        if accumulated_json[index].endswith("}"):
                            try:
                                parsed_json = json.loads(accumulated_json[index])
                            except json.JSONDecodeError:
                                pass

            elif event.type == "content_block_stop":
                index = event.index
                if index in current_content_blocks:
                    block_type = current_content_blocks[index].type
                    if block_type == "tool_use" and index in accumulated_json:
                        # Final parse of the complete JSON
                        try:
                            parsed_json = json.loads(accumulated_json[index])
                        except json.JSONDecodeError as e:
                            pass

            elif event.type == "message_delta":
                # Handle updates to the message metadata
                if event.delta.stop_reason:
                    pass

            elif event.type == "message_stop":
                pass
            # Get the final message after streaming completes
        return stream.get_final_message()

    @staticmethod
    def extract_content(response, type_filter="tool_use"):
        """
        Extract items of a specific type from a Claude API response object.

        Args:
            response: The response object from Claude API
            type_filter (str): The type of items to extract (default: "tool_use")

        Returns:
            list: A list of filtered items
        """
        items = []
        if hasattr(response, "content") and isinstance(response.content, list):
            for item in response.content:
                if hasattr(item, "type") and item.type == type_filter:
                    items.append(item)
        return items

    def get_tool_format(self) -> Dict[str, Any]:
        return {
            "name": "{name}",
            "description": "{description}",
            "input_schema": {
                "type": "object",
                "properties": "{parameters}",
                "required": "{required}",
            }
        }

    def get_keys_in_tool_output(self, tool_call: Any) -> Dict[str, Any]:
        return {
            "id": tool_call.id,
            "name": tool_call.name,
            "arguments": tool_call.input
        }

    @staticmethod
    def _get_tool_call_format() -> Dict[str, Any]:
        return {
            "type": "tool_use",
            "id": "{id}",
            "name": "{name}",
            "input": "{arguments}"
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
            "content": output_tool_calls,
        }

    def get_tool_message(self, tool_responses: List[Dict[str, Any]]) -> Any:

        tool_results = []
        for tool_response in tool_responses:
            tool_results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": tool_response["id"],
                    "content": tool_response["tool_result"],
                }
            )

        return {
            "role": "user",
            "content": tool_results
        }
