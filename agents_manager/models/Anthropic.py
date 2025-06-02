import json
from typing import List, Dict, Any, Union, Optional, Generator, Callable

from anthropic import Anthropic as Ap

from agents_manager.Container import Container
from agents_manager.Model import Model
from agents_manager.utils import populate_template, function_to_json, container_to_json


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

        self.instruction = ""
        self.client = Ap(
            api_key=kwargs.pop("api_key", None),  # type: Optional[str]
        )

    def generate_response(self) -> Dict:
        """
        Generate a non-streaming response from the OpenAI model.

        Returns:
            Union[ChatCompletion, str]: The raw ChatCompletion object if stream=False,
                                        or a string if further processed.
        """
        message = self.client.messages.create(
            model=self.name,
            system=self.instruction,
            messages=self.get_messages(),
            **self.kwargs,
        )

        return {
            "tool_calls": self.extract_content(message, "tool_use"),
            "content": self.extract_content(message, "text")[0].text,
        }

    def generate_stream_response(self) -> Generator[Dict, None, None]:
        """
        Generate a streaming response from the Anthropic model with tool_calls and content.
        Yields dictionaries containing accumulated tool_calls and content.
        """

        with self.client.messages.stream(
            model=self.name,
            system=self.instruction,
            messages=self.get_messages(),
            **self.kwargs,
        ) as stream:
            current_content_blocks = {}
            accumulated_json = {}
            result = {"tool_calls": [], "content": ""}

            current_tool = None  # Track tool call metadata, but don't accumulate input

            for event in stream:
                result = {
                    "content": None,
                    "tool_calls": None,
                }  # Fresh result dict each iteration

                # Handle text tokens as they arrive
                if (
                    event.type == "content_block_delta"
                    and event.delta.type == "text_delta"
                ):
                    result["content"] = (
                        event.delta.text
                    )  # Yield only the current text token

                # Handle tool call start
                elif (
                    event.type == "content_block_start"
                    and event.content_block.type == "tool_use"
                ):
                    current_tool = {
                        "id": event.content_block.id,
                        "name": event.content_block.name,
                        "input": None,
                    }
                    result["tool_calls"] = (
                        current_tool  # Yield tool metadata without input yet
                    )

                # Handle tool call input tokens
                elif (
                    event.type == "content_block_delta"
                    and event.delta.type == "input_json_delta"
                    and current_tool
                ):
                    # Yield the raw partial_json token as it arrives
                    result["tool_calls"] = {
                        "id": current_tool["id"],
                        "name": current_tool["name"],
                        "input": event.delta.partial_json,
                    }

                # Handle block completion
                elif event.type == "content_block_stop":
                    if current_tool:
                        # No input to finalize since we're not appending; just clear the tool
                        current_tool = None
                    # No content to yield here since we're not accumulating

                # Yield the result with the current token (if any)
                if result["content"] or result["tool_calls"]:
                    yield result

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
                    if (
                        index in current_content_blocks
                        and current_content_blocks[index].type == "text"
                    ):
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
            },
        }

    def get_keys_in_tool_output(self, tool_call: Any) -> Dict[str, Any]:
        return {
            "id": tool_call.id,
            "name": tool_call.name,
            "arguments": tool_call.input,
        }

    @staticmethod
    def _get_tool_call_format() -> Dict[str, Any]:
        return {
            "type": "tool_use",
            "id": "{id}",
            "name": "{name}",
            "input": "{arguments}",
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
                    "content": (
                        [populated_data]
                        if type(populated_data) != list
                        else populated_data
                    ),
                }
            )

        if tool_calls:
            return output_tool_calls
        else:
            return [
                {
                    "role": "assistant",
                    "content": [],
                }
            ]

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

        return {"role": "user", "content": tool_results}

    def set_system_message(self, message: str) -> None:
        self.instruction = message

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
        pass
