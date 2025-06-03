from typing import List, Dict, Any, Union, Callable, Generator

from google import genai
from google.genai import types
from openai.types.chat import ChatCompletion

from agents_manager.Container import Container
from agents_manager.Model import Model
from agents_manager.utils import function_to_json, container_to_json


class Genai(Model):
    def __init__(self, name: str, **kwargs: Any) -> None:
        """
        Initialize the Genai model with a name and optional keyword arguments.

        Args:
            name (str): The name of the Genai model (e.g., "gemini-2.0-flash").
            **kwargs (Any): Additional arguments, including optional "api_key".
        """
        super().__init__(name, **kwargs)

        if name is None:
            raise ValueError("A valid  Genai model name is required")

        args = {}
        if "api_key" in self.kwargs:
            args["api_key"] = self.kwargs.pop("api_key")
        if "api_version" in self.kwargs:
            args["api_version"] = types.HttpOptions(
                api_version=self.kwargs.pop("api_version")
            )
        if "project" in self.kwargs:
            args["project"] = self.kwargs.pop("project")
        if "location" in self.kwargs:
            args["location"] = self.kwargs.pop("location")
        if "vertexai" in self.kwargs:
            args["vertexai"] = self.kwargs.pop("vertexai")

        self.instructions = ""
        self.tools = []

        self.client = genai.Client(**args)

    def generate_response(self) -> Dict:
        """
        Generate a non-streaming response from the Genai model.

        Returns:
            Union[ChatCompletion, str]: The raw ChatCompletion object if stream=False,
                                        or a string if further processed.
        """

        kwargs = self.kwargs.copy()
        tools = kwargs.pop("tools", None)
        output_format = kwargs.pop("output_format", None)
        config = {
            "system_instruction": self.instructions,
        }
        if not self.has_tool_function_response(self.get_messages()):
            if tools:
                config.update(
                    {
                        "tools": [{"function_declarations": tools}],
                        "automatic_function_calling": {"disable": True},
                    }
                )

        if output_format:
            config.update(
                {
                    "response_mime_type": "application/json",
                    "response_schema": output_format,
                }
            )

        config.update(kwargs)

        response = self.client.models.generate_content(
            model=self.name,
            contents=self._convert_to_contents(self.get_messages()),
            config=config,
        )
        return {
            "tool_calls": response.function_calls,
            "content": response.text if not response.function_calls else "",
            "candidates": response.candidates or "",
        }

    def generate_stream_response(self) -> Generator[Dict, None, None]:
        """
        Generate a non-streaming response from the Genai model.

        Returns:
            Union[ChatCompletion, str]: The raw ChatCompletion object if stream=False,
                                        or a string if further processed.
        """

        kwargs = self.kwargs.copy()
        tools = kwargs.pop("tools", None)
        output_format = kwargs.pop("output_format", None)
        config = {
            "system_instruction": self.instructions,
        }
        if not self.has_tool_function_response(self.get_messages()):
            if tools:
                config.update(
                    {
                        "tools": [{"function_declarations": tools}],
                        "automatic_function_calling": {"disable": True},
                    }
                )

        if output_format:
            config.update(
                {
                    "response_mime_type": "application/json",
                    "response_schema": output_format,
                }
            )

        config.update(kwargs)
        response = self.client.models.generate_content_stream(
            model=self.name,
            contents=self._convert_to_contents(self.get_messages()),
            config=config,
        )

        result = {
            "tool_calls": [],
            "content": "",
        }
        for chunk in response:
            if chunk.function_calls:
                final_tool_calls = chunk.function_calls
                result["tool_calls"] = final_tool_calls
            if chunk.text is not None:
                result["content"] = chunk.text
            yield result
        return

    @staticmethod
    def has_tool_function_response(messages):
        if not messages:
            return False

        last_message = messages[-1]

        return (
            last_message.get("role") == "tool"
            and isinstance(last_message.get("content"), list)
            and any("function_response" in item for item in last_message["content"])
        )

    @staticmethod
    def _convert_to_contents(messages):
        """
        Convert a list of dictionaries with 'role' and 'content' keys
        into a list of google-genai `types.Content` objects.

        Args:
            messages (list): List of dicts with 'role' and 'content' keys.

        Returns:
            list: List of `types.Content` objects.
        """

        contents = []
        for message in messages:
            parts = message.get("content", None)
            if isinstance(parts, str):
                parts = [types.Part.from_text(text=parts)]
            if isinstance(parts, list):
                for part in message.get("parts", []):  # Safely get the "parts" list
                    if "text" in part:
                        parts.append(types.Part.from_text(text=part["text"]))
                    elif "file_data" in part:
                        file_data = part["file_data"]
                        parts.append(
                            types.Part.from_uri(
                                uri=file_data["file_uri"],
                                mime_type=file_data["mime_type"],
                            )
                        )
                    elif "inline_data" in part:
                        inline_data = part["inline_data"]
                        parts.append(
                            types.Part.from_data(
                                data=inline_data[
                                    "data"
                                ],  # Base64-encoded string or similar
                                mime_type=inline_data["mime_type"],
                            )
                        )
                    elif "function_response" in part:
                        function_response = part["function_response"]
                        name = function_response["name"]
                        response = function_response["response"]
                        parts.append(
                            types.Part.from_function_response(
                                name=name, response=response
                            )
                        )
                    elif "function_call" in part:
                        function_call = part["function_call"]
                        name = function_call["name"]
                        args = function_call["args"]
                        parts.append(
                            types.Part.from_function_call(
                                name=name,
                                args=args,
                            )
                        )
            contents.append(types.Content(parts=parts, role=message["role"]))
        return contents

    def get_tool_format(self) -> Dict[str, Any]:
        return {
            "name": "{name}",
            "description": "{description}",
            "parameters": {
                "type": "object",
                "properties": "{parameters}",
                "required": "{required}",
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
        return {"id": tool_call.id, "name": tool_call.name, "arguments": tool_call.args}

    @staticmethod
    def _content_to_json(content):
        parts_list = []
        for part in content.parts:
            part_dict = {}
            if part.function_call:
                function_call_dict = {
                    "name": part.function_call.name,
                    "args": part.function_call.args,
                }
                part_dict["function_call"] = function_call_dict
            if part_dict:
                parts_list.append({"role": content.role, "content": [part_dict]})

        if parts_list:
            return parts_list
        else:
            return [{"role": content.role, "content": []}]

    def get_assistant_message(self, response: Any):
        return self._content_to_json(response["candidates"][0].content)

    def get_tool_message(self, tool_responses: List[Dict[str, Any]]) -> Any:
        tool_results = {}
        content = []
        for tool_response in tool_responses:
            content.append(
                {
                    "function_response": {
                        "name": tool_response["name"],
                        "response": {
                            "result": tool_response["tool_result"],
                        },
                    }
                }
            )
        tool_results["role"] = "tool"
        tool_results["content"] = content

        return tool_results

    def set_system_message(self, message: str) -> None:
        self.instructions = message

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
