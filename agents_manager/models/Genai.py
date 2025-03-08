from typing import List, Dict, Any, Union, Callable, Generator

from google import genai
from google.genai import types
from openai.types.chat import ChatCompletion

from agents_manager.Model import Model
from agents_manager.utils import populate_template, function_to_json


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
        if "api_key" in kwargs:
            args["api_key"] = kwargs["api_key"]
        if "api_version" in kwargs:
            args["api_version"] = types.HttpOptions(api_version=kwargs["api_version"])
        if "project" in kwargs:
            args["project"] = kwargs["project"]
        if "location" in kwargs:
            args["location"] = kwargs["location"]
        if "vertexai" in kwargs:
            args["vertexai"] = kwargs["vertexai"]

        self.instructions = ""
        self.tools = []

        self.client = genai.Client(
            **args
        )

    def generate_response(self) -> Dict:
        """
        Generate a non-streaming response from the Genai model.

        Returns:
            Union[ChatCompletion, str]: The raw ChatCompletion object if stream=False,
                                        or a string if further processed.
        """

        # remove api_key from kwargs
        if "api_key" in self.kwargs:
            self.kwargs.pop("api_key")
        if "api_version" in self.kwargs:
            self.kwargs.pop("api_version")
        if "project" in self.kwargs:
            self.kwargs.pop("project")
        if "location" in self.kwargs:
            self.kwargs.pop("location")
        if "vertexai" in self.kwargs:
            self.kwargs.pop("vertexai")

        config = {}
        if self.kwargs.get("tools"):
            functions = self.convert_to_function_declarations(self.kwargs.get("tools"))
            tool = types.Tool(function_declarations=functions)
            config = types.GenerateContentConfig(
                system_instruction=self.instructions,
                tools=[tool],
                automatic_function_calling=types.AutomaticFunctionCallingConfig(
                    disable=True
                ),
            )
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

        # remove api_key from kwargs
        if "api_key" in self.kwargs:
            self.kwargs.pop("api_key")
        if "api_version" in self.kwargs:
            self.kwargs.pop("api_version")
        if "project" in self.kwargs:
            self.kwargs.pop("project")
        if "location" in self.kwargs:
            self.kwargs.pop("location")
        if "vertexai" in self.kwargs:
            self.kwargs.pop("vertexai")

        config = {}
        if self.kwargs.get("tools"):
            functions = self.convert_to_function_declarations(self.kwargs.get("tools"))
            tool = types.Tool(function_declarations=functions)
            config = types.GenerateContentConfig(
                system_instruction=self.instructions,
                tools=[tool],
                automatic_function_calling=types.AutomaticFunctionCallingConfig(
                    disable=True
                ),
            )

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
    def convert_to_function_declarations(json_input):
        if not isinstance(json_input, list):
            raise ValueError("Input should be a list of dictionaries")

        function_declarations = []

        for data in json_input:
            # Extract name, description, and parameters
            name = data.get('name')
            description = data.get('description')
            params = data.get('parameters', {})

            # Validate required fields
            if not name or not description:
                raise ValueError("Each function must have name and description")

            # Convert parameters to Schema format
            schema_properties = {}
            for prop_name, prop_details in params.get('properties', {}).items():
                schema_properties[prop_name] = types.Schema(
                    type=prop_details.get('type', 'STRING').upper(),
                    description=prop_details.get('description', '')
                )

            parameters = types.Schema(
                type=params.get('type', "OBJECT").upper(),
                properties=schema_properties,
                required=params.get('required', [])
            )

            # Create FunctionDeclaration and add to list
            function_declarations.append(
                types.FunctionDeclaration(
                    name=name,
                    description=description,
                    parameters=parameters
                )
            )

        return function_declarations

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
                        parts.append(types.Part.from_uri(
                            uri=file_data["file_uri"],
                            mime_type=file_data["mime_type"]
                        ))
                    elif "inline_data" in part:
                        inline_data = part["inline_data"]
                        parts.append(types.Part.from_data(
                            data=inline_data["data"],  # Base64-encoded string or similar
                            mime_type=inline_data["mime_type"]
                        ))
                    elif "function_response" in part:
                        function_response = part["function_response"]
                        name = function_response["name"]
                        response = function_response["response"]
                        parts.append(types.Part.from_function_response(
                            name=name,
                            response=response
                        ))
                    elif "function_call" in part:
                        function_call = part["function_call"]
                        name = function_call["name"]
                        args = function_call["args"]
                        parts.append(types.Part.from_function_call(
                            name=name,
                            args=args,
                        ))
            contents.append(types.Content(
                parts=parts,
                role=message["role"]
            ))
        return contents

    def get_tool_format(self) -> Dict[str, Any]:
        return {
            "name": "{name}",
            "description": "{description}",
            "parameters": {
                "type": "object",
                "properties": "{parameters}",
                "required": "{required}"
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
            "name": tool_call.name,
            "arguments": tool_call.args
        }

    @staticmethod
    def _content_to_json(content):
        parts_list = []
        for part in content.parts:
            part_dict = {}
            if part.function_call:
                function_call_dict = {
                    "name": part.function_call.name,
                    "args": part.function_call.args
                }
                part_dict["function_call"] = function_call_dict
            if part_dict:
                parts_list.append(part_dict)

        contents = [{
            "role": content.role,
            "content": parts_list
        }]

        return contents

    def get_assistant_message(self, response: Any):
        return self._content_to_json(response["candidates"][0].content)

    def get_tool_message(self, tool_responses: List[Dict[str, Any]]) -> Any:
        tool_results = []
        for tool_response in tool_responses:
            tool_results.append(
                {
                    "role": "tool",
                    "content": [
                        {
                            "function_response": {
                                "name": tool_response["name"],
                                "response": {
                                    "result": tool_response["tool_result"],
                                },
                            }
                        }
                    ],
                }
            )

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
        self.kwargs.update({
            "tools": json_tools
        })
