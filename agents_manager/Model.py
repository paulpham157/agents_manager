import json
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Generator, Callable


class Model(ABC):
    def __init__(self, name: str, **kwargs: Any) -> None:
        """
        Initialize the Model with a name and optional keyword arguments.

        Args:
            name (str): The name of the model.
            **kwargs (Any): Additional keyword arguments.
        """
        self.messages: str = ""  # Messages can be None initially
        self.name: str = name
        self.kwargs: Dict[str, Any] = kwargs

    def set_messages(self, messages: List[Dict[str, str]]) -> None:
        """
        Set the messages for the model.

        Args:
            messages (List[Dict[str, str]]): A list of message dictionaries with "role" and "content".
        """
        self.messages = json.dumps(messages)

    def get_messages(self) -> Optional[List[Dict[str, str]]]:
        """
        Get the messages for the model.

        Returns:
            Optional[List[Dict[str, str]]]: The list of message dictionaries if set, else None.
        """
        return json.loads(self.messages) if len(self.messages) > 0 else None

    def clear_messages(self) -> None:
        """
        Clear the messages for the model.
        """
        self.messages = None

    def set_kwargs(self, kwargs: Dict[str, Any]) -> None:
        """
        Update the model's keyword arguments by merging with existing ones.

        Args:
            kwargs (Dict[str, Any]): New keyword arguments to merge with existing ones.
        """
        self.kwargs = {**self.kwargs, **kwargs}

    @abstractmethod
    def generate_response(self) -> Dict:
        """
        Generate a non-streaming response based on the model's implementation.

        Returns:
            Any: The response, type depends on the concrete implementation.
        """
        return {
            "tool_calls": [],
            "content": "",
        }

    @abstractmethod
    def generate_stream_response(self) -> Generator[Dict, None, None]:
        """
        Generate a non-streaming response based on the model's implementation.

        Returns:
            Any: The response, type depends on the concrete implementation.
        """
        yield {
            "tool_calls": [],
            "content": "",
        }

    @abstractmethod
    def get_tool_format(self) -> Dict[str, Any]:
        """
        Get the format for the tool call.

        Returns:
            Dict[str, Any]: The tool call format.
        """
        return {}

    @abstractmethod
    def get_keys_in_tool_output(self, tool_call: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get the parsed tool call data.

        Args:
            tool_call (Dict[str, Any]): The tool call data.

        Returns:
            Dict[str, Any]: The parsed tool call data.
        """
        return {}

    @abstractmethod
    def get_assistant_message(self, response: Any) -> Dict[str, Any]:
        """
        Get the assistant message for prepending to the response.

        Args:
            response (Any): The response from the model.

        Returns:
            Dict[str, Any]: The assistant message.
        """
        return {}

    @abstractmethod
    def get_tool_message(self, tool_responses: List[Dict[str, Any]]) -> Any:
        """
        Get the tool message for appending to the response.

        Args:
            tool_responses (List[Dict[str, Any]]): The tool responses.
        Returns:
            Dict[str, Any]: The tool message.
        """
        return {}

    @abstractmethod
    def set_system_message(self, message: str) -> None:
        """
        Set the system message for the model.

        Args:
            message (str): The system message.
        """
        pass

    @abstractmethod
    def set_user_message(self, message: str) -> None:
        """
        Set the user message for the model.

        Args:
            message (str): The user message.
        """
        pass

    @abstractmethod
    def set_tools(self, tools: List[Callable]) -> None:
        """
        Set the tools for the model.

        Args:
            tools (List[Callable]): The tools.
        """
        pass

    @abstractmethod
    def set_output_format(self, output_format: Callable) -> None:
        """
        Set the output format for the model.

        Args:
            output_format (Callable): The output format.
        """
        pass