import json
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import copy


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
    def generate_response(self) -> Any:
        """
        Generate a non-streaming response based on the model's implementation.

        Returns:
            Any: The response, type depends on the concrete implementation.
        """
        pass

    @abstractmethod
    def get_tool_format(self) -> Dict[str, Any]:
        """
        Get the format for the tool call.

        Returns:
            Dict[str, Any]: The tool call format.
        """
        pass


    @abstractmethod
    def get_tool_call_format(self) -> Dict[str, Any]:
        """
        Get the assistant message for prepending to the response.

        Returns:
            Dict[str, Any]: The assistant message.
        """
        pass

    @abstractmethod
    def get_tool_call_id_format(self) -> Dict[str, Any]:
        """
        Get the tool message for appending to the response.

        Returns:
            Dict[str, Any]: The tool message.
        """
        pass

    @abstractmethod
    def get_parsed_tool_call_data(self, tool_call: Any)-> Dict[str, Any]:
        """
        Get the parsed tool call data.

        Returns:
            Dict[str, Any]: The parsed tool call data.
        """
        pass