from typing import Any, Optional

from openai import OpenAI

from agents_manager.models import OpenAi


class DeepSeek(OpenAi):
    def __init__(self, name: str, **kwargs: Any) -> None:
        """
        Initialize the OpenAi model with a name and optional keyword arguments.

        Args:
            name (str): The name of the OpenAI model (e.g., "gpt-3.5-turbo").
            **kwargs (Any): Additional arguments, including optional "api_key".
        """
        super().__init__(name, **kwargs)

        if name is None:
            raise ValueError("A valid  DeepSeek model name is required")

        self.client = OpenAI(
            api_key=kwargs.get("api_key"),  # type: Optional[str]
            base_url="https://api.deepseek.com"
        )
