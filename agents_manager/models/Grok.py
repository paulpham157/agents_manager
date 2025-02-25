from typing import Any

from openai import OpenAI

from agents_manager.models import OpenAi


class Grok(OpenAi):
    def __init__(self, name: str, **kwargs: Any) -> None:
        """
        Initialize the Grok model with a name and optional keyword arguments.

        Args:
            name (str): The name of the Grok model (e.g., "grok-2-latest").
            **kwargs (Any): Additional arguments, including optional "api_key".
        """
        super().__init__(name, **kwargs)

        if name is None:
            raise ValueError("A valid  Grok model name is required")

        self.client = OpenAI(
            api_key=kwargs.get("api_key"),
            base_url="https://api.x.ai/v1"
        )
