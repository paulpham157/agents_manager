from typing import Any

from openai import OpenAI

from agents_manager.models import OpenAi
import os


class GenUi(OpenAi):
    def __init__(self, name: str, **kwargs: Any) -> None:
        """
        Initialize the Grok model with a name and optional keyword arguments.

        Args:
            name (str): The name of the Grok model (e.g., "grok-2-latest").
            **kwargs (Any): Additional arguments, including optional "api_key".
        """
        super().__init__(name, **kwargs)

        if name is None:
            raise ValueError("A valid  GenUi model name is required")

        self.client = OpenAI(
            api_key=os.getenv("PROCHAT_API_KEY", kwargs.get("api_key")),
            base_url="https://www.prochat.dev/apps/api/v1",
        )
