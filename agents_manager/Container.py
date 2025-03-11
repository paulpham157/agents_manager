from typing import Any, Dict

import docker
from agents_manager.utils import replace_placeholder


class Container:
    def __init__(self, name: str, description: str, **kwargs: Any):
        self.name = name
        self.description = description
        self.environment = kwargs.get("environment", {})
        self.auth_credentials = kwargs.get("authenticate", {})
        self.return_to = kwargs.get("return_to", None)
        self.kwargs = kwargs
        self.client = None
        self.initialize()
        self._authenticate()

    def initialize(self):
        try:
            self.client = docker.from_env()
        except Exception as e:
            print(f"Error: {e}")

    def _authenticate(self):
        if self.auth_credentials:
            self.client.login(
                username=self.auth_credentials.get("username"),
                password=self.auth_credentials.get("password"),
                registry=self.auth_credentials.get("registry"),
            )

    def pull_image(self):
        """Pull the specified image from the registry."""
        image = self.kwargs.get("image")
        if not image:
            raise ValueError("No image specified in kwargs")
        self.client.images.pull(image)

    def run(self, arguments: Dict[str, Any]):
        """Run the container with provided arguments."""
        if "image" not in self.kwargs:
            raise ValueError("Image must be specified in kwargs")
        self.pull_image()
        self.kwargs["detach"] = False
        self.kwargs["remove"] = True
        self.kwargs["environment"] = arguments
        return_to = self.kwargs.pop("return_to", None)

        result = self.client.containers.run(
            **self.kwargs,
        )
        if return_to and "agent" in return_to:
            if "instruction" in return_to:
                instruction = replace_placeholder(return_to["instruction"], result)
                return_to["agent"].set_instruction(instruction=instruction)
            return return_to["agent"]
        return result
