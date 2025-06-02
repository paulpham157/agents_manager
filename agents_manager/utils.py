import inspect


def populate_template(template, data):
    if isinstance(template, dict):
        result = {}
        for key, value in template.items():
            if isinstance(value, str) and value.startswith("{") and value.endswith("}"):
                key_in_data = value[1:-1]
                result[key] = data.get(key_in_data, value)
            else:
                result[key] = populate_template(value, data)
        return result
    elif isinstance(template, list):
        return [populate_template(item, data) for item in template]
    else:
        return template


def function_to_json(func, format_template: dict = None) -> dict:
    """
    Converts a Python function into a JSON-serializable dictionary based on a custom format template.

    Args:
        func: The function to be converted.
        format_template: A dictionary specifying the desired output structure.
            Use placeholders like '{name}', '{description}', '{parameters}', '{required}'
            as keys or values to indicate where function data should be inserted.
            If None, a default format is used.

    Returns:
        A dictionary representing the function's signature in the specified format.
    """
    # Default type mapping for annotations
    type_map = {
        str: "string",
        int: "integer",
        float: "number",
        bool: "boolean",
        list: "array",
        dict: "object",
        type(None): "null",
    }

    # Get function signature
    try:
        signature = inspect.signature(func)
    except ValueError as e:
        raise ValueError(
            f"Failed to get signature for function {func.__name__}: {str(e)}"
        )

    # Build parameters dynamically
    parameters = {}
    for param in signature.parameters.values():
        param_type = (
            type_map.get(param.annotation, "string")
            if param.annotation != inspect.Parameter.empty
            else "string"
        )
        param_details = {"type": param_type}
        if param.default != inspect.Parameter.empty:
            param_details["default"] = param.default
        parameters[param.name] = param_details

    # Identify required parameters
    required = [
        param.name
        for param in signature.parameters.values()
        if param.default == inspect.Parameter.empty
    ]

    # Default format if none provided
    if format_template is None:
        format_template = {
            "type": "function",
            "function": {
                "name": "{name}",
                "description": "{description}",
                "parameters": {
                    "type": "object",
                    "properties": "{parameters}",
                    "required": "{required}",
                    "additionalProperties": False,
                },
            },
            "strict": True,
        }

    # Extract function metadata
    func_data = {
        "name": func.__name__,
        "description": (func.__doc__ or "").strip(),
        "parameters": parameters,
        "required": required if required else [],
    }

    return populate_template(format_template, func_data)


def container_to_json(container, format_template: dict = None) -> dict:
    """
    Converts a Container instance into a JSON-serializable dictionary based on a custom format template.

    Args:
        container: The Container instance to be converted.
        format_template: A dictionary specifying the desired output structure.
            Use placeholders like '{name}', '{description}', '{parameters}', '{required}'
            as keys or values to indicate where container data should be inserted.
            If None, a default format is used.

    Returns:
        A dictionary representing the container's attributes in the specified format.
    """
    # Default type mapping for annotations
    type_map = {
        "string": "string",
        "integer": "integer",
        "number": "number",
        "boolean": "boolean",
        "array": "array",
        "object": "object",
        "null": "null",
    }

    # Build parameters dynamically from environment variables
    parameters = {}
    required = []

    for env_var in container.environment:
        param_type = type_map.get(env_var.get("type", "string"), "string")
        param_details = {"type": param_type}
        parameters[env_var["name"]] = param_details
        required.append(env_var["name"])

    # Default format if none provided
    if format_template is None:
        format_template = {
            "type": "container",
            "container": {
                "name": "{name}",
                "description": "{description}",
                "parameters": {
                    "type": "object",
                    "properties": "{parameters}",
                    "required": "{required}",
                    "additionalProperties": False,
                },
            },
            "strict": True,
        }

    # Extract container metadata
    container_data = {
        "name": container.name,
        "description": container.description,
        "parameters": parameters,
        "required": required,
    }

    return populate_template(format_template, container_data)


def extract_key_values(tool_call_output: dict, keys_to_find: list) -> dict:
    """
    Extracts values for specified keys from a tool_call output dictionary.

    Args:
        tool_call_output: The dictionary representing the populated tool_call output.
        keys_to_find: A list of key names to search for (e.g., ["id", "name", "arguments"]).

    Returns:
        A dictionary mapping each specified key to its value(s) from the output.
    """
    result = {
        key: [] for key in keys_to_find
    }  # Initialize with empty lists for each key

    # Helper function to recursively search the dictionary
    def search_dict(data, target_keys):
        if isinstance(data, dict):
            for key, value in data.items():
                if key in target_keys:
                    result[key].append(value)
                search_dict(value, target_keys)
        elif isinstance(data, list):
            for item in data:
                search_dict(item, target_keys)

    # Start the search
    search_dict(tool_call_output, keys_to_find)

    # Clean up the result: single value if found once, list if multiple, omit if not found
    cleaned_result = {}
    for key, values in result.items():
        if values:  # Only include keys that were found
            cleaned_result[key] = values[0] if len(values) == 1 else values

    return cleaned_result


def replace_placeholder(instruction: str, result: bytes) -> str:
    return instruction.replace("{result}", result.decode("utf-8"))


def handover(agent_name: str, description: str, share_context: bool = False):
    """
    Hands over the task to the given agent.

    Args:
        agent_name: name of the agent you want to hand over to
        description: why do you want to handover
    """

    def handover_inner() -> str:
        return agent_name

    handover_inner.__name__ = f"handover_{agent_name}"
    handover_inner.__doc__ = description
    handover_inner.share_context = share_context

    return handover_inner


def write_log(log, logger, message, level="INFO"):
    if log:
        if level == "INFO":
            logger.info(message)
