# import inspect
#
#
# def function_to_json(func) -> dict:
#     """
#     Converts a Python function into a JSON-serializable dictionary
#     that describes the function's signature, including its name,
#     description, and parameters.
#
#     Args:
#         func: The function to be converted.
#
#     Returns:
#         A dictionary representing the function's signature in JSON format.
#     """
#     type_map = {
#         str: "string",
#         int: "integer",
#         float: "number",
#         bool: "boolean",
#         list: "array",
#         dict: "object",
#         type(None): "null",
#     }
#
#     try:
#         signature = inspect.signature(func)
#     except ValueError as e:
#         raise ValueError(
#             f"Failed to get signature for function {func.__name__}: {str(e)}"
#         )
#
#     parameters = {}
#     for param in signature.parameters.values():
#         try:
#             param_type = type_map.get(param.annotation, "string")
#         except KeyError as e:
#             raise KeyError(
#                 f"Unknown type annotation {param.annotation} for parameter {param.name}: {str(e)}"
#             )
#         parameters[param.name] = {"type": param_type}
#
#     required = [
#         param.name
#         for param in signature.parameters.values()
#         if param.default == inspect._empty
#     ]
#     return {
#         "type": "function",
#         "function": {
#             "name": func.__name__,
#             "description": func.__doc__ or "",
#             "parameters": {
#                 "type": "object",
#                 "properties": parameters,
#                 "required": required,
#                 "additionalProperties": False,
#             },
#             "strict": True,
#         },
#     }
import inspect
import pprint


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
        raise ValueError(f"Failed to get signature for function {func.__name__}: {str(e)}")

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
                }
            },
            "strict": True,
        }

    # Extract function metadata
    func_data = {
        "name": func.__name__,
        "description": (func.__doc__ or "").strip(),
        "parameters": parameters,
        "required": required if required else []
    }

    # Helper function to recursively populate the template
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

    result = populate_template(format_template, func_data)
    pprint.pp(result)
    return result


def extract_key_values(tool_call_output: dict, tool_call_template: dict) -> dict:
    """
    Extracts values for keys defined in a tool_call template from a tool_call output.

    Args:
        tool_call_output: The dictionary representing the populated tool_call output.
        tool_call_template: The template dictionary with placeholders (e.g., "{id}", "{name}").

    Returns:
        A dictionary mapping each key from the template placeholders to its value(s) in the output.
    """
    # Step 1: Extract keys_to_find from the template by looking for {key} patterns
    keys_to_find = set()

    def find_placeholder_keys(data):
        if isinstance(data, dict):
            for value in data.values():
                if isinstance(value, str) and value.startswith("{") and value.endswith("}"):
                    keys_to_find.add(value[1:-1])  # Extract the key inside {}
                find_placeholder_keys(value)
        elif isinstance(data, list):
            for item in data:
                find_placeholder_keys(item)

    find_placeholder_keys(tool_call_template)

    # Step 2: Search the tool_call_output for these keys
    result = {key: [] for key in keys_to_find}  # Initialize with empty lists

    def search_dict(data, target_keys):
        if isinstance(data, dict):
            for key, value in data.items():
                if key in target_keys:
                    result[key].append(value)
                search_dict(value, target_keys)
        elif isinstance(data, list):
            for item in data:
                search_dict(item, target_keys)

    search_dict(tool_call_output, keys_to_find)

    # Step 3: Clean up the result: single value if found once, list if multiple
    cleaned_result = {}
    for key, values in result.items():
        if values:  # Only include keys that were found
            cleaned_result[key] = values[0] if len(values) == 1 else values

    return cleaned_result
