def enum_defaults(schema):
    try:
        return schema["enum"][0]
    except IndexError:
        return None


def object_defaults(schema):
    if "properties" in schema:
        return {
            k: compute_defaults(s) for k, s in schema["properties"].items()
        }
    else:
        return None


def array_defaults(schema):
    items_schema = schema['items']
    if isinstance(items_schema, dict):
        return []

    return [compute_defaults(s) for s in schema["items"]]


def compute_defaults(schema):
    if "default" in schema:
        return schema["default"]

    # Enum
    if "enum" in schema:
        return enum_defaults(schema)

    schema_type = schema["type"]
    if schema_type == "object":
        return object_defaults(schema)

    elif schema_type == "array":
        return array_defaults(schema)

    return None
