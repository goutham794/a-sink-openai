import asyncio

from loguru import logger

from typing import Any, Dict, Tuple, Type
from pydantic import BaseModel
import inspect

HEADERS = None
RATE_LIMITER_SEMAPHORE = asyncio.Semaphore(25)


def init_openai(secret, n_jobs=None, json=True):
    global HEADERS
    HEADERS = {"Authorization": f"Bearer {secret['api_key']}"}
    if json:
        HEADERS["Content-Type"] = "application/json"

    if n_jobs:
        global RATE_LIMITER_SEMAPHORE
        RATE_LIMITER_SEMAPHORE = asyncio.Semaphore(n_jobs)


def split_valid_and_invalid_records(records, pydantic_model):
    if invalid_results := [x for x in records if not isinstance(x, pydantic_model)]:
        logger.error(f"There are {len(invalid_results)} failed OpenAI calls")
    else:
        logger.info("All calls were successful")

    valid_results = [x for x in records if isinstance(x, pydantic_model)]
    return valid_results, invalid_results




def is_basemodel_type(cls: Any) -> bool:
    """Check if a type is a Pydantic BaseModel."""
    try:
        return isinstance(cls, type) and issubclass(cls, BaseModel)
    except TypeError:
        return False

def is_dict(obj: Any) -> bool:
    """Check if an object is a dictionary."""
    return isinstance(obj, dict)

def is_list(obj: Any) -> bool:
    """Check if an object is a list."""
    return isinstance(obj, list)

def has_more_than_n_keys(d: Dict[str, Any], n: int) -> bool:
    """Check if dictionary has more than n keys."""
    return len(d.keys()) > n

def resolve_ref(*, root: Dict[str, Any], ref: str) -> Dict[str, Any]:
    """Resolve a JSON Schema $ref."""
    if not ref.startswith('#/'):
        raise ValueError(f"Only internal refs are supported, got: {ref}")
    
    parts = ref.split('/')
    current = root
    for part in parts[1:]:  # Skip the '#'
        if not is_dict(current):
            raise ValueError(f"Expected dictionary while traversing ref {ref}, got {current}")
        current = current.get(part, {})
    
    return current

def ensure_strict_json_schema(
    json_schema: Any,
    *,
    path: Tuple[str, ...],
    root: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Ensure the JSON schema conforms to OpenAI's strict standard.
    
    Args:
        json_schema: The schema to process
        path: Current path in the schema for error reporting
        root: Root schema for resolving refs
        
    Returns:
        Processed schema that conforms to strict standard
    """
    if not is_dict(json_schema):
        raise TypeError(f"Expected dictionary, got {json_schema}; path={path}")
    
    # Process $defs
    defs = json_schema.get("$defs")
    if is_dict(defs):
        for def_name, def_schema in defs.items():
            ensure_strict_json_schema(def_schema, path=(*path, "$defs", def_name), root=root)
    
    # Process definitions (older schema format)
    definitions = json_schema.get("definitions")
    if is_dict(definitions):
        for def_name, def_schema in definitions.items():
            ensure_strict_json_schema(def_schema, path=(*path, "definitions", def_name), root=root)
    
    # Make objects strict
    typ = json_schema.get("type")
    if typ == "object" and "additionalProperties" not in json_schema:
        json_schema["additionalProperties"] = False
    
    # Process object properties
    properties = json_schema.get("properties")
    if is_dict(properties):
        # Make all properties required by default
        json_schema["required"] = [prop for prop in properties.keys()]
        json_schema["properties"] = {
            key: ensure_strict_json_schema(prop_schema, path=(*path, "properties", key), root=root)
            for key, prop_schema in properties.items()
        }
    
    # Process array items
    items = json_schema.get("items")
    if is_dict(items):
        json_schema["items"] = ensure_strict_json_schema(items, path=(*path, "items"), root=root)
    
    # Process unions (anyOf)
    any_of = json_schema.get("anyOf")
    if is_list(any_of):
        json_schema["anyOf"] = [
            ensure_strict_json_schema(variant, path=(*path, "anyOf", str(i)), root=root)
            for i, variant in enumerate(any_of)
        ]
    
    # Process intersections (allOf)
    all_of = json_schema.get("allOf")
    if is_list(all_of):
        if len(all_of) == 1:
            # Single allOf can be flattened
            json_schema.update(ensure_strict_json_schema(all_of[0], path=(*path, "allOf", "0"), root=root))
            json_schema.pop("allOf")
        else:
            json_schema["allOf"] = [
                ensure_strict_json_schema(entry, path=(*path, "allOf", str(i)), root=root)
                for i, entry in enumerate(all_of)
            ]
    
    # Remove None defaults
    if "default" in json_schema and json_schema["default"] is None:
        json_schema.pop("default")
    
    # Handle refs with additional properties
    ref = json_schema.get("$ref")
    if ref and has_more_than_n_keys(json_schema, 1):
        if not isinstance(ref, str):
            raise ValueError(f"Received non-string $ref - {ref}")
        resolved = resolve_ref(root=root, ref=ref)
        if not is_dict(resolved):
            raise ValueError(f"Expected `$ref: {ref}` to resolve to a dictionary but got {resolved}")
        # Properties from the json schema take priority over the ones from the ref
        json_schema.update({**resolved, **json_schema})
        json_schema.pop("$ref")
    
    return json_schema

def to_strict_json_schema(model: Type[BaseModel]) -> Dict[str, Any]:
    """
    Convert a Pydantic model to a strict JSON schema.
    
    Args:
        model: A Pydantic BaseModel class
        
    Returns:
        Dict containing the strict JSON schema
        
    Raises:
        TypeError: If the input is not a Pydantic BaseModel
    """
    if not (inspect.isclass(model) and is_basemodel_type(model)):
        raise TypeError(f"Expected Pydantic BaseModel, got {model}")
    
    schema = model.model_json_schema()
    return ensure_strict_json_schema(schema, path=(), root=schema)

def get_json_response_format(model: Type[BaseModel]) -> Dict[str, Any]:
    """
    Convert a Pydantic BaseModel to OpenAI's response format parameter.
    
    Args:
        model: A Pydantic BaseModel class
        
    Returns:
        Dict containing the JSON schema in OpenAI's expected format
    """
    return {
        "type": "json_schema",
        "json_schema": {
            "schema": to_strict_json_schema(model),
            "name": model.__name__,
            "strict": True,
        },
    }