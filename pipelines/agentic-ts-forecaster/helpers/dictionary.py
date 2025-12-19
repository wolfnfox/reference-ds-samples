from typing import Any, Dict, Type, TypeVar

T = TypeVar('T')

def get_or_default(d: Dict[str, Any], key: str, value_type: Type[T], default: T) -> T | Any:
    """Helper function to get a value from a dictionary or return a default."""
    value = d.get(key, default)

    if value is not None and isinstance(value, value_type):
        return value
    
    if default is not None:
        return default
    
    return _get_type_default(value_type)

def _get_type_default(value_type: Type[T]) -> T | None:
    defaults: Dict[type, Any] = {
        int: 0,
        float: 0.0,
        str: "",
        bool: False,
        list: [],
        dict: {},
        set: set()
    }
    return defaults.get(value_type, None)
