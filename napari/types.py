from typing import Callable, List, Tuple, Union, Any, Dict

# layer data may be: (data,) (data, meta), or (data, meta, layer_type)
# using "Any" for the data type for now
LayerData = Union[Tuple[Any], Tuple[Any, Dict], Tuple[Any, Dict, str]]
ReaderFunction = Callable[[str], List[LayerData]]
