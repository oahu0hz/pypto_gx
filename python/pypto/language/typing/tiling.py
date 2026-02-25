# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Tiling class utilities for PyPTO Language DSL."""

from pypto.pypto_core import DataType

_PYTHON_TYPE_TO_DTYPE: dict[type, DataType] = {
    int: DataType.INT32,
    float: DataType.FP32,
    bool: DataType.BOOL,
}


def is_tiling_class(cls: object) -> bool:
    """Return True if cls is a user-defined tiling class.

    A tiling class is a plain Python class with at least one field,
    all annotated as int, float, or bool.

    Args:
        cls: Object to check

    Returns:
        True if cls is a valid tiling class
    """
    if not isinstance(cls, type):
        return False
    annotations = getattr(cls, "__annotations__", {})
    if not annotations:
        return False
    return all(v in _PYTHON_TYPE_TO_DTYPE for v in annotations.values())


def get_tiling_fields(cls: type) -> dict[str, DataType]:
    """Return ordered {field_name: DataType} for a validated tiling class.

    Args:
        cls: A tiling class (validated by is_tiling_class)

    Returns:
        Ordered dict mapping field names to their DataType

    Raises:
        ValueError: If cls is not a valid tiling class
    """
    if not is_tiling_class(cls):
        raise ValueError(f"Not a valid tiling class: {cls!r}. All fields must be annotated as int, float, or bool.")
    return {name: _PYTHON_TYPE_TO_DTYPE[py_type] for name, py_type in cls.__annotations__.items()}


__all__ = ["is_tiling_class", "get_tiling_fields"]
