# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Pointer operations for PyPTO Language DSL (ptoas scene).

These ops create pointer arithmetic and tensor views from raw pointers.
They emit PTO MLIR instructions (pto.addptr, pto.make_tensor_view) and are
distinct from the orchestration-only tensor ops.
"""

from collections.abc import Sequence

__all__ = [
    "make_tensor",
    "addptr",
]

from pypto.ir.op import ptr_ops as _ir_ops
from pypto.pypto_core.ir import Expr

from ..typing import IntLike, Ptr, Scalar, Tensor


def _normalize_intlike(seq: Sequence[IntLike]) -> list[int | Expr]:
    """Unwrap Scalar elements to Expr so the sequence matches C++ binding types."""
    return [elem.unwrap() if isinstance(elem, Scalar) else elem for elem in seq]


def make_tensor(ptr: Ptr, shape: Sequence[IntLike], stride: Sequence[IntLike]) -> Tensor:
    """Create a tensor view from a raw pointer with explicit shape and strides.

    Args:
        ptr: Raw pointer to global memory (pl.Ptr[dtype] parameter)
        shape: New shape dimensions
        stride: Stride per dimension

    Returns:
        Tensor wrapping the make_tensor_view operation
    """
    call_expr = _ir_ops.make_tensor(ptr.unwrap(), _normalize_intlike(shape), _normalize_intlike(stride))
    return Tensor(expr=call_expr)


def addptr(ptr: Ptr, offset: IntLike) -> Ptr:
    """Advance a raw pointer by an integer offset.

    Args:
        ptr: Raw pointer to global memory (pl.Ptr[dtype] parameter)
        offset: Integer offset to advance the pointer by

    Returns:
        Ptr with the same element dtype, advanced by offset
    """
    offset_val = offset.unwrap() if isinstance(offset, Scalar) else offset
    call_expr = _ir_ops.addptr(ptr.unwrap(), offset_val)
    return Ptr(expr=call_expr)
