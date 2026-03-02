# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Pointer operations for PyPTO IR (ptoas scene).

These ops emit PTO MLIR instructions (pto.addptr, pto.make_tensor_view) and are
distinct from the orchestration-only tensor ops.
"""

from collections.abc import Sequence

from pypto.pypto_core import DataType
from pypto.pypto_core import ir as _ir_core
from pypto.pypto_core.ir import Call, Expr, Span

from ..utils import _get_span_or_capture, _normalize_expr, _to_make_tuple


def make_tensor(
    ptr: Expr,
    shape: Sequence[int | Expr] | _ir_core.MakeTuple,
    stride: Sequence[int | Expr] | _ir_core.MakeTuple,
    span: Span | None = None,
) -> Call:
    """Create a tensor view from a raw pointer with explicit shape and strides.

    Emits ``pto.make_tensor_view`` in the ptoas codegen.

    Args:
        ptr: Raw pointer expression (must have PtrType)
        shape: New shape dimensions (int or Expr per dimension), or a MakeTuple
        stride: Stride per dimension (int or Expr), or a MakeTuple
        span: Optional source span for debugging (auto-captured if not provided)

    Returns:
        Call expression creating a tensor view with the given shape and strides
    """
    actual_span = _get_span_or_capture(span)
    shape_tuple = _to_make_tuple(shape, actual_span)
    stride_tuple = _to_make_tuple(stride, actual_span)
    return _ir_core.create_op_call("ptr.make_tensor", [ptr, shape_tuple, stride_tuple], {}, actual_span)


def addptr(ptr: Expr, offset: int | Expr, span: Span | None = None) -> Call:
    """Advance a raw pointer by an integer offset.

    Emits ``pto.addptr`` in the ptoas codegen.

    Args:
        ptr: Raw pointer expression (must have PtrType)
        offset: Integer offset (int or Expr with integer/index ScalarType)
        span: Optional source span for debugging (auto-captured if not provided)

    Returns:
        Call expression for pointer arithmetic (same PtrType as input)
    """
    actual_span = _get_span_or_capture(span)
    if isinstance(offset, int):
        offset_expr = _normalize_expr(offset, actual_span, int_dtype=DataType.INDEX)
    else:
        offset_expr = offset
    return _ir_core.create_op_call("ptr.addptr", [ptr, offset_expr], {}, actual_span)
