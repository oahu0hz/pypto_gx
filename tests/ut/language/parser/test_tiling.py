# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Unit tests for tiling parameter support in the PyPTO language DSL."""

import pypto.language as pl
import pytest
from pypto import ir
from pypto.language.parser.diagnostics import ParserSyntaxError, ParserTypeError, UndefinedVariableError
from pypto.language.typing.tiling import get_tiling_fields, is_tiling_class
from pypto.pypto_core import DataType


class TestTilingUtilities:
    """Tests for is_tiling_class and get_tiling_fields utilities."""

    def test_is_tiling_class_with_int_fields(self):
        class T:
            x: int
            y: int

        assert is_tiling_class(T) is True

    def test_is_tiling_class_with_float_fields(self):
        class T:
            a: float

        assert is_tiling_class(T) is True

    def test_is_tiling_class_with_bool_fields(self):
        class T:
            flag: bool

        assert is_tiling_class(T) is True

    def test_is_tiling_class_with_mixed_valid_fields(self):
        class T:
            x: int
            y: float
            enabled: bool

        assert is_tiling_class(T) is True

    def test_is_tiling_class_with_invalid_field_type(self):
        class T:
            x: int
            name: str  # str is not a valid tiling field type

        assert is_tiling_class(T) is False

    def test_is_tiling_class_with_no_annotations(self):
        class T:
            pass

        assert is_tiling_class(T) is False

    def test_is_tiling_class_with_non_class(self):
        assert is_tiling_class(42) is False
        assert is_tiling_class("string") is False
        assert is_tiling_class(None) is False

    def test_get_tiling_fields_dtype_mapping(self):
        class T:
            x: int
            y: float
            flag: bool

        fields = get_tiling_fields(T)
        assert fields == {"x": DataType.INT32, "y": DataType.FP32, "flag": DataType.BOOL}

    def test_get_tiling_fields_preserves_order(self):
        class T:
            c: float
            a: int
            b: bool

        fields = get_tiling_fields(T)
        assert list(fields.keys()) == ["c", "a", "b"]


class TestTilingParameter:
    """Tests for tiling parameter parsing in @pl.function."""

    def test_tiling_only_param_flattens_to_scalar_params(self):
        class Tiling:
            x: int
            y: float

        @pl.function
        def kernel(tiling: Tiling) -> pl.Scalar[pl.INT32]:
            result: pl.Scalar[pl.INT32] = tiling.x
            return result

        assert isinstance(kernel, ir.Function)
        assert len(kernel.params) == 2
        param_names = [p.name for p in kernel.params]
        assert "tiling_x" in param_names
        assert "tiling_y" in param_names

    def test_tiling_scalar_dtypes_are_correct(self):
        class Tiling:
            n: int
            scale: float
            flag: bool

        @pl.function
        def kernel(tiling: Tiling) -> pl.Scalar[pl.INT32]:
            result: pl.Scalar[pl.INT32] = tiling.n
            return result

        assert isinstance(kernel, ir.Function)
        assert len(kernel.params) == 3
        param_map = {p.name: p for p in kernel.params}
        assert isinstance(param_map["tiling_n"].type, ir.ScalarType)
        assert param_map["tiling_n"].type.dtype == DataType.INT32
        assert param_map["tiling_scale"].type.dtype == DataType.FP32
        assert param_map["tiling_flag"].type.dtype == DataType.BOOL

    def test_tensors_plus_tiling_last(self):
        class Tiling:
            n: int
            m: int

        @pl.function
        def kernel(
            x: pl.Tensor[[64], pl.FP32],
            y: pl.Tensor[[64], pl.FP32],
            tiling: Tiling,
        ) -> pl.Tensor[[64], pl.FP32]:
            return x

        assert isinstance(kernel, ir.Function)
        # 2 tensor params + 2 tiling fields = 4 total
        assert len(kernel.params) == 4
        param_names = [p.name for p in kernel.params]
        assert param_names[0] == "x"
        assert param_names[1] == "y"
        assert "tiling_n" in param_names
        assert "tiling_m" in param_names

    def test_tiling_name_not_registered_in_scope(self):
        """Bare tiling name used without field access raises UndefinedVariableError."""

        class Tiling:
            x: int

        with pytest.raises(UndefinedVariableError):

            @pl.function
            def kernel(tiling: Tiling) -> pl.Scalar[pl.INT32]:
                return tiling  # type: ignore[return-value]

    def test_tiling_field_access_resolves_to_correct_var(self):
        """Accessing tiling.x in the body resolves to the flattened IR var."""

        class Tiling:
            x: int

        @pl.function
        def kernel(tiling: Tiling) -> pl.Scalar[pl.INT32]:
            result: pl.Scalar[pl.INT32] = tiling.x
            return result

        assert isinstance(kernel, ir.Function)
        assert len(kernel.params) == 1
        assert kernel.params[0].name == "tiling_x"

    def test_tiling_registry_reset_between_functions(self):
        """Tiling registry is reset for each new function, preventing leakage."""

        class Tiling:
            n: int

        @pl.function
        def func1(tiling: Tiling):
            x: pl.Scalar[pl.INT32] = tiling.n
            return x

        # Second function should not see tiling from first function
        @pl.function
        def func2(x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
            return x

        assert isinstance(func1, ir.Function)
        assert isinstance(func2, ir.Function)


class TestTilingErrors:
    """Tests for error cases in tiling parameter parsing."""

    def test_tiling_not_last_raises_parser_syntax_error(self):
        """Tiling parameter that is not the last param raises ParserSyntaxError."""

        class Tiling:
            x: int

        with pytest.raises(ParserSyntaxError, match="must be the last parameter"):

            @pl.function
            def kernel(
                tiling: Tiling,  # Not last!
                x: pl.Tensor[[64], pl.FP32],
            ):
                pass

    def test_multiple_tiling_params_raises_parser_syntax_error(self):
        """More than one tiling parameter raises ParserSyntaxError."""

        class TilingA:
            x: int

        class TilingB:
            y: float

        with pytest.raises(ParserSyntaxError, match="at most 1"):

            @pl.function
            def kernel(
                ta: TilingA,
                tb: TilingB,
            ):
                pass

    def test_nonexistent_tiling_field_raises_parser_type_error(self):
        """Accessing a field that doesn't exist on tiling raises ParserTypeError."""

        class Tiling:
            x: int

        with pytest.raises(ParserTypeError, match="has no field"):

            @pl.function
            def kernel(tiling: Tiling) -> pl.Scalar[pl.INT32]:
                result: pl.Scalar[pl.INT32] = tiling.nonexistent  # type: ignore[attr-defined]
                return result
