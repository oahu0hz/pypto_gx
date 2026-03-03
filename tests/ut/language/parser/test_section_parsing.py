# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Unit tests for parsing SectionStmt with pl.section_vector() and pl.section_cube() syntax."""

import pypto.language as pl
import pytest
from pypto import ir


class TestSectionVectorParsing:
    """Test parsing of with pl.section_vector(): syntax."""

    def test_parse_simple_section_vector(self):
        """Test parsing a simple Vector section."""

        @pl.program
        class TestProgram:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                with pl.section_vector():
                    y: pl.Tensor[[64], pl.FP32] = pl.add(x, x)
                return y

        assert TestProgram is not None
        assert len(TestProgram.functions) == 1

        main_func = list(TestProgram.functions.values())[0]
        assert main_func.name == "main"
        assert isinstance(main_func.body, ir.SeqStmts)

    def test_parse_section_vector_printing(self):
        """Test that printed Vector section contains correct syntax."""

        @pl.program
        class TestProgram:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                with pl.section_vector():
                    y: pl.Tensor[[64], pl.FP32] = pl.add(x, x)
                return y

        printed = ir.python_print(TestProgram)
        assert "with pl.section_vector():" in printed


class TestSectionCubeParsing:
    """Test parsing of with pl.section_cube(): syntax."""

    def test_parse_simple_section_cube(self):
        """Test parsing a simple Cube section."""

        @pl.program
        class TestProgram:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                with pl.section_cube():
                    y: pl.Tensor[[64], pl.FP32] = pl.mul(x, x)
                return y

        assert TestProgram is not None
        assert len(TestProgram.functions) == 1

        main_func = list(TestProgram.functions.values())[0]
        assert main_func.name == "main"
        assert isinstance(main_func.body, ir.SeqStmts)

    def test_parse_section_cube_printing(self):
        """Test that printed Cube section contains correct syntax."""

        @pl.program
        class TestProgram:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                with pl.section_cube():
                    y: pl.Tensor[[64], pl.FP32] = pl.mul(x, x)
                return y

        printed = ir.python_print(TestProgram)
        assert "with pl.section_cube():" in printed


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
