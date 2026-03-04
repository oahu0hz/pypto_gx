# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Unit tests for SectionStmt class."""

import pytest
from pypto import DataType, ir


class TestSectionStmtVector:
    """Test SectionStmt with Vector kind."""

    def test_section_stmt_vector_construction(self):
        """Test basic SectionStmt construction with Vector kind."""
        span = ir.Span("test.py", 1, 1, 1, 10)
        var_x = ir.Var("x", ir.TensorType([64], DataType.FP32), span)
        var_y = ir.Var("y", ir.TensorType([64], DataType.FP32), span)

        body = ir.AssignStmt(var_y, var_x, span)
        section = ir.SectionStmt(ir.SectionKind.Vector, body, span)

        assert section.section_kind == ir.SectionKind.Vector
        assert isinstance(section.body, ir.AssignStmt)

    def test_section_stmt_vector_structural_equality(self):
        """Test structural equality for Vector SectionStmt."""
        span = ir.Span("test.py", 1, 1, 1, 10)
        var_x = ir.Var("x", ir.TensorType([64], DataType.FP32), span)
        var_y = ir.Var("y", ir.TensorType([64], DataType.FP32), span)

        body1 = ir.AssignStmt(var_y, var_x, span)
        section1 = ir.SectionStmt(ir.SectionKind.Vector, body1, span)

        body2 = ir.AssignStmt(var_y, var_x, span)
        section2 = ir.SectionStmt(ir.SectionKind.Vector, body2, span)

        assert ir.structural_equal(section1, section2)


class TestSectionStmtCube:
    """Test SectionStmt with Cube kind."""

    def test_section_stmt_cube_construction(self):
        """Test basic SectionStmt construction with Cube kind."""
        span = ir.Span("test.py", 1, 1, 1, 10)
        var_x = ir.Var("x", ir.TensorType([64], DataType.FP32), span)
        var_y = ir.Var("y", ir.TensorType([64], DataType.FP32), span)

        body = ir.AssignStmt(var_y, var_x, span)
        section = ir.SectionStmt(ir.SectionKind.Cube, body, span)

        assert section.section_kind == ir.SectionKind.Cube
        assert isinstance(section.body, ir.AssignStmt)

    def test_section_stmt_cube_structural_equality(self):
        """Test structural equality for Cube SectionStmt."""
        span = ir.Span("test.py", 1, 1, 1, 10)
        var_x = ir.Var("x", ir.TensorType([64], DataType.FP32), span)
        var_y = ir.Var("y", ir.TensorType([64], DataType.FP32), span)

        body1 = ir.AssignStmt(var_y, var_x, span)
        section1 = ir.SectionStmt(ir.SectionKind.Cube, body1, span)

        body2 = ir.AssignStmt(var_y, var_x, span)
        section2 = ir.SectionStmt(ir.SectionKind.Cube, body2, span)

        assert ir.structural_equal(section1, section2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
