#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import torch
from layout import *

class Tensor:
    def __init__(self, tensor: torch.Tensor, assumed_align=None):
        """初始构造函数：仅从真实的 torch.Tensor 构造"""
        self._dtype = str(tensor.dtype)
        self._addr = tensor.data_ptr()
        self._layout = Layout(tensor.shape, tensor.stride)
        self._assumed_align = assumed_align

    @classmethod
    def _create_derived(cls, addr, layout, dtype, assumed_align):
        """内部辅助方法：用于从现有信息派生新 Tensor 实例"""
        # 使用 __new__ 跳过原来的 __init__ 逻辑
        instance = cls.__new__(cls)
        instance._addr = addr
        instance._layout = layout
        instance._dtype = dtype
        instance._assumed_align = assumed_align
        return instance

    def divide(self, layout: Layout, tile: Layout) -> Tensor:
        """
        基于传入的 layout 和 tile 进行逻辑切分
        """
        new_layout = logical_divide(layout, tile)
        # 返回一个指向相同地址，但应用了新 layout 的对象
        return self._create_derived(
            self._addr, 
            new_layout, 
            self._dtype, 
            self._assumed_align
        )

    def divide(self, tile: Layout) -> Tensor:
        """
        基于对象自身的 layout 和传入的 tile 进行逻辑切分
        """
        new_layout = logical_divide(self._layout, tile)
        return self._create_derived(
            self._addr, 
            new_layout, 
            self._dtype, 
            self._assumed_align
        )

    @property
    def dtype(self):
        return self._dtype

    @property
    def addr(self):
        return hex(self._addr)

    @property
    def layout(self):
        return f"Layout({self._layout.shape},{self._layout.stride})"

    @property
    def shape(self):
        return self._layout.shape

    @property
    def stride(self):
        return self._layout.stride


def from_torch(tensor: torch.Tensor, assumed_align=None):
    """从torch.Tensor构造张量"""
    return Tensor(tensor, assumed_align=assumed_align)

