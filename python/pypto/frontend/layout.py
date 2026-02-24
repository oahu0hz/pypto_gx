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
"""
"""

from layout_util import *
from typing import List

class Layout():
  def __init__(self, _shape, _stride=None):
    # 描述Tensor的shape，最高维在index0，最高维在Index Dim - 1，例如shape(128, 256)，128是外层轴（高维），256是内层轴（低维）
    self.shape  = _shape
    if _stride is None:
      self.stride = get_stride(self.shape)
    else:
      self.stride = _stride

  # operator ==
  def __eq__(self, other):
    return self.shape == other.shape and self.stride == other.stride

  # operator len(L)  (len [rank] like tuples)
  def __len__(self):
    if is_tuple(self.shape):
      return len(self.shape)
    else:
      return 1

  # operator ()    (map coord to idx)
  def __call__(self, *args):
    """
    Map a logical coordinate to a linear index (Coord has no Underscore slice operators)
    OR
    Slice the layout and return the sublayout (Coord has an Underscore slice op)

    Follow the same behavior of `Layout::operator(Coord const&)` in cute C++
    """
    if has_none(args):
      if len(args) == 1:
        return Layout(slice_(args[0], self.shape), slice_(args[0], self.stride))
      else:
        return Layout(slice_(args, self.shape), slice_(args, self.stride))
    else:
      if len(args) == 1:
        return crd2idx(args[0], self.shape, self.stride)
      else:
        return crd2idx(args, self.shape, self.stride)

  # operator []    (get-i like tuples)
  def __getitem__(self, i):
    if is_tuple(self.shape):
      return Layout(self.shape[i], self.stride[i])
    else:
      assert i == 0
      return Layout(self.shape, self.stride)

  # size(layout)   Size of the domain
  def size(self):
    return product(self.shape)

  # cosize(layout)   Size of the codomain
  def cosize(self):
    return self(self.size() - 1) + 1

  # print and str
  def __str__(self):
    return f"{self.shape}:{self.stride}"

  # error msgs and representation
  def __repr__(self):
    return f"Layout({self.shape},{self.stride})"


def logical_divide(layout_a: Layout, layout_b: Layout) -> Layout:
    """
    生成描述块网格的Layout
    
    参数:
        layout_a: 原始Layout (例如: shape=[2048, 2048], stride=[2048, 1])
        layout_b: 块大小的Layout (例如: shape=[128, 128], stride=[128, 1])
    
    返回:
        描述块网格的Layout，其中:
        - shape: 每个维度的块数
        - stride: 块之间的内存步长
    """
    if len(layout_a) != len(layout_b):
        raise ValueError(f"维度不匹配: {len(layout_a)} != {len(layout_b)}")
    
    # 计算每个维度的块数（向下取整，忽略不整除的部分）
    blocks_per_dim = []
    for i in range(len(layout_a)):
        blocks_per_dim.append((layout_a.shape[i] + layout_b.shape[i] - 1) // layout_b.shape[i])
    
    # 计算块之间的步长
    # 关键：块之间的步长 = 块大小 × 原始步长
    block_strides = []
    for i in range(len(layout_a)):
        block_strides.append(layout_b.shape[i] * layout_a.stride[i])
    
    return Layout(_shape=tuple(blocks_per_dim), _stride=tuple(block_strides))


def test_layout01():
    layout_a = Layout(_shape=(12, 8), _stride=(8, 1))
    layout_b = Layout(_shape=(3, 4), _stride=(4, 1))

    divided = logical_divide(layout_a, layout_b)
    assert divided.shape == (4, 2)
    assert divided.stride == (24, 4)
    print("Test_layout01 Pass")


def test_layout02():
    layout_a = Layout(_shape=(8, 32, 2048, 2048))
    layout_b = Layout(_shape=(1, 1, 128, 128))
    assert layout_a.stride == (134217728, 4194304, 2048, 1)
    divided = logical_divide(layout_a, layout_b)
    assert divided.shape == (8, 32, 16, 16)
    assert divided.stride == (134217728, 4194304, 262144, 128)
    print("Test_layout02 Pass")


def test_layout03():
    layout_a = Layout(_shape=(8, 32, 2050, 2048))
    layout_b = Layout(_shape=(1, 1, 128, 128))
    assert layout_a.stride == (134348800, 4198400, 2048, 1)

    divided = logical_divide(layout_a, layout_b)
    assert divided.shape == (8, 32, 17, 16)
    assert divided.stride == (134348800, 4198400, 262144, 128)
    print("Test_layout03 Pass")


def test_layout04():
    layout_a = Layout(_shape=(17, 55, 3333, 7777))
    layout_b = Layout(_shape=(4, 3, 128, 128))
    assert layout_a.stride == (1425640755, 25920741, 7777, 1)

    divided = logical_divide(layout_a, layout_b)
    assert divided.shape == ((5, 19, 27, 61))
    assert divided.stride == (5702563020, 77762223, 995456, 128)
    print("Test_layout04 Pass")



if __name__ == "__main__":
  test_layout01()
  test_layout02()
  test_layout03()
  test_layout04()