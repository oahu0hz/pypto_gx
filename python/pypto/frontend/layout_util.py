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

from functools import reduce
from itertools import chain
from typing import Union

def is_tuple(x):
  return isinstance(x, tuple)


def product(a):
  if is_tuple(a):
    return reduce(lambda val,elem : val*product(elem), a, 1)
  else:
    return a


def get_stride(shape):
  if is_tuple(shape):
    shape_dim = len(shape)
    stride = [0] * shape_dim
    stride[shape_dim - 1] = 1
    for i in range(shape_dim - 2, -1, -1):
      stride[i] = stride[i + 1] * shape[i + 1]
    return tuple(stride)



def crd2idx(crd, shape, stride=None):
  if stride is None:
    stride = get_stride(shape)

  if is_tuple(crd):
    if is_tuple(shape):                # tuple tuple tuple
      assert len(crd) == len(shape) and len(crd) == len(stride)
      return sum(crd2idx(c, s, d) for c, s, d in zip(crd, shape, stride))
    else:                              # tuple "int" "int"
      assert False, f"crd={crd}, shape={shape}"           # Error
  else:
    if crd is None:
      crd = 0

    if is_tuple(shape):                # "int" tuple tuple
      assert len(shape) == len(stride)
      result = 0
      for i in range(len(shape)-1):
        result += crd2idx(crd % product(shape[i]), shape[i], stride[i])
        crd = crd // product(shape[i])
      return result + crd2idx(crd, shape[-1], stride[-1])
    else:                              # "int" "int" "int"
      return crd * stride


def slice_(crd: Union[None, tuple, int],
           trg: Union[tuple, int]):
  if is_tuple(crd):
    if is_tuple(trg):                  # tuple tuple
      assert len(crd) == len(trg)
      # match C++ behavior of `filter_tuple` using `tuple_cat(...)`
      return tuple(chain(*filter(lambda x: x != (), [slice_(c, s) for c, s in zip(crd, trg)])))
    else:
      assert False                     # tuple "int" : Error
  elif crd is None:
    # match C++ behavior `return cute::tuple<B>{b};`
    return (trg,)
  else:
    return ()


def has_none(a: Union[None, tuple, int]):
  if is_tuple(a):
    return any(has_none(v) for v in a)
  else:
    return a is None