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
import math
import ctypes
import numpy as np
from typing import Dict, Optional
from abc import ABC

# 1. 定义结构体，必须继承自 ctypes.Structure
class TilingDataStructSample(ctypes.Structure):
    # _fields_ 定义了内存布局，顺序必须与 C 语言结构体完全一致
    _fields_ = [
        ("block_size", ctypes.c_uint32),   # 对应 C 的 uint32_t
        ("stride", ctypes.c_uint32),       # 对应 C 的 uint32_t
        ("gain", ctypes.c_float),          # 对应 C 的 float
        ("core_num", ctypes.c_int32)       # 对应 C 的 int32_t
    ]

    def __repr__(self):
        return f"<TilingData: block={self.block_size}, stride={self.stride}, gain={self.gain}>"

class TilingBase(ABC):
    def __init__(self, struct_cls: type[ctypes.Structure]):
        self._struct_cls = struct_cls
        self._tiling_data = struct_cls()
        self._core_num: Optional[int] = None
        self._workspace_size: int = 0
        self._workspace_tensor: Optional[torch.Tensor] = None
        self._workspace_ptr = None
        self._device = None

    def set_tiling_params(self, **kwargs):
        """
        动态设置结构体参数
        例如：set_tiling_params(block_size=128, stride=64)
        """
        for key, value in kwargs.items():
            if hasattr(self._data_obj, key):
                setattr(self._data_obj, key, value)
            else:
                raise AttributeError(f"tiling结构体中没有字段: {key}")

    def alloc_workspace(self):
        """
        申请指定大小的 workspace 内存空间
        """
        physical_workspace_size = 0
        if self.workspace_size > 0 and self._workspace_size is not None:
            physical_workspace_size += self._workspace_size
        if self._tiling_size > 0:
            physical_workspace_size += self._tiling_size
        
        # 如果已经申请过，先释放旧的
        if self._workspace_ptr:
            return

        try:
            # 使用 ctypes 申请一段连续的系统内存 (匿名缓冲区)
            # 在实际 NPU 开发中，这里会替换为底层驱动的 Malloc 接口
            # self._workspace_ptr = ctypes.cast(
            #     (ctypes.c_byte * size)(), 
            #     ctypes.c_void_p
            # )
            # self.workspace_size = size
            self._workspace_tensor = torch.empty(physical_workspace_size, dtype=torch.uint8, device=self._device)
            self._workspace_ptr = self._workspace_tensor.data_ptr()
            print(f"[Memory] 成功申请 {physical_workspace_size} 字节 workspace, 地址: {self._workspace_ptr.value:#x}")
            return self._workspace_ptr.value if self._workspace_ptr.value else 0
        except Exception as e:
            print(f"[Error] 内存申请失败: {e}")
            return

    def copy_tiling_to_workspace(self):
        if self._workspace_tensor is None:
            raise RuntimeError("Workspace not allocated")

        struct_size = ctypes.sizeof(self._data_obj)
        ptr = self._workspace_ptr + self._workspace_size

        # 执行拷贝：注意，如果 Tensor 在 GPU 上，普通的 memmove 会报错
        # 我们需要先判断设备类型
        if self._workspace_tensor.is_cuda or "npu" in str(self._workspace_tensor.device):
            # 如果是 GPU/NPU，需要将 Python 结构体转为 bytes 再用 torch 拷贝
            src_bytes = self._binary_data
            # 将 bytes 转换为 tensor 并传送到对应设备
            src_tensor = torch.from_numpy(np.frombuffer(src_bytes, dtype=np.uint8)).to(self._workspace_tensor.device)
            
            # 使用切片赋值进行内存拷贝
            self._workspace_tensor[self._workspace_size : self._workspace_size + struct_size] = src_tensor
        else:
            # CPU 场景直接使用 ctypes.memmove
            ctypes.memmove(ptr, ctypes.addressof(self._data_obj), struct_size)

    @property
    def _tiling_size(self) -> int:
        """获取 C 结构体在内存中的实际字节数"""
        return ctypes.sizeof(self._data_obj)

    @property
    def _binary_data(self) -> bytes:
        """将结构体转换为二进制字节流，用于传递给底层硬件"""
        return bytes(self._data_obj)

    @property
    def core_num(self):
        return self._core_num

    @core_num.setter
    def core_num(self, core_num):
        if not isinstance(core_num, int):
            raise TypeError(f"core_num必须是整数，而不是{type(core_num).__name__}")
        if core_num <= 0:
            raise ValueError("core_num必须大于0。")
        self._core_num = core_num

    @property
    def workspace_size(self) -> int:
        return self._workspace_size

    @workspace_size.setter
    def workspace_size(self, workspace_size):
        """设置算子需要的workspace空间大小，实际的workspace空间大小会做512bytes对齐"""
        if not isinstance(workspace_size, int):
            raise TypeError(f"workspace_size必须是整数，而不是{type(workspace_size).__name__}")
        if workspace_size < 0:
            raise ValueError(f"Workspace空间大小不能为负数。")
        self._workspace_size = math.ceil(workspace_size / 512) * 512
