# Copyright (c) Huawei Technologies Co., Ltd. 2025.
import os
import argparse
import torch

import tilelang
import tilelang.language as T

torch.npu.set_device(7)
tilelang.cache.clear_cache()

num = 4
M = 4
N = 8

@tilelang.jit(target="npuir")
def vec_for_reduce(dtype = "float32"):
    BLOCK_SIZE = 1
    @T.prim_func
    def vecForReduce(
            Input: T.Tensor((num, M, N), dtype),
            Output: T.Tensor((M, 1), dtype)
    ):
        with T.Kernel(BLOCK_SIZE, is_npu=True) as (cid, _):
            reduce_result = T.alloc_shared([M, 1], dtype=dtype)
            temp = T.alloc_shared([M, 1], dtype=dtype)
            src = T.alloc_shared([M, N], dtype=dtype)

            value_zero = 0
            T.npuir_brc(value_zero, reduce_result)
            for i in T.Pipelined(num):
                T.copy(Input[i, :, :], src)
                T.npuir_reduce(src, temp, 1, "sum", clear = True)
                T.npuir_add(temp, reduce_result, reduce_result)

            T.copy(reduce_result, Output)

    return vecForReduce

def test_for_reduce():
    input = torch.randn([num, M, N], dtype=torch.float32).npu()
    output = torch.randn([M, 1], dtype=torch.float32).npu()
    torch.manual_seed(88888888)
    print("input")
    print(input)
    vecForReduce = vec_for_reduce()
    vecForReduce(input, output)

    ref_output = torch.sum(input, dim=2, keepdim=True).sum(dim=0)

    print("output")
    print(output)

    print("ref_output")
    print(ref_output)

    torch.testing.assert_close(output, ref_output, rtol=1e-2, atol=1e-2)
    print("\033[92mAll check passed!\033[0m")


if __name__ == "__main__":
    os.environ['TILELANG_ASCEND_MODE'] = 'Developer'
    print("Running in developer mode")
    print(">>>>>> Test reduce in T.pipelined <<<<<<")
    test_for_reduce()