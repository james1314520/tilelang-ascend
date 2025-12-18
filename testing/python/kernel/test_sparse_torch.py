import torch

def next_power_of_2(n):
    if n <= 0:
        return 1
    
    # 如果n已经是2的幂，返回下一个2的幂
    if (n & (n - 1)) == 0:
        return n * 2
    
    # 使用位运算找到下一个2的幂
    n -= 1
    n |= n >> 1
    n |= n >> 2
    n |= n >> 4
    n |= n >> 8
    n |= n >> 16
    n |= n >> 32
    n += 1
    
    return n

def get_dtype_min(dtype):
    if dtype == torch.bool:
        return False
    elif dtype.is_floating_point:
        return torch.finfo(dtype).min
    else:
        return torch.iinfo(dtype).min


def sparse_attention_fwd(
    heads, # H = 128
    dim, # part DQK 512
    tail_dim, # part DQK 64
    topk, # topk = 2048
    kv_group=1, # HKV = 1
    sm_scale=None,
    is_causal=True,
    CP0=True,
    block_I=64,
    num_stages=2,
    threads=256,
):
    # assert dim is 2**x
    # assert tail_dim is 2**x
    # assert topk is block_I*x
    assert is_causal == True, "non-casual is not supported"
    if sm_scale is None:
        sm_scale = (1.0 / (dim + tail_dim)) ** 0.5 * 1.44269504  # log2(e)
    else:
        sm_scale = sm_scale * 1.44269504  # log2(e)

    batch = 1 # ?? T.symbolic("batch") # div batch 1
    seq_len = 10 # ?? 先用10来做计算，T.symbolic("seq_len") # div S 4096
    seq_len_kv = 32 # ?? T.symbolic("seq_len_kv") div SKV 32768
    
    head_kv = heads // kv_group
    q_shape = [batch, seq_len, heads, dim + tail_dim]
    kv_shape = [batch, seq_len_kv, kv_group, dim + tail_dim]
    o_shape = [batch, seq_len, heads, dim]
    indices_shape = [batch, seq_len, kv_group, topk]
    lse_shape = [batch, seq_len, heads]
    indices_dtype = "int32"
    dtype = "float16"
    accum_dtype = "float"

    G = kv_group
    H = head_kv
    padded_H = max(next_power_of_2(head_kv), 16)
    if padded_H != H:
        assert (
            kv_group == 1
        ), "here we solve the H padding automically, other wise you should handle Q copy and Output copy with your mask (when kv_group == 1, use g_i * padded_H:(g_i+1) * padded_H would be handled automically)"
    BI = block_I
    NI = topk // block_I # ?? tilelang.cdiv(topk, block_I)
    D = dim
    D_tail = tail_dim

    if head_kv > 64:
        assert head_kv % 64 == 0, "head_kv should be a multiple of 64"
        REPLICATE_H = head_kv // 64
    else:
        REPLICATE_H = 1

    H_per_block = padded_H if REPLICATE_H == 1 else 64

    def main(
        Q: torch.tensor(q_shape),  # [batch, seq_len, heads, dim + tail_dim] "bfloat16"
        KV: torch.tensor(kv_shape),  # [batch, seq_len_kv, kv_group, dim + tail_dim] "bfloat16"
        Indices: torch.tensor(indices_shape), # [batch, seq_len, kv_group, topk] "int32"
        Output: torch.tensor(o_shape),  # [batch, seq_len, heads, dim] "bfloat16"
        Lse: torch.tensor(lse_shape),  # [batch, seq_len, heads] "float"
    ):
        #print("seq_len:", seq_len)
        for bx in range(1):
        #for bx in range(seq_len * REPLICATE_H):
            for by in range(batch):  # 1
                for bz in range(kv_group):  # 1
                    Q_shared = torch.zeros((H_per_block, D), dtype=torch.float16)
                    Q_tail_shared = torch.zeros((H_per_block, D_tail), dtype=torch.float16)
                    KV_shared = torch.zeros((BI, D), dtype=torch.float16)
                    K_tail_shared = torch.zeros((BI, D_tail), dtype=torch.float16)
                    O_shared = torch.zeros((H_per_block, D), dtype=torch.float16)
                    Lse_shared = torch.zeros((H_per_block,), dtype=torch.float)
                    mask = torch.zeros((BI), dtype=torch.bool)

                    acc_o = torch.zeros((H_per_block, D), dtype=torch.float)
                    acc_s = torch.zeros((H_per_block, BI), dtype=torch.float)
                    S_shared = torch.zeros((H_per_block, BI), dtype=torch.float16)
                    sumexp = torch.zeros((H_per_block), dtype=torch.float)
                    sumexp_i = torch.zeros((H_per_block), dtype=torch.float)
                    alpha = torch.zeros((H_per_block), dtype=torch.float)
                    m_i = torch.zeros((H_per_block), dtype=torch.float)
                    m_i_prev = torch.zeros((H_per_block), dtype=torch.float)


                    sumexp.fill_(0)
                    acc_o.fill_(0)
                    m_i.fill_(-(2**30)) # avoid -inf - inf to cause nan

                    b_i, g_i = by, bz
                    #print(REPLICATE_H)
                    #print(bx)
                    s_i = bx if REPLICATE_H == 1 else (bx // REPLICATE_H)
                    q_i = s_i
                    max_kv_i = q_i
                    #print(q_i)

                    H0 = g_i * padded_H + (0 if REPLICATE_H == 1 else (bx % REPLICATE_H) * 64)
                    H1 = H0 + H_per_block

                    Q_shared = Q[b_i, s_i, H0:H1, :D]
                    Q_tail_shared = Q[b_i, s_i, H0:H1, D:]
                    # print(Q_shared)
                    # print(Q_shared.shape)
                    # print(Q)
                    # print(Q.shape)
                    #print(Output.shape)
                    print("NI:",NI)
                    for i_i in range(1):
                        for bi_i in range(BI):
                            mask[bi_i] = Indices[b_i, s_i, g_i, i_i * BI + bi_i] <= max_kv_i
                           # print(mask[bi_i])
                        #print(Indices)

                        for bi_i in range(BI):
                            for d_i in range(D):
                                KV_shared[bi_i, d_i] = KV[b_i, Indices[b_i, s_i, g_i, i_i * BI + bi_i], g_i, d_i]

                        for bi_i in range(BI):
                            for d_i in range(D_tail):
                                K_tail_shared[bi_i, d_i] = KV[b_i, Indices[b_i, s_i, g_i, i_i * BI + bi_i], g_i, D + d_i]

                        for h_i in range(H_per_block):
                            for bi_i in range(BI):
                                if mask[bi_i]:
                                    acc_s[h_i, bi_i] = 0
                                else:
                                    acc_s[h_i, bi_i] = get_dtype_min(acc_s.dtype)

                        #print(acc_s)
                        acc_s = torch.matmul(Q_shared, KV_shared.T)
                        #acc_s += torch.matmul(Q_shared, KV_shared.T)
                        #print(acc_s)
                        acc_s += torch.matmul(Q_tail_shared, K_tail_shared)
                        m_i_prev = m_i
                        #print(acc_s)
                        m_i = torch.max(acc_s, dim=1).values
                        for h_i in range(H_per_block):
                            alpha[h_i] = torch.exp2((m_i_prev[h_i] - m_i[h_i]) * sm_scale)
                        #print(alpha)
                        for h_i in range(H_per_block):
                            for bi_i in range(BI):
                                acc_s[h_i, bi_i] = torch.exp2(acc_s[h_i, bi_i] * sm_scale - m_i[h_i] * sm_scale)
                        sumexp_i = torch.sum(acc_s, dim=1)
                        for h_i in range(H_per_block):
                            sumexp[h_i] = sumexp[h_i] * alpha[h_i] + sumexp_i[h_i]
                        for h_i in range(H_per_block):
                            for d_i in range(D):
                                acc_o[h_i, d_i] = acc_o[h_i, d_i] * alpha[h_i]
                        #print(acc_o)
                        S_shared = acc_s
                        S_shared = S_shared.half()
                        #print(S_shared)
                        acc_o += torch.matmul(S_shared, KV_shared)
                        acc_o = acc_o.float()

                    for h_i in range(H_per_block):
                        for d_i in range(D):
                            acc_o[h_i, d_i] /= sumexp[h_i]
                    for h_i in range(H_per_block):
                        sumexp[h_i] = torch.log2(sumexp[h_i]) + m_i[h_i] * sm_scale
                    Output[b_i, s_i, H0:H1, :] = acc_o
                    Lse[b_i, s_i, H0:H1] = sumexp

    return main


def sparse_attention_fwd_interface(
    q, kv, indices, sm_scale=None, return_p_sum: bool = False, d_v=512
):

    is_casual = True
    assert return_p_sum == False, "This kernel file is for fwd only"
    assert q.is_contiguous() and kv.is_contiguous() and indices.is_contiguous()
    batch, seq_len, heads, dim_plus_tail_dim = q.shape
    _, seq_len_kv, kv_group, _ = kv.shape

    assert dim_plus_tail_dim == 576, "you should assign dim otherwise"
    dim = d_v

    assert kv.shape[-1] == dim_plus_tail_dim
    tail_dim = dim_plus_tail_dim - dim
    assert kv.shape[0] == batch
    _, _, _, topk = indices.shape
    assert indices.shape == (batch, seq_len, kv_group, topk)

    # print("seq_len_kv:", seq_len_kv)
    # print("kv_group:", kv_group)
    # print("seq_len:", seq_len)
    # print("dim_plus_tail_dim:", dim_plus_tail_dim)

    Output = torch.full((batch, 10, heads, dim), -1, dtype=torch.float16)
    # lse_shape = [batch, seq_len, heads]
    # Output = torch.tensor(o_shape)
    Lse = torch.full((batch, 10, heads), -1, dtype=torch.float16)
    print("Output.shape", Output.shape)
    print("Lse.shape", Lse.shape)

    kernel = sparse_attention_fwd(
        heads, dim, tail_dim, topk, kv_group, sm_scale, is_casual
    )

    kernel(q, kv, indices, Output, Lse)
    print(Output.shape)
    print(Output)

    return Output, Lse


def ref_sparse_attention_fwd_interface(q, kv, indices, sm_scale=None, is_casual=True):
    q = q.float()
    kv = kv.float()
    indices = indices.transpose(1, 2)
    b, sq, h, dim_q = q.shape
    b, sk, g, _ = kv.shape

    assert kv.shape[-1] == 576, "you should assign dim otherwise"
    dim = 512
    k = kv
    v = kv[..., :dim]

    b, _, _, dim_v = v.shape
    num_kv_per_index = 1
    g_index = g
    h_index = h // g
    compressed_casual_mask = torch.arange(0, sq, dtype=torch.int32).view(
        -1, 1
    ) >= torch.arange(1 - 1, sk * 1, 1, dtype=torch.int32).view(1, -1)

    mask = q.new_zeros(b, g_index, sq, sk + 1, dtype=torch.bool).scatter(
        3, indices.long(), 1
    )
    mask = mask[..., :-1]
    mask = mask & compressed_casual_mask.view(1, 1, sq, sk)
    mask[:, :, : 1 - 1, 0] = True
    mask = mask.view(b, g_index, 1, sq, sk)

    q = q.view(b, sq, g, -1, dim_q)
    score = torch.einsum("bmghd,bngd->bghmn", q, k)
    sm_scale = dim_q**-0.5 if sm_scale is None else sm_scale
    score = score.masked_fill(~mask, float("-inf")).mul(sm_scale)
    p = score.softmax(dim=-1)
    p = p.view(b, g_index, h_index, -1, sq, sk)
    p = p.view(b, g, -1, sq, sk)
    o = torch.einsum("bghmn,bngd->bmghd", p.type(v.dtype), v)
    o = o.reshape(b, sq, h, dim_v)
    return o.to(torch.float16)


def print_red_warning(message):
    print(f"\033[31mWARNING: {message}\033[0m")


def calc_sim(x, y, name="tensor"):
    x, y = x.data.double(), y.data.double()
    denominator = (x * x + y * y).sum()
    if denominator == 0:
        print_red_warning(f"{name} all zero")
        return 1
    sim = 2 * (x * y).sum() / denominator
    return sim


def assert_similar(x, y, eps=1e-8, name="tensor"):
    sim = calc_sim(x, y, name)
    diff = 1.0 - sim
    if not (0 <= diff <= eps):
        print_red_warning(f"{name} Error: {diff}")
        assert False

    import torch
    import numpy as np
    import random

def test_sparse_attn_mla_fwd():
    B, S, SKV, H, HKV, DQK, DV, topk, dtype = (
        1, # B
        4096, # S
        32768, # SKV
        128, # H
        1, # HKV
        576, # DQK
        512, # DV
        2048, # topk
        torch.float16,
    )

    # def set_random_seed(seed):
    #     # 设置 Python 内置随机种子
    #     random.seed(seed)
    #     # 设置 NumPy 随机种子
    #     np.random.seed(seed)
    #     # 设置 PyTorch CPU 随机种子
    #     torch.manual_seed(seed)
    #     # 设置 PyTorch GPU 随机种子（单卡）
    #     if torch.cuda.is_available():
    #         torch.cuda.manual_seed(seed)
    #         # 设置 PyTorch 所有 GPU 随机种子（多卡）
    #         torch.cuda.manual_seed_all(seed)
    #     # 禁用 cuDNN 的随机性（保证卷积操作等的确定性）
    #     torch.backends.cudnn.deterministic = True
    #     # 禁用 cuDNN 的自动优化（避免因算法选择不同导致的随机性）
    #     torch.backends.cudnn.benchmark = False
    #
    # # 调用函数设置种子（例如种子值为 42）
    # set_random_seed(42)

    torch.random.manual_seed(0)
    q = torch.randn((B, S, H, DQK), dtype=dtype).requires_grad_(True)
    kv = torch.randn((B, SKV, HKV, DQK), dtype=dtype).requires_grad_(True)
    print(q)



    indices = torch.full((B, S, HKV, topk), SKV - 1, dtype=torch.int32)
    for b in range(B):
        for t in range(S):
            for h in range(HKV):
                i_i = torch.randperm(max(1, t))[:topk]
                # print(i_i)
                indices[b, t, h, : len(i_i)] = i_i
                # print(indices[b, t, h, : ])

    # for b in range(B):
    #     for t in range(S):
    #         for h in range(HKV):
    #             for i in range(topk):
    #                 v = indices[b, t, h, i]
    #                 if v >= 32768:
    #                     indices[b, t, h, i] = 32767


    tl_out, tl_lse = sparse_attention_fwd_interface(q, kv, indices)

    def fn():
        return f(q, kv, indices)


if __name__ == "__main__":
    #print(torch.tensor([2,3,4,5], dtype=torch.float32))
    test_sparse_attn_mla_fwd()