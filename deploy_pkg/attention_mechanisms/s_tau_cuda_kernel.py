"""
█ s_tau_cuda_kernel.py — CUDA C++ 行内编译 fused s^τ (v5)

vs v4 (autograd.Function):
  - forward: 单次 kernel 完成 clamp+pow+sum+normalize
  - backward: 单次 kernel 完成 score_grad + tau 项归约
  - 不依赖 Triton，所有架构兼容
  - 首次 import 自动编译

用法:
    from s_tau_cuda_kernel import s_tau_norm_cuda
"""
import torch
from torch.utils.cpp_extension import load_inline

CUDA_SRC = r'''
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

// Forward: each block → 1 row [L_kv]
__global__ void fwd_kernel(
    const float* s, float* a, float* cl, const float* tau,
    int Lk, int H, int Lq, float eps
) {
    int row = blockIdx.x, h = (row / Lq) % H, tid = threadIdx.x;
    float th = tau[h];
    __shared__ float sm[256];
    int idx = row * Lk + tid;
    float sv = s[idx];
    float sc = fmaxf(sv, eps);
    float p = expf(th * logf(sc));
    cl[idx] = sc;
    sm[tid] = p;
    for (int s2 = blockDim.x/2; s2 > 0; s2 >>= 1) {
        __syncthreads();
        if (tid < s2) sm[tid] += sm[tid + s2];
    }
    __syncthreads();
    a[idx] = p / (sm[0] + eps);
}

// Backward: score grad + tau terms
__global__ void bwd_kernel(
    const float* go, const float* a, const float* cl, const float* tau,
    float* sg, float* t1b, float* t2b,
    int Lk, int H, int Lq, float eps
) {
    int row = blockIdx.x, h = (row / Lq) % H, tid = threadIdx.x;
    float th = tau[h];
    __shared__ float sm[768]; // 256 * 3
    int idx = row * Lk + tid;
    float gov = go[idx], av = a[idx], sc = cl[idx];
    float inv = 1.0f / sc, ls = logf(sc), ago = av * gov;
    sm[tid] = ago;
    sm[256 + tid] = ago * ls;
    sm[512 + tid] = av * ls;
    for (int s2 = blockDim.x/2; s2 > 0; s2 >>= 1) {
        __syncthreads();
        if (tid < s2) {
            sm[tid] += sm[tid + s2];
            sm[256 + tid] += sm[256 + tid + s2];
            sm[512 + tid] += sm[512 + tid + s2];
        }
    }
    __syncthreads();
    float wsum = sm[0];
    sg[idx] = th * av * (gov - wsum) * inv;
    if (tid == 0) { t1b[row] = sm[256]; t2b[row] = sm[512] * wsum; }
}

void fwd_launch(torch::Tensor scores, torch::Tensor attn,
    torch::Tensor clamped, torch::Tensor tau, float eps) {
    int Lk = scores.size(3), H = scores.size(1), Lq = scores.size(2);
    int rows = scores.numel() / Lk;
    int tpb = (Lk < 256) ? 32 : 256;
    while (tpb > Lk) tpb >>= 1;
    fwd_kernel<<<rows, tpb, tpb*sizeof(float)>>>(
        scores.data_ptr<float>(), attn.data_ptr<float>(),
        clamped.data_ptr<float>(), tau.data_ptr<float>(),
        Lk, H, Lq, eps);
}

void bwd_launch(torch::Tensor go, torch::Tensor attn,
    torch::Tensor clamped, torch::Tensor tau, torch::Tensor sg,
    torch::Tensor t1b, torch::Tensor t2b, float eps) {
    int Lk = attn.size(3), H = attn.size(1), Lq = attn.size(2);
    int rows = attn.numel() / Lk;
    int tpb = (Lk < 256) ? 32 : 256;
    while (tpb > Lk) tpb >>= 1;
    bwd_kernel<<<rows, tpb, 3*tpb*sizeof(float)>>>(
        go.data_ptr<float>(), attn.data_ptr<float>(),
        clamped.data_ptr<float>(), tau.data_ptr<float>(),
        sg.data_ptr<float>(), t1b.data_ptr<float>(),
        t2b.data_ptr<float>(), Lk, H, Lq, eps);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fwd", &fwd_launch, "");
    m.def("bwd", &bwd_launch, "");
}
'''

_cached = None

def _load():
    global _cached
    if _cached is None:
        _cached = load_inline(
            name='s_tau_cuda',
            cpp_sources='void fwd_launch(); void bwd_launch();',
            cuda_sources=CUDA_SRC,
            functions=['fwd', 'bwd'],
            verbose=False,
        )
    return _cached


class _CudaFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, scores, per_head_tau, eps=1e-8):
        out_dtype = scores.dtype
        s32 = scores.to(torch.float32).contiguous()
        t32 = per_head_tau.to(torch.float32).contiguous()
        B, H, Lq, Lk = scores.shape
        attn = torch.empty_like(s32)
        cl = torch.empty_like(s32)
        _load().fwd(s32, attn, cl, t32, float(eps))
        ctx.save_for_backward(attn, cl, t32)
        ctx.eps, ctx.out_dtype = eps, out_dtype
        ctx.B, ctx.H, ctx.Lq, ctx.Lk = B, H, Lq, Lk
        return attn.to(out_dtype)

    @staticmethod
    def backward(ctx, grad_out):
        attn, cl, t32 = ctx.saved_tensors
        eps, B, H, Lq, Lk = ctx.eps, ctx.B, ctx.H, ctx.Lq, ctx.Lk
        go32 = grad_out.to(torch.float32).contiguous()
        sg = torch.empty_like(attn)
        t1 = torch.empty(B*H*Lq, device=attn.device, dtype=torch.float32)
        t2 = torch.empty(B*H*Lq, device=attn.device, dtype=torch.float32)
        _load().bwd(go32, attn, cl, t32, sg, t1, t2, float(eps))
        gt = (t1.view(B, H, Lq).sum(dim=(0, 2)) - t2.view(B, H, Lq).sum(dim=(0, 2)))
        return sg.to(ctx.out_dtype), gt.to(ctx.out_dtype), None


def s_tau_norm_cuda(scores, per_head_tau, eps=1e-8):
    return _CudaFn.apply(scores, per_head_tau, eps)
