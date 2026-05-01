归档说明 — 2026-05-01

s_tau_fused_v2_backup.py  → v2 稳定备份（5090 验证过，1.64× 加速）
s_tau_triton.py            → Triton 内核（Blackwell sm_120 不兼容）

当前活跃算子: s_tau_fused.py (v4 autograd.Function, 主用)
待编译:       s_tau_cuda_kernel.py (v5 CUDA C++, 预期 1.12×)
