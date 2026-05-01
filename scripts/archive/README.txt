归档说明 — 2026-05-01

一次性调试/验证脚本，不参与项目核心流程：
- _check*.py      → attention 算子梯度/结果验证 (6 个)
- _deploy*.py     → 一次性部署脚本
- _launch*.py     → 一次性启动脚本
- _qwen_*.py      → Qwen 早期版本 (v2 保留在上级)
- _tau_generate/verify → 参数生成/验证
- 其余            → 零散调试

保留在上级的重用脚本：
  _qwen_stau_v2.py   ← Qwen3/Qwen3.5 monkey-patch (活跃)
  _gpt2_stau.py      ← GPT-2 monkey-patch
  _ppl_benchmark.py  ← PPL 基准测量
  _qwen_gpu_final.py ← GPU 最终实验
  _multitype_test.py ← 多文本类型测试
  _equiv_experiment.py ← 等价性数值验证
