# CANN SFA Homework

## 文件结构

```bash
.
├── cann_sfa_common.py
├── intrinsic_speed.py
├── phase_diagram.py
└── outputs     # 实验结果输出目录
    ├── intrinsic_speed
    │   ├── intrinsic_speed.png
    │   └── intrinsic_speed.npz
    └── phase_diagram
        ├── phase_diagram.png
        └── state_map.npz
```

- `cann_sfa_common.py`
  公共模型与工具函数。通常不需要修改。

- `intrinsic_speed.py`
  实验 1。绘制 intrinsic speed 随 scaled SFA strength 变化的曲线。

- `phase_diagram.py`
  实验 2。绘制 `smooth / oscillatory / traveling` 三种状态的相图。

## 关于缓存

每个脚本最上方都有一个 `Config` 数据类，用来集中管理实验参数。其中 `recompute` 参数会控制是否使用缓存：

- `True`：忽略现有缓存，重新扫描。
- `False`：如果已有缓存，则直接读取缓存。

两个脚本都支持缓存。其目的在于：

- 避免每次运行都重复扫描
- 当参数扫描较慢时，可以先保存已有结果

建议：

- 第一次调试代码时，使用 `recompute = True`
- 代码正确、结果稳定后，可以改成 `recompute = False`，直接复用缓存，便于调试绘图风格
- 如果修改了 `scaled_m_start / scaled_m_end / n_samples`，且 `recompute = False`，`intrinsic_speed.py` 会尽量复用旧缓存中仍然重合的采样点；如果希望完全重算，请将 `recompute = True` 或手动删除缓存文件
