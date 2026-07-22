# Computational application chapter: code map

本目录是论文 computational application case 的唯一代码入口。模型训练、前文性能验证和其他论文图不在本目录中复制，应用案例通过适配器读取项目现有模型实现、固定检查点和数据划分。

## 可复现入口

从项目根目录依次执行：

```powershell
E:\anaconda\envs\ggnn39\python.exe experiments\computational_application_case\run_all.py `
  --config experiments\computational_application_case\configs\auditable_virtual_screening.yaml `
  --force
E:\anaconda\envs\ggnn39\python.exe `
  experiments\computational_application_case\scripts\build_protocol_stability_outputs.py --force
E:\anaconda\envs\ggnn39\python.exe `
  experiments\computational_application_case\scripts\build_refactored_application_case.py
E:\anaconda\envs\ggnn39\python.exe `
  experiments\computational_application_case\scripts\package_chapter_artifacts.py
```

前三步分别生成正式 random-IL 结果、两个独立协议敏感性结果以及 Figure 5、Figure 6 和 SI 图。最后一步只做文件复制、CSV形状统计和SHA-256清单生成，不修改数值。

## 目录职责

| 路径 | 职责 |
|---|---|
| `run_all.py` | 完整应用案例流水线入口 |
| `configs/auditable_virtual_screening.yaml` | 正式 random-IL 单检查点协议 |
| `configs/protocol_stability_balanced.yaml` | balanced-IL 协议敏感性配置 |
| `configs/protocol_stability_ion_family.yaml` | ion-family 压力测试配置 |
| `src/` | 身份审计、适用域、代理量、筛选、Pareto及参考电芯计算 |
| `scripts/build_protocol_stability_outputs.py` | 独立运行两个辅助检查点，不做模型平均 |
| `scripts/build_refactored_application_case.py` | 构建当前章节表格、PNG/PDF/SVG和审计文件 |
| `scripts/package_chapter_artifacts.py` | 仅整理当前应用章节的PNG、SVG和CSV结果 |
| `tests/` | 离线单元测试和章节契约测试 |
| `outputs_primary_audited/` | 正式计算的完整运行输出 |
| `chapter_results/` | 面向论文交付的精简结果包 |
| `CODE_INVENTORY.csv` | 当前代码、配置和说明文件的哈希清单 |

## 结果边界

正式候选只由 random-IL checkpoint 产生。Balanced-IL和ion-family checkpoint只用于跨协议决策稳定性；参考电芯映射只用于Top-8固定后的事后解释。该代码不声称任何候选已通过真实电化学、液态温区或安全验证。
