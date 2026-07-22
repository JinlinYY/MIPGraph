# Application chapter result package

本文件夹集中保存 computational application case 的论文交付结果。

- `figures/`：正文 Figure 5、Figure 6以及两张SI图，每张同时提供600 dpi PNG和由原Matplotlib对象直接导出的SVG。
- `csv/`：当前章节实际使用的身份审计、筛选轨迹、硬约束、Pareto排序、Top-8选择、bootstrap、跨协议稳定性、参考电芯映射和资格角色记录。
- `manifest.csv`：每个文件的原始位置、字节数、CSV行列数和SHA-256。

本结果包不包含旧的 agent-assisted [BMIM] 案例文件，也不包含模型训练和前文性能验证结果。SVG不是由PNG描摹或转换得到，而是与PNG在同一次绘图过程中直接导出。

重新生成：

```powershell
E:\anaconda\envs\ggnn39\python.exe `
  experiments\computational_application_case\scripts\build_refactored_application_case.py
E:\anaconda\envs\ggnn39\python.exe `
  experiments\computational_application_case\scripts\package_chapter_artifacts.py
```
