# Authoritative manuscript figure source data

该目录是论文所有面板级 CSV 的唯一权威位置。这里只保存处理后的作图源
数据、字段字典、数据清单和说明文件；正式 PNG/SVG/PDF/TIFF 统一存放在
`experiments/result_analysis/figures/`。

## 数据包

1. `computational_application_case/`
   - 身份审计、硬约束、Pareto、bootstrap、参考电芯映射等 18 个唯一 CSV。
2. `dataset_statistics/`
   - 数据集统计图 a–h 的面板级 CSV。
3. `interpretability_feature_importance_4x3/`
   - 可解释性、节点、边、官能团及汇总面板 CSV。
4. `performance_results/`
   - 六种性质散点、跨协议指标和汇总表。
5. `molecular_origin_analysis/`
   - 分子结构—宏观性质主图 a–d、SI 热容尺寸控制 CSV 及人工校订字段字典。
6. `Intro-method/`
   - 非定量概念图的面板内容清单。

## 权威清单

- `manifest.csv`：逐文件记录数据包、生产脚本、大小、行列数和 SHA-256。
- `column_dictionary.csv`：汇总所有源表字段的语义、单位/尺度、CSV
  存储类型、定义来源和生产脚本。定义来源分为 `curated-common`、
  `curated-bundle`、`curated-local` 和 `schema-rule`；不允许占位说明。
- `molecular_origin_analysis/column_dictionary.csv`：该分析的详细人工字段定义。

重新生成总清单和字段目录：

```powershell
python experiments\manuscript_figure_source_data\rebuild_manifest.py
```

脚本会拒绝该目录中的图文件和逐字节重复 CSV，防止再次产生双重权威副本。

应用案例生成器只向
`computational_application_case/` 发布 Figure 5、Figure 6 和对应 SI
实际使用的 18 个面板表，并在发布后自动重建总字段字典和清单；不再向
`LaTex-MIPGraph/Fig/source_data/` 创建第二套副本。

重新生成 molecular-origin 源数据：

```powershell
python experiments\molecular_origin_analysis\run_all.py --stage all
python experiments\molecular_origin_analysis\package_source_data.py
python experiments\manuscript_figure_source_data\rebuild_manifest.py
```
