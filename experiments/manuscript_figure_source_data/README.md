# Manuscript figure source data

该目录集中保存论文主图及其可审计 CSV 源数据。各子目录彼此独立，
并提供图文件、面板级 CSV、文件映射及 SHA-256 完整性记录。

## Bundles

1. `dataset_statistics/`
   - 对应数据集统计图；
   - 包含各面板的 CSV 源数据和当前图文件审计记录。
2. `interpretability_feature_importance_4x3/`
   - 对应模型可解释性与特征重要性图；
   - 包含节点、边、官能团及汇总面板源数据。
3. `performance_results/`
   - 对应模型性能结果图；
   - 包含各数据划分协议的预测散点与汇总指标。
4. `molecular_origin_analysis/`
   - 对应分子结构—宏观性质关系主图及 SI 热容尺寸控制图；
   - 包含主图 a–d 的独立 CSV、SI CSV、列字典和 SHA-256 清单。

`manifest.csv` 是前三个既有数据包的总清单；新增的
`molecular_origin_analysis/manifest.csv` 是该分析模块的自包含清单。

替换前三个数据包中的图件或面板 CSV 后，重新生成校验清单：

```powershell
python experiments\manuscript_figure_source_data\rebuild_manifest.py
```

重新生成 molecular-origin 数据包：

```powershell
python experiments\molecular_origin_analysis\run_all.py --stage all
python experiments\molecular_origin_analysis\package_source_data.py
```
