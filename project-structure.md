
## 一、项目总目录结构

```
MarketMicrostructureModeling/
│
├── data/
│   ├── raw/                      # 原始高频订单簿与成交数据（由你提供）
│   │   ├── orderbook_*.csv
│   │   ├── trades_*.csv
│   │   └── meta/
│   │       └── instrument_info.csv
│   ├── interim/                  # 中间清洗与重采样结果
│   ├── features/                 # 特征工程输出（OFI、Imbalance 等）
│   ├── models/                   # 模型权重与参数文件
│   ├── simulation/               # 交易所重放与ABM仿真日志
│   └── reports/                  # 回测、TCA、容量等分析结果
│
├── src/
│   ├── config.py                 # 全局路径、参数、常量定义
│   ├── 0_data_preprocessing.py   # 数据清洗、重采样、结构化
│   ├── 1_feature_engineering.py  # 微结构特征提取（OFI、Microprice、Imbalance等）
│   ├── 2_model_deeplob.py        # DeepLOB/Transformer 深度模型
│   ├── 3_model_baselines.py      # Avellaneda-Stoikov、Almgren-Chriss、Hawkes基线
│   ├── 4_exchange_simulator.py   # 交易所级重放引擎（撮合、队列推进）
│   ├── 5_strategy_engine.py      # 策略逻辑（信号→执行→风控闭环）
│   ├── 6_validation.py           # Purged CV、Embargo、校准、鲁棒性测试
│   ├── 7_evaluation_metrics.py   # P&L、TCA、容量、稳定性等指标计算
│   └── 8_reporting.py            # 生成报告与可视化（Matplotlib/Plotly）
│
├── notebooks/
│   ├── exploratory_analysis.ipynb   # 初步数据分布、特征分析
│   ├── model_comparison.ipynb       # 深度模型与基线对比分析
│   └── tca_analysis.ipynb           # TCA与容量评估图表
│
├── utils/
│   ├── io_utils.py                # 通用文件读写（CSV/Parquet）
│   ├── metrics_utils.py           # 回测与性能指标计算函数
│   ├── feature_utils.py           # OFI、Imbalance计算工具函数
│   ├── plotting_utils.py          # 可视化模板
│   └── simulation_utils.py        # 队列推进与事件生成辅助函数
│
├── tests/
│   ├── test_model_baselines.py
│   ├── test_exchange_simulator.py
│   ├── test_strategy_engine.py
│   └── test_validation.py
│
├── outputs/
│   ├── charts/                    # 各类可视化图表输出
│   ├── logs/                      # 日志与运行记录
│   └── summary_report.md          # 关键实验总结（文本形式）
│
└── README.md
```

---

## 二、文件与模块功能说明

### （1）data 目录

* **raw/**
  存放你提供的原始高频数据，如 L2 订单簿、逐笔成交、报价信息。

  * `orderbook_*.csv`：包含价格档位、买卖盘深度、时间戳。
  * `trades_*.csv`：逐笔成交数据（方向、数量、价格）。
  * `instrument_info.csv`：标的、交易时间、tick size、最小数量单位等。

* **interim/**
  存储经过清洗与重采样后的数据，例如统一时间间隔的L2快照。

* **features/**
  存储提取的特征矩阵，如 OFI、Microprice、Depth/Queue Imbalance、Cancel Rate 等。

* **models/**
  保存训练好的模型权重（DeepLOB、Transformer）及基线参数（Avellaneda、Almgren）。

* **simulation/**
  保存 ABM 仿真与交易所重放日志，包括成交队列、延迟、滑点数据。

* **reports/**
  存储回测与绩效报告（P&L、TCA、容量曲线、风险分析）。

---

### （2）src 目录

| 文件                           | 功能描述                                                       |
| ---------------------------- | ---------------------------------------------------------- |
| **config.py**                | 定义全局路径、常量（采样频率、模型参数、交易成本等）。                                |
| **0_data_preprocessing.py**  | 读取原始CSV，清洗异常值，统一时间步长，生成结构化输入。                              |
| **1_feature_engineering.py** | 计算微结构特征（OFI、Microprice、Queue Imbalance、Cancel Intensity等）。 |
| **2_model_deeplob.py**       | 实现 DeepLOB 或 Transformer 模型，用于价格方向预测。                      |
| **3_model_baselines.py**     | 实现 Avellaneda-Stoikov、Almgren-Chriss、Hawkes 基线模型；提供可解释对照。  |
| **4_exchange_simulator.py**  | 实现价格时间优先撮合、队列推进、集合竞价重放；生成现金流与队列位置日志。                       |
| **5_strategy_engine.py**     | 将预测信号转化为交易动作（下单、撤单、滑移控制、风险约束）；形成信号→执行→风控闭环。                |
| **6_validation.py**          | 执行 Purged/Embargo 时序交叉验证与 Platt/Isotonic 校准；输出稳定性报告。       |
| **7_evaluation_metrics.py**  | 计算净P&L、TCA分解、容量曲线、波动与回撤；生成量化指标表。                           |
| **8_reporting.py**           | 汇总结果生成 PDF/HTML 报告与图表，可输出单策略/多策略对比结果。                      |

---

### （3）notebooks

包含分析与展示用的 Jupyter 笔记本，用于数据探索、模型比较、绩效展示。

* `exploratory_analysis.ipynb`：原始订单簿数据统计与可视化。
* `model_comparison.ipynb`：不同模型信号与P&L对比。
* `tca_analysis.ipynb`：拆单路径、冲击成本、容量曲线展示。

---

### （4）utils

公共函数模块，供主脚本调用。

* `io_utils.py`：统一数据读写接口（CSV/Parquet）。
* `metrics_utils.py`：计算年化收益、夏普比、最大回撤等指标。
* `feature_utils.py`：实现 OFI、Imbalance、Cancel Rate 计算函数。
* `plotting_utils.py`：封装标准可视化模板（K线、冲击曲线、容量曲线等）。
* `simulation_utils.py`：辅助生成订单事件流与仿真日志。

---

### （5）tests

轻量级单元测试，确保每个核心模块（模型、撮合、策略、验证）运行正确。

---

### （6）outputs

存放最终的输出结果，包括可视化图表、日志和实验总结文件。

---

### （7）README.md

简要说明项目运行步骤：

1. 将原始数据放入 `data/raw/`；
2. 依次运行 `src/0_data_preprocessing.py → 1_feature_engineering.py → 2_model_deeplob.py → 5_strategy_engine.py`；
3. 生成回测与报告。

---
