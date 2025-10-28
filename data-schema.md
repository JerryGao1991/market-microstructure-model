

# 1) `orderbook_snapshots`（L2 订单簿快照）

**用途**：特征工程（OFI、Microprice、Imbalance 等）、深度模型输入、基础回放。
**文件示例**：`data/raw/orderbook_snapshots/date=2025-09-15/instrument_id=AAPL.P.XNAS.csv`
**主键（逻辑）**：`instrument_id, ts_event`（同一时间多条用 `book_seq` 去重）
**最小档位 N**：建议 10 档（支持 5/10/20 档配置）

| 列名                | 类型                        | 必填 | 描述与单位                                       | 取值/规则                       |
| ----------------- | ------------------------- | -- | ------------------------------------------- | --------------------------- |
| instrument_id     | STRING                    | 是  | 内部统一标识，和 `instrument_info.instrument_id` 一致 | 例：`AAPL.P.XNAS`             |
| venue             | STRING                    | 否  | 交易场所代码                                      | 例：`XNAS`、`ARCA`             |
| ts_event          | TIMESTAMP (UTC, ns/us/ms) | 是  | 快照时间（行情时间）                                  | 单调非降                        |
| book_seq          | BIGINT                    | 否  | 行情/快照序号                                     | 并列时间点的排序键                   |
| bid_px_1…bid_px_N | DOUBLE                    | 是  | 买一至买N价格                                     | 非增序：bid_px_k ≥ bid_px_{k+1} |
| bid_sz_1…bid_sz_N | DOUBLE/INT                | 是  | 买一至买N数量（股/张）                                | 非负，建议 ≥0                    |
| ask_px_1…ask_px_N | DOUBLE                    | 是  | 卖一至卖N价格                                     | 非减序：ask_px_k ≤ ask_px_{k+1} |
| ask_sz_1…ask_sz_N | DOUBLE/INT                | 是  | 卖一至卖N数量（股/张）                                | 非负，建议 ≥0                    |
| quote_condition   | STRING                    | 否  | 报价状态/标志                                     | 例：`OPEN`,`HALT`,`LULD`      |
| mid_px            | DOUBLE                    | 否  | (可冗余) 中间价                                   | = (bid_px_1 + ask_px_1)/2   |
| spread_px         | DOUBLE                    | 否  | (可冗余) 价差                                    | = ask_px_1 − bid_px_1       |
| book_depth        | INT                       | 否  | (可冗余) 档位数                                   | 与 N 一致                      |

**校验**：`ask_px_1 ≥ bid_px_1`；价格/数量能被 `tick_size/lot_size` 整除（来自 `instrument_info`）。

---

# 2) `trades`（逐笔成交）

**用途**：TCA 分解、冲击估计、trade sign/主动方识别、现金流对账。
**文件示例**：`data/raw/trades/date=2025-09-15/instrument_id=AAPL.P.XNAS.csv`
**主键（逻辑）**：`instrument_id, ts_event, trade_id`（若无 trade_id，可用 `event_seq`）

| 列名              | 类型                        | 必填 | 描述与单位      | 取值/规则                |
| --------------- | ------------------------- | -- | ---------- | -------------------- |
| instrument_id   | STRING                    | 是  | 内部统一标识     | 对齐 `instrument_info` |
| venue           | STRING                    | 否  | 交易场所代码     | 例：`XNAS`             |
| ts_event        | TIMESTAMP (UTC, ns/us/ms) | 是  | 成交时间（行情时间） | 单调非降                 |
| trade_id        | STRING/BIGINT             | 否  | 成交唯一标识     | 无则留空并用 `event_seq`   |
| event_seq       | BIGINT                    | 否  | 同时刻序号      | 消歧用                  |
| price           | DOUBLE                    | 是  | 成交价        | ≥0                   |
| quantity        | DOUBLE/INT                | 是  | 成交量（股/张）   | ≥0                   |
| aggressor_side  | STRING                    | 否  | 主动方方向      | `B`/`S`/`U`(未知)      |
| trade_condition | STRING                    | 否  | 成交条件码      | 供应商定义                |
| buyer_id        | STRING                    | 否  | 买方经纪/参与者   | 可空                   |
| seller_id       | STRING                    | 否  | 卖方经纪/参与者   | 可空                   |

**校验**：`price` 与 `quantity` 能被 `tick_size/lot_size` 整除；时间与 `orderbook_snapshots` 同时区。

---

# 3) `instrument_info`（标的信息）

**用途**：统一 ID、最小变动单位、最小下单单位、主场所、货币等参数；其他表据此校验。
**文件示例**：`data/raw/meta/instrument_info.csv`
**主键**：`instrument_id`

| 列名                     | 类型         | 必填 | 描述                       | 取值/规则                          |
| ---------------------- | ---------- | -- | ------------------------ | ------------------------------ |
| instrument_id          | STRING     | 是  | 内部统一 ID（建议包含代码.资产类型.主场所） | 例：`AAPL.P.XNAS`                |
| symbol                 | STRING     | 是  | 市场代码                     | 例：`AAPL`                       |
| venue_primary          | STRING     | 是  | 主交易场所                    | 例：`XNAS`                       |
| asset_class            | STRING     | 是  | 资产类型                     | `EQUITY`/`ETF`/`FUTURES`/`FX`… |
| currency               | STRING     | 是  | 报价货币                     | `USD`/`EUR`…                   |
| tick_size              | DOUBLE     | 是  | 最小价格档位                   | >0                             |
| lot_size               | DOUBLE/INT | 是  | 最小交易单位                   | >0                             |
| price_display_decimals | INT        | 否  | 显示精度                     | 例：4                            |
| fee_scheme             | STRING     | 否  | 费率方案摘要                   | 例：`MAKER/TAKER`                |
| isin                   | STRING     | 否  | 国际证券识别码                  | 可空                             |
| sector                 | STRING     | 否  | 行业                       | 可空                             |
| status                 | STRING     | 否  | 状态                       | `ACTIVE`/`HALTED`/`DELISTED`…  |
| effective_from         | DATE       | 否  | 参数生效日                    | 用于历史变更                         |
| effective_to           | DATE       | 否  | 参数失效日                    | 用于历史变更                         |

**校验**：`tick_size>0, lot_size>0`；`instrument_id` 必须唯一；其他原始表的 `instrument_id` 必须在此表中存在。

---

# 4) `trading_calendar`（交易日与会话时间）

**用途**：会话切片、重放边界、Purged/Embargo CV、竞价段识别。
**文件示例**：`data/raw/meta/trading_calendar.csv`
**主键（逻辑）**：`venue, date`

| 列名                | 类型              | 必填 | 描述                          | 取值/规则      |
| ----------------- | --------------- | -- | --------------------------- | ---------- |
| venue             | STRING          | 是  | 交易所/交易平台代码                  | 例：`XNAS`   |
| date              | DATE            | 是  | 交易日（建议以场所本地日历，但时间字段一律存 UTC） | 唯一         |
| session_open_utc  | TIMESTAMP (UTC) | 是  | 当日连续竞价开始时间（UTC）             | 不为空        |
| session_close_utc | TIMESTAMP (UTC) | 是  | 当日连续竞价结束时间（UTC）             | ≥ open     |
| auction_open_utc  | TIMESTAMP (UTC) | 否  | 盘前/集合竞价开始（UTC）              | 可空         |
| auction_close_utc | TIMESTAMP (UTC) | 否  | 盘前/集合竞价结束（UTC）              | 可空         |
| is_trading_day    | BOOLEAN         | 是  | 是否为交易日                      | true/false |
| is_half_day       | BOOLEAN         | 否  | 是否半日市                       | 默认 false   |
| notes             | STRING          | 否  | 熔断、规则变更、特别公告                | 可空         |

**校验**：`session_close_utc ≥ session_open_utc`；若有竞价段，`auction_open_utc ≤ auction_close_utc`。

---

## 目录与分区建议

* `orderbook_snapshots/` 与 `trades/` 采用 **按日期与 instrument 分区**：
  `data/raw/orderbook_snapshots/date=YYYY-MM-DD/instrument_id=.../*.csv`
  `data/raw/trades/date=YYYY-MM-DD/instrument_id=.../*.csv`
* `instrument_info.csv` 与 `trading_calendar.csv` 放在 `data/raw/meta/`。

---

## 时间与单位统一规范

* **时区**：所有 `TIMESTAMP` 字段均为 **UTC**；若源是本地时区，请转换后提供。
* **精度**：优先 `ns`，其次 `us` 或 `ms`；四表保持一致。
* **价格/数量**：价格以原币种；数量以股/张；应与 `tick_size/lot_size` 匹配。

---

## 质量与一致性规则（到货验收要点）

1. **跨表键一致**：`instrument_id` 必须在 `instrument_info` 中存在；`venue` 必须能在 `trading_calendar` 中找到对应日期记录。
2. **序列有效**：`orderbook_snapshots.ts_event` 与 `trades.ts_event` 在同一标的下单调非降；如存在重复时间戳，使用 `book_seq / event_seq` 消歧。
3. **价差与档位**：`ask_px_1 ≥ bid_px_1`；买档非增、卖档非减；数量非负。
4. **刻度校验**：`(price % tick_size) == 0`（或在浮点误差阈内）；`(quantity % lot_size) == 0`（若交易所强约束）。
5. **会话边界**：`trades/quotes` 不应出现在 `session_open_utc` 之前或 `session_close_utc` 之后（若出现需在 `notes` 解释，如盘前/盘后）。

---


