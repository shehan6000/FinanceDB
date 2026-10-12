# FinanceDB

**A hybrid SQL + semantic query engine for financial time series, built on neural compression.**

FinanceDB turns daily market data into compact learned embeddings and stores them in two complementary engines: **DuckDB** for SQL filtering and aggregation, and **Qdrant** for vector similarity search. A custom query layer (parser, cost-based optimizer, executor) routes each query to the right backend, so you can mix standard SQL with queries like *"find windows that look like March 2020"*.

```sql
SELECT * FROM timeseries WHERE symbol = 'AAPL' AND timestamp BETWEEN '2020-01-01' AND '2021-01-01';
SELECT * FROM timeseries WHERE SIMILAR_TO('2020-03-16', threshold=0.85);
SELECT * FROM timeseries WHERE ANOMALY_SCORE() > 0.95;
```

---

## Key Features

- **Neural compression:** an LSTM autoencoder compresses each 20-day × 20-feature window (1,600 bytes) into a 64-dim embedding, with optional uint8 quantization.
- **Hybrid storage:** DuckDB (metadata, embedding blobs, SQL) plus Qdrant (HNSW cosine similarity), kept consistent by a unified `StorageManager`.
- **Query engine:** regex-based parser with custom operators (`SIMILAR_TO`, `PATTERN_MATCH`, `ANOMALY_SCORE`), a cost-based optimizer, and a step-by-step executor with per-step timing and `explain()`.
- **Semantic retrieval:** similarity search by date or embedding, cross-symbol search, a reusable pattern library, reconstruction-error anomaly detection, and K-Means pattern discovery.
- **Benchmarks:** scripts comparing execution strategies and compression methods (neural vs. zstd vs. gzip vs. raw).

## Architecture

```
 Yahoo Finance
      │
      ▼
┌─────────────────────────────┐
│ Ingestion                   │  data_loader → feature_extractor → windowing
│ 25 stocks, 20 features,     │  (20-day sliding windows, 70/15/15 split)
│ 20-day windows              │
└──────────────┬──────────────┘
               ▼
┌─────────────────────────────┐
│ Neural Compression          │  LSTM encoder → 64-dim embedding → LSTM decoder
│ (+ uint8 quantizer)         │
└──────────────┬──────────────┘
               ▼
┌─────────────────────────────┐
│ StorageManager              │
│  ├─ DuckDB  (SQL, metadata) │
│  └─ Qdrant  (HNSW vectors)  │
└──────────────┬──────────────┘
               ▼
┌─────────────────────────────┐
│ Query Engine                │  Parser → Optimizer → Executor
└──────────────┬──────────────┘
               ▼
┌─────────────────────────────┐
│ Semantic Retrieval          │  Similarity · Patterns · Anomalies · Clustering
└─────────────────────────────┘
```

## Data and Model at a Glance

| Item | Value |
|---|---|
| Universe | AAPL, MSFT, GOOGL, AMZN, TSLA + 20 more S&P 500 stocks |
| Date range | 2020-01-01 to 2024-01-01 |
| Features (20) | Log returns, rolling volatility, volume change, high-low spread, RSI, MACD, Bollinger Bands, 5/10-day momentum, and others (z-score normalized) |
| Window | 20 days × 20 features, stride 1 day |
| Splits | 70% train / 15% val / 15% test |
| Encoder / decoder | 2-layer LSTM, hidden dim 128 |
| Embedding | 64 dimensions (float32 = 256 B, uint8 = 64 B) |
| Training | Adam (lr 1e-3), batch 32, up to 50 epochs, early stopping (patience 5) |

**Compression ratios (by construction):** 1,600 B → 256 B float32 embedding = **6.25×**; 1,600 B → 64 B uint8 embedding = **25×**.

---

## Installation

```bash
git clone https://github.com/<your-username>/financedb.git
cd financedb
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Core dependencies: PyTorch, DuckDB, Qdrant client (local mode, no server needed), yfinance, pandas, numpy, scikit-learn, plotly, tqdm. Streamlit is included for a demo UI.

## Quick Start

Run the pipeline in order. Each stage depends on the previous one.

```bash
# 1. Download data, extract features, build windows
python scripts/01_download_and_process.py \
    --start-date 2020-01-01 --end-date 2024-01-01
#   optional: --symbols AAPL MSFT ... | --force-redownload

# 2. Train and evaluate the compression model
python -m models.train
python -m models.evaluate

# 3. Encode windows and ingest into DuckDB + Qdrant
python scripts/02_ingest_to_storage.py

# 4. Benchmarks
python -m demo.benchmark
python -m demo.compression_benchmark
```

### Querying

```python
from query.query_engine import QueryEngine

engine = QueryEngine()

# Standard SQL
result = engine.execute(
    "SELECT * FROM timeseries WHERE symbol = 'AAPL' "
    "AND timestamp BETWEEN '2020-01-01' AND '2021-01-01'"
)

# Semantic similarity
result = engine.execute("SELECT * FROM timeseries WHERE SIMILAR_TO('2020-03-16', threshold=0.85)")

# Inspect the plan without running it
print(engine.explain("SELECT * FROM timeseries WHERE SIMILAR_TO('2020-03-16', threshold=0.85)"))

# Time a query
print(engine.benchmark("SELECT AVG(volume) FROM timeseries WHERE symbol = 'AAPL'", runs=10))
```

Each `QueryResult` carries the data, `execution_time_ms`, `rows_scanned`, `rows_decompressed`, and the executed plan.

### Semantic retrieval

```python
from retrieval.similarity import SimilarityEngine
from retrieval.patterns import PatternLibrary
from retrieval.anomaly import AnomalyDetector
from retrieval.clustering import PatternDiscovery

sim = SimilarityEngine()
sim.find_similar_to_date("2020-03-16", symbol="AAPL", top_k=10)
sim.find_similar_across_symbols("2020-03-16", symbols=["AAPL", "MSFT", "TSLA"])

patterns = PatternLibrary()
patterns.add_pattern("my_pattern", ["2020-03-12", "2020-03-16"])

detector = AnomalyDetector()
detector.detect_anomalies(threshold=0.95)

PatternDiscovery().discover_patterns(k=10)
```

> Check the imports and signatures against your code; adjust if they differ.

---

## Project Structure

```
financedb/
├── ingestion/
│   ├── data_loader.py        # FinancialDataLoader (yfinance, cached, tqdm)
│   ├── feature_extractor.py  # 20 technical features, z-score normalization
│   └── windowing.py          # WindowGenerator, train/val/test splits
├── models/
│   ├── compression_model.py  # LSTM autoencoder (CompressionModel)
│   ├── train.py              # ModelTrainer, early stopping, loss curves
│   ├── evaluate.py           # MSE, per-feature error, error heatmap
│   └── quantizer.py          # EmbeddingQuantizer (float32 → uint8)
├── storage/
│   ├── duckdb_store.py       # DuckDBStore (SQL, indexes, batch insert)
│   ├── vector_store.py       # QdrantStore (HNSW, cosine)
│   └── storage_manager.py    # StorageManager (unified, transactional writes)
├── query/
│   ├── parser.py             # QueryParser (SQL + custom operators)
│   ├── optimizer.py          # QueryOptimizer (cost model, ExecutionPlan)
│   ├── executor.py           # ExecutionEngine
│   └── query_engine.py       # QueryEngine (main user-facing API)
├── retrieval/
│   ├── similarity.py         # SimilarityEngine
│   ├── patterns.py           # PatternLibrary (configs/patterns.yaml)
│   ├── anomaly.py            # AnomalyDetector (reconstruction error)
│   └── clustering.py         # PatternDiscovery (K-Means + t-SNE)
├── demo/
│   ├── benchmark.py          # query performance benchmark
│   └── compression_benchmark.py
├── scripts/
│   ├── 01_download_and_process.py
│   └── 02_ingest_to_storage.py
├── configs
├── data/{raw,processed}/
```

## How the Query Optimizer Routes Work

| Query shape | Route |
|---|---|
| `SIMILAR_TO(...)` | Qdrant vector search |
| Filters on `timestamp` / `symbol` | DuckDB indexes |
| Aggregations | Approximate on embeddings if possible, otherwise exact on decompressed data |

The cost model is intentionally simple: scan ≈ 0.001 ms/row, decompress ≈ 0.1 ms/row, vector search ≈ log(n) ms. It picks the lowest-cost plan, and `explain()` shows the choice.

## Benchmarks

Both benchmark scripts write reports to `docs/`:

- `docs/benchmark_results.md`: three strategies per query (decompress-all baseline, filter-on-embeddings then decompress matches, approximate aggregation on embeddings), measuring time, rows scanned, rows decompressed, and memory.
- `docs/compression_comparison.md`: neural compression vs. zstd vs. gzip vs. raw (ratio, compress/decompress time, reconstruction error, on-disk size).

Fill in your measured numbers here after running them:

| Metric | Result |
|---|---|
| Reconstruction MSE (test set) | _TBD_ |
| Neural compression vs. raw | _TBD_ |
| Neural compression vs. zstd | _TBD_ |
| Optimized vs. baseline query speedup | _TBD_ |
| Total windows ingested | _TBD_ |

> Report only what the scripts measure, and state the dataset size and hardware.

## Limitations

- **Lossy compression.** Decompressed values are reconstructions, so exact-value queries (e.g. `AVG(volume)`) on embeddings are approximate. Report the reconstruction error alongside any speedup.
- **Small, single-source dataset.** 25 US equities from Yahoo Finance over four years, at daily resolution.
- **Simple parsing.** The parser is regex-based, not a full SQL parser, so unsupported syntax will fail.
- **Simplified cost model.** Costs are fixed constants, not learned or calibrated to your hardware.
- **Pattern examples need in-range dates.** Pattern examples must fall inside the ingested date range (2020 to 2024).
- **No production concerns.** No concurrency control, access control, or incremental re-training.

## Roadmap

- [ ] Streamlit demo UI
- [ ] Incremental ingestion of new data
- [ ] Calibrated cost model from measured timings
- [ ] Evaluate on intraday data and a larger symbol universe
- [ ] Compare against baseline vector search over raw features

