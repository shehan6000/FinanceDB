# FinanceDB Project - Complete Context Prompt


---

## PROJECT OVERVIEW

I'm working on **FinanceDB**, a PhD research project for a compression-aware time-series database system. This is both an academic research project (targeting 5-year PhD completion) and a future commercial product.

### Core Innovation
Traditional time-series databases compress data for storage, then decompress it to answer queries. This creates a bottleneck. Our approach uses **neural embeddings** that serve simultaneously as:
1. Compressed storage (100x smaller)
2. Query execution substrate (10x faster queries)
3. Semantic search index (find similar patterns)

### Primary Domain
Financial time-series data (high-frequency trading, order books, market analytics), where institutions spend $10-20M annually on data storage and struggle with slow analytical queries.

---

## TECHNICAL APPROACH

### Architecture (7 Modules)

**Module 1: Data Ingestion**
- Downloads financial data from Yahoo Finance (demo) or TAQ/LOBSTER (production)
- Preprocesses into fixed-size temporal windows (e.g., 5-minute snapshots)
- Extracts features (price changes, volume, spread, volatility)
- Output: DataWindow objects [60 timesteps × 50 features]

**Module 2: Neural Compression Engine**
- **Encoder**: Hierarchical Transformer architecture
  - Local attention: Within-window patterns (50-second)
  - Global attention: Cross-window patterns (full 5 minutes)
  - Output: 256-D embedding
- **Decoder**: Reconstructs original data from embedding
- **Training**: Multi-task loss = Reconstruction + Rate penalty + Contrastive (semantic)
- **Quantization**: Float32 → Uint8 for 4x additional compression
- **Result**: 50-100x compression with <1% reconstruction error

**Module 3: Storage Layer (Hybrid)**
- **Hot Tier** (6 months): Vector DB (Pinecone/Weaviate) - full embeddings, fast similarity search
- **Warm Tier** (5 years): Columnar storage (Parquet + DuckDB) - quantized embeddings, analytical queries
- **Cold Tier** (5+ years): Archive (S3 Glacier) - heavily compressed, rare access
- StorageManager routes data and queries to appropriate tier

**Module 4: Indexing**
- Temporal index (B-tree on timestamps)
- Vector index (HNSW for similarity search)
- Metadata index (symbols, regimes, features)
- Optional: Learned index (neural network predicts position)

**Module 5: Query Processing**
- **Parser**: Standard SQL + extensions (SIMILAR_TO, PATTERN_MATCH, CLUSTER_ID, ANOMALY_SCORE)
- **Optimizer**: Compression-aware cost model
  - Decides: Decompress now vs. filter compressed first
  - Routes to vector DB or columnar storage
- **Executor**: Custom operators (CompressedScan, SelectiveDecompress, ApproximateAggregate)
- **Result**: 10x speedup by avoiding unnecessary decompression

**Module 6: Semantic Retrieval**
- Similarity search: "Find days like 2008 crash" → cosine similarity in embedding space
- Pattern library: Pre-defined patterns (head-and-shoulders, flash crash, spoofing)
- Clustering: Unsupervised pattern discovery via K-means
- Anomaly detection: High reconstruction error = unusual market behavior

**Module 7: API & User Interface**
- REST API (FastAPI)
- Python client library
- CLI tool
- Streamlit web UI for demos

---

## DEMONSTRATION PROJECT

For the PhD application demo, I'm building a **free, self-contained version** that runs on a laptop:

### Tech Stack (100% Free)
- Python + PyTorch (neural compression)
- DuckDB (embedded analytical database)
- Qdrant (open-source vector DB, local mode)
- Streamlit (interactive web UI)
- Yahoo Finance API (free stock data)

### Demo Dataset
- Symbols: AAPL, MSFT, GOOGL, AMZN, TSLA (S&P 500 stocks)
- Date range: 2020-2024 (4 years)
- Data: Daily OHLCV (Open, High, Low, Close, Volume)
- Windows: 20-day sliding windows with 20 features each

### Demo Flow
1. Download real stock data (5 minutes)
2. Train compression model (30 min CPU, 5 min GPU)
3. Ingest to database (10 minutes)
4. Launch Streamlit demo (interactive)

### Demo Pages
1. **Data Overview**: Show downloaded stocks, date ranges, statistics
2. **Compression Demo**: Display 50x compression ratio, reconstruction quality
3. **Query Performance**: Benchmark 3 approaches, show 10x speedup
4. **Semantic Search**: "Find days similar to March 2020 crash"
5. **Pattern Discovery**: Clustering visualization, anomaly detection

---

## PROJECT STRUCTURE

```
financedb_demo/
├── requirements.txt              # Dependencies
├── README.md                     # Setup instructions
│
├── data/
│   ├── raw/                      # Downloaded stock data
│   ├── processed/                # Cleaned windows
│   └── cache/                    # Intermediate files
│
├── models/
│   ├── compression_model.py      # Neural architecture
│   ├── train.py                  # Training script
│   └── checkpoints/              # Saved models
│
├── storage/
│   ├── duckdb_store.py           # Columnar storage
│   ├── vector_store.py           # Qdrant integration
│   └── financedb.duckdb          # Database file
│
├── ingestion/
│   ├── data_loader.py            # Download from Yahoo Finance
│   ├── preprocessor.py           # Clean and window data
│   └── feature_extractor.py      # Technical indicators
│
├── query/
│   ├── query_engine.py           # SQL + semantic queries
│   ├── optimizer.py              # Compression-aware optimization
│   └── operators.py              # Custom operators
│
├── demo/
│   ├── app.py                    # Streamlit demo app
│   ├── benchmark.py              # Performance comparison
│   └── visualizations.py         # Charts and plots
│
└── scripts/
    ├── 01_download_data.py
    ├── 02_train_model.py
    └── 03_ingest_data.py
```

---

## KEY RESEARCH QUESTIONS

**RQ1: Neural Compression** (ML Focus)
- Can neural models exploit financial domain structure to achieve 100x compression?
- What architecture (Transformer, LSTM, CNN) works best for time-series?
- How to balance compression ratio, reconstruction quality, and semantic preservation?

**RQ2: Compression-Aware Query Processing** (Systems Focus)
- Which SQL operators can execute directly on embeddings without decompression?
- What cost models guide query optimizers when decompression is expensive?
- How to integrate vector databases with traditional storage optimally?

**RQ3: Semantic Information Retrieval** (IR Focus)
- How to extend SQL for content-based similarity queries?
- Can unsupervised clustering discover novel financial patterns?
- What evaluation methodology validates retrieval quality for time-series?



---

## VALIDATION METRICS

### Technical Success Criteria
- ✅ Compression ratio: 50-100x vs. raw data
- ✅ Reconstruction error: <1% on critical features
- ✅ Query speedup: 10x vs. decompress-then-query
- ✅ Similarity search precision@10: >90%
- ✅ Scalability: Sub-second queries on 10TB dataset

### Business Success Criteria
- ✅ Storage cost reduction: >90% ($15M → $1.5M/year)
- ✅ User satisfaction: >4/5 on Likert scale
- ✅ Design partners: 5+ institutions during PhD
- ✅ First revenue: $100K ARR within 1 year post-PhD

---

## IMPORTANT DESIGN DECISIONS

1. **Why hybrid storage (vector DB + columnar)?**
   - Vector DB excellent for similarity but expensive at scale
   - Columnar storage cheap but slower for similarity
   - Hybrid: Hot data in vector DB, historical in columnar

2. **Why not streaming/real-time?**
   - PhD scope: Focus on historical queries first
   - Streaming adds complexity without core innovation
   - Can extend post-PhD

3. **Why financial data?**
   - Clear business value ($10-20M pain point)
   - Rich temporal structure (good for compression)
   - Access to industry validation partners
   - Generalizes to other domains (healthcare, IoT)

4. **Why open-source core?**
   - Academic credibility and reproducibility
   - Community adoption and citations
   - Commercial model: Managed service, not licenses


---
