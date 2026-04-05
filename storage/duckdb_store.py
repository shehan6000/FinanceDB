"""
FinanceDB - DuckDB Storage Layer (Fixed for Upserts)
Columnar storage for compressed embeddings using DuckDB.

Author: FinanceDB Project
Module: 3 - Storage Layer (DuckDB)
"""

import duckdb
import json
import pickle
from pathlib import Path
from typing import List, Dict, Optional, Any, Tuple
from datetime import datetime
import numpy as np
from loguru import logger


class DuckDBStore:
    """
    DuckDB-based columnar storage for financial time-series embeddings.
    
    Fixed: Uses 'INSERT OR REPLACE' to handle duplicate window_ids gracefully.
    """
    
    def __init__(self, db_path: str = "storage/financedb.duckdb", enable_logging: bool = True):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize connection
        self.conn = duckdb.connect(str(self.db_path))
        
        # Setup logging
        self.log_handler_id = None
        if enable_logging:
            self.log_handler_id = logger.add(
                self.db_path.parent / "duckdb_store.log",
                rotation="10 MB",
                level="INFO"
            )
        
        self._create_schema()
        logger.info(f"Initialized DuckDBStore at {self.db_path}")
    
    def _create_schema(self):
        """Create database schema with tables and indexes."""
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS timeseries (
                window_id VARCHAR PRIMARY KEY,
                timestamp TIMESTAMP NOT NULL,
                symbol VARCHAR NOT NULL,
                embedding BLOB NOT NULL,
                metadata JSON,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        try:
            self.conn.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON timeseries(timestamp)")
            self.conn.execute("CREATE INDEX IF NOT EXISTS idx_symbol ON timeseries(symbol)")
            self.conn.execute("CREATE INDEX IF NOT EXISTS idx_symbol_timestamp ON timeseries(symbol, timestamp)")
            logger.info("Schema and indexes verified")
        except Exception as e:
            logger.warning(f"Index creation notice: {e}")
    
    def insert(
        self,
        window_id: str,
        timestamp: datetime,
        symbol: str,
        embedding: np.ndarray,
        metadata: Optional[Dict] = None
    ) -> bool:
        """Insert or update a single embedding."""
        try:
            embedding_bytes = pickle.dumps(embedding, protocol=pickle.HIGHEST_PROTOCOL)
            metadata_json = json.dumps(metadata) if metadata else None
            
            # FIXED: Added 'OR REPLACE' to handle PRIMARY KEY conflicts
            self.conn.execute("""
                INSERT OR REPLACE INTO timeseries (window_id, timestamp, symbol, embedding, metadata)
                VALUES (?, ?, ?, ?, ?)
            """, [window_id, timestamp, symbol, embedding_bytes, metadata_json])
            
            return True
        except Exception as e:
            logger.error(f"Failed to insert window {window_id}: {e}")
            return False
    
    def insert_batch(
        self,
        records: List[Tuple[str, datetime, str, np.ndarray, Optional[Dict]]]
    ) -> int:
        """Batch insert or update multiple embeddings."""
        if not records:
            return 0
        
        try:
            batch_data = []
            for window_id, timestamp, symbol, embedding, metadata in records:
                embedding_bytes = pickle.dumps(embedding, protocol=pickle.HIGHEST_PROTOCOL)
                metadata_json = json.dumps(metadata) if metadata else None
                batch_data.append([window_id, timestamp, symbol, embedding_bytes, metadata_json])
            
            # FIXED: Added 'OR REPLACE' so one duplicate doesn't fail the whole batch
            self.conn.executemany("""
                INSERT OR REPLACE INTO timeseries (window_id, timestamp, symbol, embedding, metadata)
                VALUES (?, ?, ?, ?, ?)
            """, batch_data)
            
            logger.info(f"Batch processed {len(records)} records (Upsert mode)")
            return len(records)
        except Exception as e:
            logger.error(f"Batch insert failed: {e}")
            return 0

    # --- Query Methods (Unchanged but included for completeness) ---

    def query(self, sql_string: str) -> List[Dict]:
        try:
            result = self.conn.execute(sql_string).fetchall()
            columns = [desc[0] for desc in self.conn.description]
            return [dict(zip(columns, row)) for row in result]
        except Exception as e:
            logger.error(f"Query failed: {e}")
            return []
    
    def get_by_id(self, window_id: str) -> Optional[Dict]:
        try:
            result = self.conn.execute("""
                SELECT window_id, timestamp, symbol, embedding, metadata
                FROM timeseries WHERE window_id = ?
            """, [window_id]).fetchone()
            
            if result:
                window_id, timestamp, symbol, embedding_bytes, metadata_json = result
                return {
                    'window_id': window_id,
                    'timestamp': timestamp,
                    'symbol': symbol,
                    'embedding': pickle.loads(embedding_bytes),
                    'metadata': json.loads(metadata_json) if metadata_json else None
                }
            return None
        except Exception as e:
            logger.error(f"Failed to get window {window_id}: {e}")
            return None

    def count(self) -> int:
        return self.conn.execute("SELECT COUNT(*) FROM timeseries").fetchone()[0]

    def close(self):
        if self.conn:
            self.conn.close()
            if self.log_handler_id:
                logger.remove(self.log_handler_id)

    def __enter__(self): return self
    def __exit__(self, exc_type, exc_val, exc_tb): self.close()