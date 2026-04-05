"""
FinanceDB - Similarity Engine
Find and analyze similar financial patterns using compressed embeddings.

Author: FinanceDB Project
Module: 5 - Semantic Retrieval (Similarity Search)
"""

import sys
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from loguru import logger

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from storage.storage_manager import StorageManager


class SimilarityEngine:
    """
    Engine for finding similar financial patterns in compressed embeddings.
    
    Features:
    - Find windows similar to a reference date
    - Cross-symbol pattern matching
    - Pairwise date comparison
    - Similarity visualization
    """
    
    def __init__(
        self,
        storage_manager: StorageManager,
        distance_metric: str = "cosine"
    ):
        """
        Initialize similarity engine.
        
        Args:
            storage_manager: StorageManager instance
            distance_metric: Distance metric ('cosine', 'euclidean', 'dot')
        """
        self.storage = storage_manager
        self.distance_metric = distance_metric
        
        logger.info("Initialized SimilarityEngine")
        logger.info(f"  Distance metric: {distance_metric}")
    
    def find_similar_to_date(
        self,
        reference_date: str,
        symbol: str,
        top_k: int = 10,
        min_similarity: float = 0.0,
        exclude_reference: bool = True
    ) -> List[Dict]:
        """
        Find windows similar to a reference date for a given symbol.
        
        Args:
            reference_date: Reference date (YYYY-MM-DD)
            symbol: Stock ticker symbol
            top_k: Number of similar windows to return
            min_similarity: Minimum similarity threshold (0-1)
            exclude_reference: Whether to exclude the reference window itself
            
        Returns:
            List of similar windows with metadata and scores
            
        Example:
            results = engine.find_similar_to_date('2020-03-16', 'AAPL', top_k=10)
        """
        try:
            # Step 1: Get reference window using SQL query
            reference_datetime = pd.to_datetime(reference_date)
            
            # Build SQL query to get reference window
            start_date = reference_datetime.strftime('%Y-%m-%d 00:00:00')
            end_date = (reference_datetime + pd.Timedelta(days=1)).strftime('%Y-%m-%d 00:00:00')
            
            sql = f"""
            SELECT window_id, timestamp, symbol, embedding, metadata
            FROM timeseries 
            WHERE symbol = '{symbol}' 
            AND timestamp >= '{start_date}'
            AND timestamp < '{end_date}'
            LIMIT 1
            """
            
            reference_windows = self.storage.query_sql(sql)
            
            if not reference_windows:
                logger.warning(f"No window found for {symbol} on {reference_date}")
                return []
            
            reference_window = reference_windows[0]
            
            # Deserialize embedding if it's in bytes
            if isinstance(reference_window.get('embedding'), bytes):
                import pickle
                reference_embedding = pickle.loads(reference_window['embedding'])
            else:
                reference_embedding = reference_window['embedding']
            
            logger.info(
                f"Finding similar windows to {symbol} on {reference_date} "
                f"(window_id: {reference_window['window_id']})"
            )
            
            # Step 2: Search for similar embeddings
            results = self.storage.search_similar(
                query_embedding=reference_embedding,
                top_k=top_k + (1 if exclude_reference else 0),  # Get extra if excluding
                symbol_filter=None,  # Search across all symbols
                enrich_metadata=True
            )
            
            # Step 3: Filter and format results
            filtered_results = []
            
            for result in results:
                # Skip reference window if requested
                if exclude_reference and result['window_id'] == reference_window['window_id']:
                    continue
                
                # Skip if below similarity threshold
                if result['score'] < min_similarity:
                    continue
                
                # Format result
                filtered_results.append({
                    'window_id': result['window_id'],
                    'symbol': result['symbol'],
                    'timestamp': result['timestamp'],
                    'similarity_score': result['score'],
                    'metadata': result.get('metadata', {}),
                    'embedding': result.get('embedding')
                })
                
                if len(filtered_results) >= top_k:
                    break
            
            logger.info(f"Found {len(filtered_results)} similar windows")
            
            return filtered_results
            
        except Exception as e:
            logger.error(f"Similarity search failed: {e}")
            raise
    
    def find_similar_to_embedding(
        self,
        embedding: np.ndarray,
        top_k: int = 10,
        min_similarity: float = 0.0,
        symbol_filter: Optional[str] = None
    ) -> List[Dict]:
        """
        Find windows similar to a given embedding vector.
        
        Args:
            embedding: Query embedding vector
            top_k: Number of similar windows to return
            min_similarity: Minimum similarity threshold
            symbol_filter: Optional symbol to filter results
            
        Returns:
            List of similar windows with scores
        """
        try:
            logger.info(f"Searching for similar embeddings (top_k={top_k})")
            
            # Search in vector store
            results = self.storage.search_similar(
                query_embedding=embedding,
                top_k=top_k,
                symbol_filter=symbol_filter,
                enrich_metadata=True
            )
            
            # Filter by similarity threshold
            filtered_results = [
                {
                    'window_id': r['window_id'],
                    'symbol': r['symbol'],
                    'timestamp': r['timestamp'],
                    'similarity_score': r['score'],
                    'metadata': r.get('metadata', {}),
                    'embedding': r.get('embedding')
                }
                for r in results
                if r['score'] >= min_similarity
            ]
            
            logger.info(f"Found {len(filtered_results)} similar windows")
            
            return filtered_results
            
        except Exception as e:
            logger.error(f"Embedding search failed: {e}")
            raise
    
    def find_similar_across_symbols(
        self,
        reference_date: str,
        symbols: List[str],
        top_k: int = 10,
        min_similarity: float = 0.7
    ) -> Dict[str, List[Dict]]:
        """
        Find similar patterns across multiple symbols for a reference date.
        
        Args:
            reference_date: Reference date (YYYY-MM-DD)
            symbols: List of symbols to search
            top_k: Number of results per symbol
            min_similarity: Minimum similarity threshold
            
        Returns:
            Dictionary mapping symbol -> list of similar windows
            
        Example:
            results = engine.find_similar_across_symbols(
                '2020-03-16',
                ['AAPL', 'MSFT', 'GOOGL'],
                top_k=5
            )
        """
        try:
            results_by_symbol = {}
            
            for symbol in symbols:
                logger.info(f"Searching {symbol} for patterns like {reference_date}")
                
                similar_windows = self.find_similar_to_date(
                    reference_date=reference_date,
                    symbol=symbol,
                    top_k=top_k,
                    min_similarity=min_similarity,
                    exclude_reference=True
                )
                
                results_by_symbol[symbol] = similar_windows
                
                logger.info(f"  Found {len(similar_windows)} similar windows in {symbol}")
            
            return results_by_symbol
            
        except Exception as e:
            logger.error(f"Cross-symbol search failed: {e}")
            raise
    
    def compare_dates(
        self,
        date1: str,
        symbol1: str,
        date2: str,
        symbol2: str
    ) -> Dict[str, Any]:
        """
        Compute similarity score between two specific dates.
        
        Args:
            date1: First date (YYYY-MM-DD)
            symbol1: First symbol
            date2: Second date (YYYY-MM-DD)
            symbol2: Second symbol
            
        Returns:
            Dictionary with similarity metrics
            
        Example:
            comparison = engine.compare_dates(
                '2020-03-16', 'AAPL',
                '2008-09-15', 'MSFT'
            )
        """
        try:
            # Build SQL queries for both dates
            date1_dt = pd.to_datetime(date1)
            date2_dt = pd.to_datetime(date2)
            
            sql1 = f"""
            SELECT window_id, timestamp, symbol, embedding, metadata
            FROM timeseries 
            WHERE symbol = '{symbol1}' 
            AND timestamp >= '{date1_dt.strftime('%Y-%m-%d 00:00:00')}'
            AND timestamp < '{(date1_dt + pd.Timedelta(days=1)).strftime('%Y-%m-%d 00:00:00')}'
            LIMIT 1
            """
            
            sql2 = f"""
            SELECT window_id, timestamp, symbol, embedding, metadata
            FROM timeseries 
            WHERE symbol = '{symbol2}' 
            AND timestamp >= '{date2_dt.strftime('%Y-%m-%d 00:00:00')}'
            AND timestamp < '{(date2_dt + pd.Timedelta(days=1)).strftime('%Y-%m-%d 00:00:00')}'
            LIMIT 1
            """
            
            # Get windows using SQL
            windows1 = self.storage.query_sql(sql1)
            windows2 = self.storage.query_sql(sql2)
            
            if not windows1 or not windows2:
                logger.warning(f"One or both windows not found")
                return {'similarity_score': None, 'error': 'Windows not found'}
            
            # Deserialize embeddings
            import pickle
            emb1_bytes = windows1[0].get('embedding')
            emb2_bytes = windows2[0].get('embedding')
            
            emb1 = pickle.loads(emb1_bytes) if isinstance(emb1_bytes, bytes) else emb1_bytes
            emb2 = pickle.loads(emb2_bytes) if isinstance(emb2_bytes, bytes) else emb2_bytes
            
            # Compute similarity
            similarity = self._compute_similarity(emb1, emb2)
            
            result = {
                'date1': date1,
                'symbol1': symbol1,
                'window_id1': windows1[0]['window_id'],
                'date2': date2,
                'symbol2': symbol2,
                'window_id2': windows2[0]['window_id'],
                'similarity_score': float(similarity),
                'distance_metric': self.distance_metric
            }
            
            logger.info(
                f"Similarity between {symbol1}@{date1} and {symbol2}@{date2}: "
                f"{similarity:.4f}"
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Date comparison failed: {e}")
            raise
    
    def _compute_similarity(
        self,
        emb1: np.ndarray,
        emb2: np.ndarray
    ) -> float:
        """
        Compute similarity between two embeddings.
        
        Args:
            emb1: First embedding
            emb2: Second embedding
            
        Returns:
            Similarity score (0-1 for cosine, unbounded for others)
        """
        if self.distance_metric == "cosine":
            # Cosine similarity: dot(a, b) / (norm(a) * norm(b))
            dot_product = np.dot(emb1, emb2)
            norm1 = np.linalg.norm(emb1)
            norm2 = np.linalg.norm(emb2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            return dot_product / (norm1 * norm2)
        
        elif self.distance_metric == "euclidean":
            # Euclidean distance (lower is more similar)
            distance = np.linalg.norm(emb1 - emb2)
            # Convert to similarity (inverse)
            return 1.0 / (1.0 + distance)
        
        elif self.distance_metric == "dot":
            # Dot product (unnormalized)
            return np.dot(emb1, emb2)
        
        else:
            raise ValueError(f"Unknown distance metric: {self.distance_metric}")
    
    # ========================================================================
    # VISUALIZATION METHODS
    # ========================================================================
    
    def plot_similarity_matrix(
        self,
        dates: List[str],
        symbol: str,
        figsize: Tuple[int, int] = (10, 8),
        save_path: Optional[str] = None
    ):
        """
        Plot heatmap of pairwise similarities between dates.
        
        Args:
            dates: List of dates (YYYY-MM-DD)
            symbol: Stock ticker symbol
            figsize: Figure size
            save_path: Optional path to save figure
        """
        try:
            logger.info(f"Generating similarity matrix for {len(dates)} dates")
            
            # Get embeddings for all dates
            embeddings = []
            valid_dates = []
            
            for date in dates:
                date_dt = pd.to_datetime(date)
                sql = f"""
                SELECT window_id, timestamp, symbol, embedding, metadata
                FROM timeseries 
                WHERE symbol = '{symbol}' 
                AND timestamp >= '{date_dt.strftime('%Y-%m-%d 00:00:00')}'
                AND timestamp < '{(date_dt + pd.Timedelta(days=1)).strftime('%Y-%m-%d 00:00:00')}'
                LIMIT 1
                """
                
                windows = self.storage.query_sql(sql)
                
                if windows:
                    import pickle
                    emb_bytes = windows[0].get('embedding')
                    emb = pickle.loads(emb_bytes) if isinstance(emb_bytes, bytes) else emb_bytes
                    embeddings.append(emb)
                    valid_dates.append(date)
                else:
                    logger.warning(f"No window for {symbol} on {date}")
            
            if len(embeddings) < 2:
                logger.warning("Not enough valid dates for similarity matrix")
                return
            
            # Compute pairwise similarities
            n = len(embeddings)
            similarity_matrix = np.zeros((n, n))
            
            for i in range(n):
                for j in range(n):
                    similarity_matrix[i, j] = self._compute_similarity(
                        embeddings[i],
                        embeddings[j]
                    )
            
            # Plot heatmap
            fig, ax = plt.subplots(figsize=figsize)
            
            sns.heatmap(
                similarity_matrix,
                annot=True,
                fmt='.3f',
                cmap='RdYlGn',
                xticklabels=valid_dates,
                yticklabels=valid_dates,
                vmin=0,
                vmax=1,
                ax=ax,
                cbar_kws={'label': 'Similarity Score'}
            )
            
            ax.set_title(f'Pairwise Similarity Matrix - {symbol}', fontsize=14, fontweight='bold')
            ax.set_xlabel('Date', fontsize=12)
            ax.set_ylabel('Date', fontsize=12)
            
            plt.xticks(rotation=45, ha='right')
            plt.yticks(rotation=0)
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Saved similarity matrix to {save_path}")
            
            plt.show()
            
        except Exception as e:
            logger.error(f"Similarity matrix plotting failed: {e}")
            raise
    
    def plot_similar_windows(
        self,
        reference_date: str,
        reference_symbol: str,
        matches: List[Dict],
        max_plots: int = 6,
        figsize: Tuple[int, int] = (15, 10),
        save_path: Optional[str] = None
    ):
        """
        Plot reference window alongside its most similar matches.
        
        Args:
            reference_date: Reference date
            reference_symbol: Reference symbol
            matches: List of similar windows (from find_similar_to_date)
            max_plots: Maximum number of matches to plot
            figsize: Figure size
            save_path: Optional path to save figure
        """
        try:
            logger.info(f"Plotting similar windows to {reference_symbol}@{reference_date}")
            
            # Get reference window using SQL
            date_dt = pd.to_datetime(reference_date)
            sql = f"""
            SELECT window_id, timestamp, symbol, embedding, metadata
            FROM timeseries 
            WHERE symbol = '{reference_symbol}' 
            AND timestamp >= '{date_dt.strftime('%Y-%m-%d 00:00:00')}'
            AND timestamp < '{(date_dt + pd.Timedelta(days=1)).strftime('%Y-%m-%d 00:00:00')}'
            LIMIT 1
            """
            
            ref_windows = self.storage.query_sql(sql)
            
            if not ref_windows:
                logger.warning("Reference window not found")
                return
            
            # Determine grid layout
            n_matches = min(len(matches), max_plots)
            n_cols = 3
            n_rows = (n_matches + 2) // n_cols  # +1 for reference
            
            fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
            axes = axes.flatten() if n_rows > 1 else [axes]
            
            # Plot reference window (first subplot)
            ax_ref = axes[0]
            self._plot_single_window(
                ax_ref,
                ref_windows[0],
                title=f"REFERENCE: {reference_symbol} @ {reference_date}",
                color='red'
            )
            
            # Plot matches
            for i, match in enumerate(matches[:max_plots]):
                if i + 1 >= len(axes):
                    break
                
                ax = axes[i + 1]
                
                # Get match window by ID
                match_window = self.storage.read_by_id(match['window_id'])
                
                if match_window:
                    self._plot_single_window(
                        ax,
                        match_window,
                        title=f"{match['symbol']} @ {match['timestamp']}\nSimilarity: {match['similarity_score']:.3f}",
                        color='blue'
                    )
            
            # Hide unused subplots
            for i in range(n_matches + 1, len(axes)):
                axes[i].axis('off')
            
            plt.suptitle(
                f"Similar Windows to {reference_symbol} @ {reference_date}",
                fontsize=16,
                fontweight='bold',
                y=0.98
            )
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Saved similar windows plot to {save_path}")
            
            plt.show()
            
        except Exception as e:
            logger.error(f"Similar windows plotting failed: {e}")
            raise
    
    def _plot_single_window(
        self,
        ax,
        window_data: Dict,
        title: str,
        color: str = 'blue'
    ):
        """
        Plot a single window's data.
        
        Args:
            ax: Matplotlib axis
            window_data: Window dictionary with embedding
            title: Plot title
            color: Line color
        """
        # Get embedding
        import pickle
        embedding_bytes = window_data.get('embedding')
        
        if embedding_bytes is not None:
            # Deserialize if needed
            embedding = pickle.loads(embedding_bytes) if isinstance(embedding_bytes, bytes) else embedding_bytes
            
            # Plot embedding as a line (first 20 dimensions for visualization)
            n_dims_to_plot = min(20, len(embedding))
            ax.plot(range(n_dims_to_plot), embedding[:n_dims_to_plot], 
                    color=color, linewidth=2, alpha=0.7)
            ax.set_title(title, fontsize=10, fontweight='bold')
            ax.set_xlabel('Embedding Dimension', fontsize=8)
            ax.set_ylabel('Value', fontsize=8)
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'No embedding data', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(title, fontsize=10)


# ============================================================================
# EXAMPLES & TESTS
# ============================================================================

def test_similarity_engine():
    """Test the similarity engine."""
    print("=" * 80)
    print("SimilarityEngine Unit Tests")
    print("=" * 80)
    
    try:
        # Initialize storage and engine
        print("\nInitializing storage...")
        storage = StorageManager(
            duckdb_path="storage/financedb.duckdb",
            qdrant_path="storage/qdrant_data"
        )
        
        engine = SimilarityEngine(storage, distance_metric="cosine")
        print("✓ SimilarityEngine initialized")
        
        # First, let's check what data we have
        stats = storage.get_statistics()
        print(f"\nDatabase Statistics:")
        print(f"  Total windows: {stats['total_windows']:,}")
        print(f"  Unique symbols: {stats['unique_symbols']}")
        print(f"  Date range: {stats.get('date_range', {})}")
        
        # Get some sample symbols
        print("\nGetting sample symbols...")
        sample_query = "SELECT DISTINCT symbol FROM timeseries LIMIT 10"
        symbols_result = storage.query_sql(sample_query)
        sample_symbols = [row['symbol'] for row in symbols_result]
        
        if not sample_symbols:
            print("No symbols found in database. Exiting test.")
            storage.close()
            return
        
        print(f"Found {len(sample_symbols)} symbols: {', '.join(sample_symbols)}")
        
        # Get dates for the first symbol
        first_symbol = sample_symbols[0]
        dates_query = f"""
        SELECT timestamp FROM timeseries 
        WHERE symbol = '{first_symbol}' 
        ORDER BY timestamp 
        LIMIT 5
        """
        dates_result = storage.query_sql(dates_query)
        sample_dates = [row['timestamp'].strftime('%Y-%m-%d') for row in dates_result]
        
        if not sample_dates:
            print(f"No dates found for {first_symbol}. Exiting test.")
            storage.close()
            return
        
        print(f"\nSample dates for {first_symbol}:")
        for i, date in enumerate(sample_dates, 1):
            print(f"  {i}. {date}")
        
        # Test 1: Find similar to date
        print("\n" + "=" * 80)
        print("Test 1: Find Similar to Date")
        print("=" * 80)
        
        reference_date = sample_dates[0]
        reference_symbol = first_symbol
        
        print(f"\nSearching for windows similar to {reference_symbol} on {reference_date}")
        results = engine.find_similar_to_date(
            reference_date=reference_date,
            symbol=reference_symbol,
            top_k=5,
            min_similarity=0.6
        )
        
        if results:
            print(f"\nFound {len(results)} similar windows:")
            print("-" * 80)
            for i, result in enumerate(results, 1):
                print(f"{i}. {result['symbol']:6s} @ {result['timestamp']} | "
                      f"Score: {result['similarity_score']:.4f}")
            print("\n✓ Similar date search successful")
        else:
            print("\n⚠ No similar windows found (this is okay if data is sparse)")
        
        # Test 2: Compare two dates
        print("\n" + "=" * 80)
        print("Test 2: Compare Two Dates")
        print("=" * 80)
        
        if len(sample_dates) >= 2:
            comparison = engine.compare_dates(
                date1=sample_dates[0],
                symbol1=first_symbol,
                date2=sample_dates[1],
                symbol2=first_symbol
            )
            
            print(f"\nSimilarity between consecutive days:")
            print(f"  {comparison['symbol1']} @ {comparison['date1']}")
            print(f"  {comparison['symbol2']} @ {comparison['date2']}")
            print(f"  Similarity: {comparison['similarity_score']:.4f}")
            print("\n✓ Date comparison successful")
        else:
            print("Need at least 2 dates for comparison test")
        
        # Test 3: Cross-symbol search (simplified)
        print("\n" + "=" * 80)
        print("Test 3: Cross-Symbol Search")
        print("=" * 80)
        
        if len(sample_symbols) >= 2:
            print(f"\nTesting with {len(sample_symbols[:3])} symbols")
            
            cross_results = engine.find_similar_across_symbols(
                reference_date=reference_date,
                symbols=sample_symbols[:3],
                top_k=3,
                min_similarity=0.5
            )
            
            print(f"\nSimilar patterns across symbols for {reference_date}:")
            print("-" * 80)
            for symbol, matches in cross_results.items():
                print(f"\n{symbol}: {len(matches)} matches")
                for match in matches[:2]:
                    print(f"  - {match['symbol']} @ {match['timestamp']} | "
                          f"Score: {match['similarity_score']:.4f}")
            print("\n✓ Cross-symbol search successful")
        else:
            print("Need at least 2 symbols for cross-symbol search")
        
        # Test 4: Embedding similarity search
        print("\n" + "=" * 80)
        print("Test 4: Embedding Similarity Search")
        print("=" * 80)
        
        if results:
            print(f"\nUsing first result's embedding for similarity search...")
            query_embedding = results[0]['embedding']
            
            if query_embedding is not None:
                similar_to_emb = engine.find_similar_to_embedding(
                    embedding=query_embedding,
                    top_k=5,
                    min_similarity=0.5
                )
                
                print(f"\nFound {len(similar_to_emb)} windows similar to embedding:")
                for i, result in enumerate(similar_to_emb[:3], 1):
                    print(f"{i}. {result['symbol']:6s} @ {result['timestamp']} | "
                          f"Score: {result['similarity_score']:.4f}")
                print("\n✓ Embedding similarity search successful")
            else:
                print("No embedding available for test")
        
        # Cleanup
        storage.close()
        
        print("\n" + "=" * 80)
        print("✓ All tests completed!")
        print("=" * 80)
        
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()


def demo_similarity_search():
    """Interactive demo of similarity search."""
    print("\n" + "=" * 80)
    print("SimilarityEngine Interactive Demo")
    print("=" * 80)
    
    try:
        # Initialize
        storage = StorageManager(
            duckdb_path="storage/financedb.duckdb",
            qdrant_path="storage/qdrant_data"
        )
        
        engine = SimilarityEngine(storage)
        
        print("\n✓ Engine initialized")
        
        # Get statistics
        stats = storage.get_statistics()
        print(f"  Database: {stats['total_windows']:,} windows")
        print(f"  Unique symbols: {stats['unique_symbols']}")
        
        if stats['total_windows'] == 0:
            print("\n⚠ Database is empty. Please add data first.")
            storage.close()
            return
        
        # Get available symbols
        symbols_query = "SELECT DISTINCT symbol FROM timeseries ORDER BY symbol LIMIT 20"
        symbols_result = storage.query_sql(symbols_query)
        available_symbols = [row['symbol'] for row in symbols_result]
        
        print(f"\nAvailable symbols (first 20): {', '.join(available_symbols)}")
        
        # Demo queries with fallbacks
        demo_queries = []
        
        # Try to find some actual dates in the database
        for symbol in available_symbols[:3]:
            dates_query = f"""
            SELECT timestamp FROM timeseries 
            WHERE symbol = '{symbol}' 
            ORDER BY timestamp DESC
            LIMIT 3
            """
            dates_result = storage.query_sql(dates_query)
            
            if dates_result:
                for i, row in enumerate(dates_result):
                    if i >= 1:  # Just take one date per symbol
                        break
                    date_str = row['timestamp'].strftime('%Y-%m-%d')
                    demo_queries.append({
                        'name': f'{symbol} Example',
                        'date': date_str,
                        'symbol': symbol,
                        'description': f'Find patterns similar to {symbol} on {date_str}'
                    })
        
        if not demo_queries:
            print("\n⚠ No valid dates found for demo.")
            storage.close()
            return
        
        for i, query in enumerate(demo_queries[:3], 1):
            print(f"\n{'-' * 80}")
            print(f"Demo Query {i}: {query['name']}")
            print(f"  Date: {query['date']}")
            print(f"  Symbol: {query['symbol']}")
            print(f"  Goal: {query['description']}")
            print(f"{'-' * 80}")
            
            # Find similar windows
            results = engine.find_similar_to_date(
                reference_date=query['date'],
                symbol=query['symbol'],
                top_k=5,
                min_similarity=0.6
            )
            
            if results:
                print(f"\nTop similar windows:")
                for j, result in enumerate(results, 1):
                    print(f"  {j}. {result['symbol']:6s} @ {result['timestamp']} | "
                          f"Similarity: {result['similarity_score']:.4f}")
            else:
                print(f"\n⚠ No similar windows found (try lowering min_similarity)")
        
        # Cleanup
        storage.close()
        
        print("\n" + "=" * 80)
        print("✓ Demo completed!")
        print("=" * 80)
        
    except Exception as e:
        print(f"\n✗ Demo failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == '--demo':
        demo_similarity_search()
    else:
        test_similarity_engine()