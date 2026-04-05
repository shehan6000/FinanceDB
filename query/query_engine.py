"""
FinanceDB - Query Engine
High-level interface for executing queries on compressed financial data.

Author: FinanceDB Project
Module: 4 - Query Processing (Main Interface)
"""

import sys
import time
from pathlib import Path
from typing import List, Dict, Optional, Any
from datetime import datetime
import numpy as np
import pandas as pd
import torch
from loguru import logger
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from query.parser import QueryParser, Query
from query.optimizer import QueryOptimizer, ExecutionPlan
from query.executor import ExecutionEngine, QueryResult
from storage.storage_manager import StorageManager
from models.compression_model import CompressionModel

# Initialize rich console
console = Console()


class QueryEngine:
    """
    High-level query engine for FinanceDB.
    
    Provides a simple interface for executing SQL and semantic queries
    on compressed financial time-series data.
    
    Features:
    - Parse, optimize, and execute queries in one call
    - Explain query plans without execution
    - Benchmark query performance
    - Error handling and logging
    - Optional query result caching
    """
    
    def __init__(
        self,
        storage_manager: StorageManager,
        compression_model: Optional[CompressionModel] = None,
        device: str = "auto",
        enable_cache: bool = False,
        cache_size: int = 100
    ):
        """
        Initialize query engine.
        
        Args:
            storage_manager: StorageManager instance
            compression_model: Trained compression model (for decompression)
            device: Device for model inference ('cpu', 'cuda', or 'auto')
            enable_cache: Whether to cache query results
            cache_size: Maximum number of cached results
        """
        self.storage = storage_manager
        
        # Determine device
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        # Initialize components
        self.parser = QueryParser()
        
        # Get total windows for optimizer cost model
        total_windows = self.storage.duckdb.count()
        self.optimizer = QueryOptimizer(total_windows=total_windows)
        
        self.executor = ExecutionEngine(
            storage_manager=storage_manager,
            compression_model=compression_model,
            device=self.device
        )
        
        # Optional caching
        self.enable_cache = enable_cache
        self.cache_size = cache_size
        self.query_cache: Dict[str, QueryResult] = {}
        
        logger.info("Initialized QueryEngine")
        logger.info(f"  Total windows: {total_windows:,}")
        logger.info(f"  Device: {self.device}")
        logger.info(f"  Cache: {'Enabled' if enable_cache else 'Disabled'}")
    
    def execute(
        self,
        query_string: str,
        use_cache: bool = True,
        verbose: bool = False
    ) -> QueryResult:
        """
        Execute a query and return results.
        
        Args:
            query_string: SQL or semantic query string
            use_cache: Whether to use cached results
            verbose: Whether to print execution details
            
        Returns:
            QueryResult with data and performance metrics
            
        Examples:
            # SQL query
            result = engine.execute("SELECT * FROM timeseries WHERE symbol = 'AAPL'")
            
            # Similarity search
            result = engine.execute("SELECT TOP 10 SIMILAR_TO('2020-03-16')")
            
            # Hybrid query
            result = engine.execute(
                "SELECT TOP 5 * WHERE symbol = 'AAPL' AND SIMILAR_TO('crash_pattern')"
            )
        """
        try:
            # Check cache
            if self.enable_cache and use_cache and query_string in self.query_cache:
                logger.debug(f"Cache hit for query: {query_string[:50]}...")
                if verbose:
                    console.print("[green]✓ Using cached result[/green]")
                return self.query_cache[query_string]
            
            if verbose:
                console.print(f"\n[bold cyan]Executing Query:[/bold cyan] {query_string}")
            
            # Step 1: Parse
            if verbose:
                console.print("[dim]1. Parsing query...[/dim]")
            parsed_query = self.parser.parse(query_string)
            
            if verbose:
                console.print(f"   → Type: {parsed_query.type.value}")
            
            # Step 2: Optimize
            if verbose:
                console.print("[dim]2. Optimizing execution plan...[/dim]")
            plan = self.optimizer.optimize(parsed_query)
            
            if verbose:
                # Count steps in the execution tree
                def count_steps(step):
                    count = 1
                    if step.children:
                        for child in step.children:
                            count += count_steps(child)
                    return count
                
                total_steps = count_steps(plan.root_step)
                console.print(f"   → Steps: {total_steps}")
                console.print(f"   → Estimated cost: {plan.total_cost_ms:.2f} ms")
            
            # Step 3: Execute
            if verbose:
                console.print("[dim]3. Executing plan...[/dim]")
            result = self.executor.execute(plan)
            
            if verbose:
                console.print(f"   → Actual time: {result.execution_time_ms:.2f} ms")
                console.print(f"   → Rows returned: {len(result.data) if result.data else 0}")
            
            # Cache result
            if self.enable_cache:
                self._cache_result(query_string, result)
            
            if verbose:
                console.print("[green]✓ Query completed successfully[/green]\n")
            
            return result
            
        except Exception as e:
            logger.error(f"Query execution failed: {e}")
            if verbose:
                console.print(f"[red]✗ Query failed: {e}[/red]")
            raise
    
    def explain(
        self,
        query_string: str,
        show_cost: bool = True
    ) -> ExecutionPlan:
        """
        Show execution plan without running query.
        
        Args:
            query_string: SQL or semantic query string
            show_cost: Whether to display cost estimates
            
        Returns:
            ExecutionPlan object
            
        Example:
            plan = engine.explain("SELECT TOP 10 SIMILAR_TO('2020-03-16')")
            print(plan.explain())
        """
        try:
            # Parse and optimize
            parsed_query = self.parser.parse(query_string)
            plan = self.optimizer.optimize(parsed_query)
            
            # Display plan
            if show_cost:
                console.print(plan.explain())
            
            return plan
            
        except Exception as e:
            logger.error(f"Query parsing/optimization failed: {e}")
            console.print(f"[red]✗ EXPLAIN failed: {e}[/red]")
            raise
    
    def benchmark(
        self,
        query_string: str,
        runs: int = 10,
        warmup: int = 2
    ) -> Dict[str, float]:
        """
        Benchmark query performance.
        
        Args:
            query_string: SQL or semantic query string
            runs: Number of benchmark runs
            warmup: Number of warmup runs (not counted)
            
        Returns:
            Dictionary with performance statistics
            
        Example:
            stats = engine.benchmark("SELECT TOP 5 SIMILAR_TO('2020-03-16')", runs=10)
            print(f"Average: {stats['avg_ms']:.2f} ms")
        """
        try:
            console.print(f"\n[bold cyan]Benchmarking Query ({runs} runs):[/bold cyan]")
            console.print(f"{query_string}\n")
            
            # Warmup runs
            console.print(f"[dim]Running {warmup} warmup iterations...[/dim]")
            for i in range(warmup):
                self.execute(query_string, use_cache=False, verbose=False)
            
            # Actual benchmark runs
            console.print(f"[dim]Running {runs} benchmark iterations...[/dim]")
            timings = []
            
            for i in range(runs):
                result = self.execute(query_string, use_cache=False, verbose=False)
                timings.append(result.execution_time_ms)
                console.print(f"  Run {i+1}/{runs}: {result.execution_time_ms:.2f} ms")
            
            # Calculate statistics
            stats = {
                'runs': runs,
                'timings_ms': timings,
                'min_ms': min(timings),
                'max_ms': max(timings),
                'avg_ms': sum(timings) / len(timings),
                'median_ms': sorted(timings)[len(timings) // 2],
                'std_ms': np.std(timings),
            }
            
            # Display summary
            console.print(f"\n[bold]Benchmark Results:[/bold]")
            console.print(f"  Average:  {stats['avg_ms']:.2f} ms")
            console.print(f"  Median:   {stats['median_ms']:.2f} ms")
            console.print(f"  Min:      {stats['min_ms']:.2f} ms")
            console.print(f"  Max:      {stats['max_ms']:.2f} ms")
            console.print(f"  Std Dev:  {stats['std_ms']:.2f} ms")
            
            return stats
            
        except Exception as e:
            logger.error(f"Benchmark failed: {e}")
            console.print(f"[red]✗ Benchmark failed: {e}[/red]")
            raise
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get database and cache statistics.
        
        Returns:
            Dictionary with database statistics
        """
        try:
            stats = self.storage.get_statistics()
            
            if self.enable_cache:
                stats['cache'] = {
                    'enabled': True,
                    'size': len(self.query_cache),
                    'max_size': self.cache_size
                }
            
            return stats
            
        except Exception as e:
            logger.warning(f"Could not get full statistics: {e}")
            # Return basic statistics
            basic_stats = {
                'total_windows': self.storage.duckdb.count(),
                'cache': {
                    'enabled': self.enable_cache,
                    'size': len(self.query_cache),
                    'max_size': self.cache_size
                }
            }
            return basic_stats
    
    def clear_cache(self):
        """Clear query result cache."""
        if self.enable_cache:
            self.query_cache.clear()
            logger.info("Query cache cleared")
            console.print("[green]✓ Cache cleared[/green]")
    
    def _cache_result(self, query_string: str, result: QueryResult):
        """Add result to cache, evicting oldest if necessary."""
        if len(self.query_cache) >= self.cache_size:
            # Simple FIFO eviction
            oldest_query = next(iter(self.query_cache))
            del self.query_cache[oldest_query]
        
        self.query_cache[query_string] = result
    
    def close(self):
        """Close storage connections."""
        self.storage.close()
        logger.info("QueryEngine closed")


# ============================================================================
# DEMO & EXAMPLES
# ============================================================================

def demo_query_engine():
    """Demonstrate query engine capabilities."""
    console.print(Panel.fit(
        "[bold yellow]FinanceDB Query Engine Demo[/bold yellow]\n"
        "Execute SQL and semantic queries on compressed data",
        border_style="cyan"
    ))
    
    try:
        # Initialize storage and model
        console.print("\n[bold]1. Initializing Query Engine[/bold]")
        
        storage = StorageManager(
            duckdb_path="storage/financedb.duckdb",
            qdrant_path="storage/qdrant_data"
        )
        
        model_path = Path("models/checkpoints/best_model.pt")
        if model_path.exists():
            model, _ = CompressionModel.load_checkpoint(str(model_path))
            console.print("   ✓ Loaded compression model")
        else:
            model = None
            console.print("   ⚠ No compression model (decompression disabled)")
        
        engine = QueryEngine(
            storage_manager=storage,
            compression_model=model,
            enable_cache=True
        )
        
        console.print(f"   ✓ Ready! Database has {storage.duckdb.count():,} windows\n")
        
        # Example 1: SQL Query
        console.print("[bold]2. SQL Query Example[/bold]")
        query1 = "SELECT * FROM timeseries WHERE symbol = 'AAPL' LIMIT 5"
        result1 = engine.execute(query1, verbose=True)
        
        # Display results
        if result1.data:
            console.print(f"\n[bold]Results ({len(result1.data)} rows):[/bold]")
            table = Table(show_header=True, header_style="bold cyan")
            if result1.data:
                # Get column names from first row
                columns = list(result1.data[0].keys())
                for col in columns:
                    table.add_column(col)
                
                # Add rows (limit to 5 for display)
                for i, row in enumerate(result1.data[:5]):
                    table.add_row(*[str(row.get(col, '')) for col in columns])
                console.print(table)
        
        # Example 2: Similarity Search
        console.print("\n[bold]3. Similarity Search Example[/bold]")
        query2 = "SELECT TOP 5 SIMILAR_TO('2020-03-16', threshold=0.8)"
        result2 = engine.execute(query2, verbose=True)
        
        # Display results
        if result2.data:
            console.print(f"\n[bold]Results ({len(result2.data)} rows):[/bold]")
            table = Table(show_header=True, header_style="bold cyan")
            if result2.data:
                # Get column names from first row
                columns = list(result2.data[0].keys())
                for col in columns:
                    table.add_column(col)
                
                # Add rows (limit to 5 for display)
                for i, row in enumerate(result2.data[:5]):
                    table.add_row(*[str(row.get(col, '')) for col in columns])
                console.print(table)
        
        # Example 3: EXPLAIN
        console.print("\n[bold]4. Query Plan Example (EXPLAIN)[/bold]")
        query3 = "SELECT TOP 10 * WHERE symbol = 'MSFT' AND SIMILAR_TO('pattern')"
        plan = engine.explain(query3)
        
        # Example 4: Benchmark
        console.print("\n[bold]5. Benchmark Example[/bold]")
        query4 = "SELECT TOP 5 SIMILAR_TO('2020-03-16')"
        stats = engine.benchmark(query4, runs=5, warmup=1)
        
        # Example 5: Database Statistics
        console.print("\n[bold]6. Database Statistics[/bold]")
        try:
            db_stats = engine.get_statistics()
            
            table = Table(show_header=True, header_style="bold cyan")
            table.add_column("Metric", style="dim")
            table.add_column("Value", justify="right")
            
            # Safely get statistics with defaults
            total_windows = db_stats.get('total_windows', storage.duckdb.count())
            table.add_row("Total Windows", f"{total_windows:,}")
            
            unique_symbols = db_stats.get('unique_symbols', 'N/A')
            table.add_row("Unique Symbols", f"{unique_symbols}")
            
            date_range = db_stats.get('date_range', {})
            table.add_row("Date Range", f"{date_range.get('start', 'N/A')} to {date_range.get('end', 'N/A')}")
            
            storage_stats = db_stats.get('storage', {})
            table.add_row("Storage (DuckDB)", f"{storage_stats.get('duckdb_mb', 'N/A'):.2f} MB")
            table.add_row("Storage (Qdrant)", f"{storage_stats.get('qdrant_mb', 'N/A'):.2f} MB")
            table.add_row("Total Storage", f"{storage_stats.get('total_mb', 'N/A'):.2f} MB")
            
            console.print(table)
            
        except Exception as e:
            console.print(f"[yellow]⚠ Statistics unavailable: {e}[/yellow]")
            # Show basic statistics
            table = Table(show_header=True, header_style="bold cyan")
            table.add_column("Metric", style="dim")
            table.add_column("Value", justify="right")
            
            try:
                total_windows = storage.duckdb.count()
                table.add_row("Total Windows", f"{total_windows:,}")
                
                # Try to get unique symbols
                try:
                    unique_result = storage.duckdb.execute_query("SELECT COUNT(DISTINCT symbol) FROM timeseries")
                    unique_symbols = unique_result[0][0] if unique_result else "N/A"
                except:
                    unique_symbols = "N/A"
                table.add_row("Unique Symbols", f"{unique_symbols}")
                
                # Try to get date range
                try:
                    date_result = storage.duckdb.execute_query(
                        "SELECT MIN(timestamp) as start, MAX(timestamp) as end FROM timeseries"
                    )
                    if date_result:
                        start = date_result[0][0].strftime('%Y-%m-%d') if date_result[0][0] else 'N/A'
                        end = date_result[0][1].strftime('%Y-%m-%d') if date_result[0][1] else 'N/A'
                        date_range = f"{start} to {end}"
                    else:
                        date_range = "N/A to N/A"
                except:
                    date_range = "N/A to N/A"
                table.add_row("Date Range", date_range)
                
                console.print(table)
            except Exception as e2:
                console.print(f"[red]Failed to get basic statistics: {e2}[/red]")
        
        # Cleanup
        engine.close()
        
        console.print("\n[green]✓ Demo completed successfully![/green]")
        
    except Exception as e:
        console.print(f"\n[red]✗ Demo failed: {e}[/red]")
        import traceback
        traceback.print_exc()


def interactive_shell():
    """Interactive query shell."""
    console.print(Panel.fit(
        "[bold yellow]FinanceDB Interactive Query Shell[/bold yellow]\n"
        "Type queries or commands. Type 'help' for options, 'exit' to quit.",
        border_style="cyan"
    ))
    
    try:
        # Initialize
        storage = StorageManager(
            duckdb_path="storage/financedb.duckdb",
            qdrant_path="storage/qdrant_data"
        )
        
        model_path = Path("models/checkpoints/best_model.pt")
        if model_path.exists():
            model, _ = CompressionModel.load_checkpoint(str(model_path))
        else:
            model = None
        
        engine = QueryEngine(storage, model, enable_cache=True)
        
        console.print(f"\n✓ Connected to database ({storage.duckdb.count():,} windows)")
        console.print("Type 'help' for available commands\n")
        
        # Main loop
        while True:
            try:
                query = console.input("[bold cyan]financedb>[/bold cyan] ")
                
                if not query.strip():
                    continue
                
                query_lower = query.strip().lower()
                
                # Commands
                if query_lower in ['exit', 'quit', 'q']:
                    console.print("[yellow]Goodbye![/yellow]")
                    break
                
                elif query_lower == 'help':
                    console.print("\n[bold]Available Commands:[/bold]")
                    console.print("  exit, quit, q     - Exit the shell")
                    console.print("  help              - Show this help")
                    console.print("  stats             - Show database statistics")
                    console.print("  clear             - Clear query cache")
                    console.print("  explain <query>   - Show execution plan")
                    console.print("\n[bold]Example Queries:[/bold]")
                    console.print("  SELECT * FROM timeseries WHERE symbol = 'AAPL' LIMIT 10")
                    console.print("  SELECT TOP 5 SIMILAR_TO('2020-03-16', threshold=0.85)")
                    console.print("  SELECT symbol, COUNT(*) FROM timeseries GROUP BY symbol\n")
                
                elif query_lower == 'stats':
                    try:
                        stats = engine.get_statistics()
                        console.print(f"\n[bold]Database Statistics:[/bold]")
                        console.print(f"  Total windows: {stats.get('total_windows', 'N/A'):,}")
                        console.print(f"  Unique symbols: {stats.get('unique_symbols', 'N/A')}")
                        storage_info = stats.get('storage', {})
                        console.print(f"  Storage: {storage_info.get('total_mb', 'N/A'):.2f} MB")
                    except Exception as e:
                        console.print(f"[yellow]⚠ Statistics unavailable: {e}[/yellow]")
                        console.print(f"  Total windows: {storage.duckdb.count():,}")
                    console.print()
                
                elif query_lower == 'clear':
                    engine.clear_cache()
                
                elif query_lower.startswith('explain '):
                    explain_query = query[8:].strip()
                    plan = engine.explain(explain_query)
                
                else:
                    # Execute query
                    result = engine.execute(query, verbose=True)
                    
                    # Show results
                    if result.data:
                        console.print(f"\n[bold]Results ({len(result.data)} rows):[/bold]")
                        if len(result.data) > 0:
                            # Create a table with all columns
                            table = Table(show_header=True, header_style="bold cyan")
                            columns = list(result.data[0].keys())
                            for col in columns:
                                table.add_column(col)
                            
                            # Add rows (limit to 10 for display)
                            for i, row in enumerate(result.data[:10]):
                                table.add_row(*[str(row.get(col, '')) for col in columns])
                            console.print(table)
                            
                            if len(result.data) > 10:
                                console.print(f"... ({len(result.data) - 10} more rows)")
                        else:
                            console.print("No results found")
                    
                    console.print()
            
            except KeyboardInterrupt:
                console.print("\n[yellow]Query interrupted[/yellow]")
                continue
            
            except Exception as e:
                console.print(f"[red]Error: {e}[/red]")
                continue
        
        # Cleanup
        engine.close()
        
    except Exception as e:
        console.print(f"\n[red]Failed to start shell: {e}[/red]")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == '--shell':
        # Interactive shell mode
        interactive_shell()
    else:
        # Demo mode
        demo_query_engine()