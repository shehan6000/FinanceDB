"""
FinanceDB - Query Optimizer
Compression-aware query optimization with cost-based planning.

Author: FinanceDB Project
Module: 4 - Query Processing (Optimizer)
"""

import math
import json
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple, Any
from enum import Enum

# Import from the same directory
try:
    from .parser import Query, QueryType
except ImportError:
    # Fallback for direct execution
    from parser import Query, QueryType


class OperatorType(Enum):
    """Physical operator types."""
    VECTOR_SEARCH = "vector_search"       # Qdrant similarity search
    SQL_SCAN = "sql_scan"                 # DuckDB table scan
    INDEX_LOOKUP = "index_lookup"         # DuckDB index scan
    DECOMPRESS = "decompress"             # Decode embeddings
    FILTER = "filter"                     # Apply predicates
    AGGREGATE = "aggregate"               # Grouping/aggregation
    SORT = "sort"                         # Order results
    LIMIT = "limit"                       # Top-k selection
    UNION = "union"                       # Combine results
    PROJECT = "project"                   # Select columns


@dataclass
class ExecutionStep:
    """Single step in execution plan."""
    operator: OperatorType
    description: str
    estimated_rows: int
    estimated_cost_ms: float
    children: Optional[List['ExecutionStep']] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        """Validate the step after initialization."""
        if self.estimated_rows < 0:
            raise ValueError(f"estimated_rows cannot be negative: {self.estimated_rows}")
        if self.estimated_cost_ms < 0:
            raise ValueError(f"estimated_cost_ms cannot be negative: {self.estimated_cost_ms}")
        
        # Ensure children is a list
        if self.children is None:
            self.children = []


@dataclass
class ExecutionPlan:
    """Complete query execution plan."""
    query: Query
    root_step: ExecutionStep
    total_cost_ms: float
    total_rows: int
    uses_compression: bool
    is_approximate: bool = False
    
    def explain(self, verbose: bool = False) -> str:
        """Generate EXPLAIN-style output.
        
        Args:
            verbose: If True, include all metadata details
        """
        lines = []
        lines.append("=" * 80)
        lines.append("QUERY EXECUTION PLAN")
        lines.append("=" * 80)
        lines.append(f"Query Type: {self.query.type.value.upper()}")
        lines.append(f"Total Estimated Cost: {self.total_cost_ms:.2f} ms")
        lines.append(f"Estimated Rows: {self.total_rows:,}")
        lines.append(f"Uses Compression: {'Yes' if self.uses_compression else 'No'}")
        lines.append(f"Approximate Results: {'Yes' if self.is_approximate else 'No'}")
        lines.append("")
        lines.append("Execution Tree:")
        lines.append("-" * 80)
        
        def format_step(step: ExecutionStep, level: int = 0) -> List[str]:
            """Recursively format execution step tree."""
            indent = "  " * level
            step_lines = []
            
            # Step header
            step_lines.append(f"{indent}{step.operator.value.upper()}")
            step_lines.append(f"{indent}  Description: {step.description}")
            step_lines.append(f"{indent}  Rows: {step.estimated_rows:,}")
            step_lines.append(f"{indent}  Cost: {step.estimated_cost_ms:.2f} ms")
            
            # Metadata (if verbose or important)
            if step.metadata:
                if verbose:
                    for key, value in step.metadata.items():
                        step_lines.append(f"{indent}  {key}: {value}")
                else:
                    # Show only critical metadata
                    critical_keys = {'threshold', 'top_k', 'approximate', 'indexed', 'filtered_search'}
                    for key, value in step.metadata.items():
                        if key in critical_keys:
                            step_lines.append(f"{indent}  {key}: {value}")
            
            # Recursively process children
            if step.children:
                step_lines.append(f"{indent}  Children:")
                for child in step.children:
                    step_lines.extend(format_step(child, level + 2))
            
            return step_lines
        
        lines.extend(format_step(self.root_step))
        lines.append("=" * 80)
        return "\n".join(lines)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert plan to dictionary for serialization."""
        def step_to_dict(step: ExecutionStep) -> Dict[str, Any]:
            """Recursively convert step to dictionary."""
            step_dict = {
                'operator': step.operator.value,
                'description': step.description,
                'estimated_rows': step.estimated_rows,
                'estimated_cost_ms': step.estimated_cost_ms,
            }
            if step.metadata:
                step_dict['metadata'] = step.metadata
            if step.children:
                step_dict['children'] = [step_to_dict(c) for c in step.children]
            return step_dict
        
        return {
            'query_type': self.query.type.value,
            'total_cost_ms': self.total_cost_ms,
            'total_rows': self.total_rows,
            'uses_compression': self.uses_compression,
            'is_approximate': self.is_approximate,
            'plan': step_to_dict(self.root_step)
        }


class QueryOptimizer:
    """
    Compression-aware query optimizer.
    
    Features:
    - Cost-based optimization
    - Decides between vector search and SQL
    - Determines when to decompress
    - Generates execution plans
    """
    
    def __init__(
        self,
        total_windows: int = 25000,
        avg_embedding_dim: int = 64,
        enable_approximation: bool = True,
        enable_caching: bool = True
    ):
        """
        Initialize optimizer.
        
        Args:
            total_windows: Total number of windows in database
            avg_embedding_dim: Average embedding dimension
            enable_approximation: Allow approximate aggregations
            enable_caching: Enable plan caching
        """
        if total_windows <= 0:
            raise ValueError("total_windows must be positive")
        if avg_embedding_dim <= 0:
            raise ValueError("avg_embedding_dim must be positive")
            
        self.total_windows = total_windows
        self.avg_embedding_dim = avg_embedding_dim
        self.enable_approximation = enable_approximation
        self.enable_caching = enable_caching
        
        # Cache for previously optimized queries
        self.plan_cache: Dict[str, ExecutionPlan] = {}
        
        # Cost model parameters (in milliseconds)
        # These should be calibrated based on actual performance measurements
        self.costs = {
            'scan_per_row': 0.001,        # Sequential scan per row
            'index_lookup': 0.01,          # Index lookup overhead
            'decompress_per_row': 0.1,     # Decompression per window
            'vector_search_base': 1.0,     # HNSW base cost
            'vector_search_log': 0.5,      # HNSW log factor
            'filter_per_row': 0.0005,      # Predicate evaluation
            'aggregate_per_row': 0.002,    # Aggregation per row
            'sort_per_row': 0.003,         # Sorting per row
            'union_per_row': 0.001,        # Union operation per row
            'project_per_row': 0.0001,     # Projection per row
        }
        
        # Statistics for better selectivity estimation
        self.stats = {
            'total_symbols': 25,
            'default_selectivity': 0.1,
            'anomaly_rate': 0.05,  # Assume 5% anomalies
        }
    
    def optimize(self, query: Query) -> ExecutionPlan:
        """
        Generate optimal execution plan for query.
        
        Args:
            query: Parsed query object
            
        Returns:
            ExecutionPlan with steps and cost estimates
            
        Raises:
            ValueError: If query is invalid
        """
        if not query:
            raise ValueError("Query cannot be None")
        
        # Check cache first
        cache_key = self._generate_cache_key(query)
        if self.enable_caching and cache_key in self.plan_cache:
            cached_plan = self.plan_cache[cache_key]
            # Update query reference (cache stores optimized plan structure)
            cached_plan.query = query
            return cached_plan
        
        try:
            # Route to specialized optimizer based on query type
            if query.type == QueryType.SIMILARITY:
                plan = self._optimize_similarity(query)
            elif query.type == QueryType.PATTERN:
                plan = self._optimize_pattern(query)
            elif query.type == QueryType.ANOMALY:
                plan = self._optimize_anomaly(query)
            elif query.type == QueryType.HYBRID:
                plan = self._optimize_hybrid(query)
            elif query.type == QueryType.SQL:
                plan = self._optimize_sql(query)
            else:
                raise ValueError(f"Unsupported query type: {query.type}")
            
            # Cache the plan
            if self.enable_caching:
                self.plan_cache[cache_key] = plan
            
            return plan
            
        except Exception as e:
            raise RuntimeError(f"Failed to optimize query: {str(e)}") from e
    
    def _generate_cache_key(self, query: Query) -> str:
        """Generate cache key for query."""
        # Create a unique key based on query components
        components = [
            query.type.value,
            str(query.top_k),
            str(query.similarity_threshold) if query.similarity_threshold else "None",
            str(query.anomaly_threshold) if query.anomaly_threshold else "None",
            str(query.similarity_ref) if query.similarity_ref else "None",
            str(query.pattern_name) if query.pattern_name else "None",
        ]
        
        if query.filters:
            components.append(json.dumps(query.filters, sort_keys=True))
        if query.aggregations:
            components.append(json.dumps(sorted(query.aggregations), sort_keys=True))
        if query.order_by:
            components.append(json.dumps(query.order_by, sort_keys=True))
        if query.limit:
            components.append(str(query.limit))
        if query.columns:
            components.append(json.dumps(sorted(query.columns), sort_keys=True))
        
        return "|".join(components)
    
    def _optimize_similarity(self, query: Query) -> ExecutionPlan:
        """Optimize pure similarity search query."""
        if not query.similarity_ref:
            raise ValueError("Similarity reference required for similarity search")
        
        # Step 1: Vector search in Qdrant
        search_cost = self._estimate_vector_search_cost(
            self.total_windows,
            query.top_k
        )
        
        vector_search = ExecutionStep(
            operator=OperatorType.VECTOR_SEARCH,
            description=f"HNSW similarity search for '{query.similarity_ref}'",
            estimated_rows=query.top_k,
            estimated_cost_ms=search_cost,
            metadata={
                'threshold': query.similarity_threshold,
                'top_k': query.top_k,
                'reference': query.similarity_ref
            }
        )
        
        root_step = vector_search
        
        # Step 2: Optional decompression for results
        if self._needs_decompression(query):
            decompress_cost = query.top_k * self.costs['decompress_per_row']
            decompress_step = ExecutionStep(
                operator=OperatorType.DECOMPRESS,
                description="Decompress top-k embeddings to features",
                estimated_rows=query.top_k,
                estimated_cost_ms=decompress_cost,
                children=[root_step]
            )
            root_step = decompress_step
        
        # Step 3: Project selected columns if specified
        if query.columns and query.columns != ["*"]:
            project_cost = query.top_k * self.costs['project_per_row']
            project_step = ExecutionStep(
                operator=OperatorType.PROJECT,
                description=f"Select columns: {', '.join(query.columns)}",
                estimated_rows=query.top_k,
                estimated_cost_ms=project_cost,
                children=[root_step]
            )
            root_step = project_step
        
        # Step 4: Apply limit if not already handled
        if query.limit and query.limit < query.top_k:
            limit_step = ExecutionStep(
                operator=OperatorType.LIMIT,
                description=f"Limit to {query.limit} results",
                estimated_rows=query.limit,
                estimated_cost_ms=0.01,
                children=[root_step]
            )
            root_step = limit_step
        
        total_cost = self._calculate_total_cost(root_step)
        
        return ExecutionPlan(
            query=query,
            root_step=root_step,
            total_cost_ms=total_cost,
            total_rows=root_step.estimated_rows,
            uses_compression=True,
            is_approximate=False
        )
    
    def _optimize_pattern(self, query: Query) -> ExecutionPlan:
        """Optimize pattern matching query."""
        if not query.pattern_name:
            raise ValueError("Pattern name required for pattern matching")
        
        # Step 1: Load pattern embedding
        load_pattern = ExecutionStep(
            operator=OperatorType.INDEX_LOOKUP,
            description=f"Load pattern '{query.pattern_name}' embedding",
            estimated_rows=1,
            estimated_cost_ms=self.costs['index_lookup'],
            metadata={'pattern': query.pattern_name}
        )
        
        # Step 2: Vector search
        search_cost = self._estimate_vector_search_cost(
            self.total_windows,
            query.top_k
        )
        
        vector_search = ExecutionStep(
            operator=OperatorType.VECTOR_SEARCH,
            description=f"Find windows matching pattern '{query.pattern_name}'",
            estimated_rows=query.top_k,
            estimated_cost_ms=search_cost,
            children=[load_pattern],
            metadata={
                'threshold': query.similarity_threshold,
                'top_k': query.top_k,
                'pattern': query.pattern_name
            }
        )
        
        root_step = vector_search
        
        total_cost = self._calculate_total_cost(root_step)
        
        return ExecutionPlan(
            query=query,
            root_step=root_step,
            total_cost_ms=total_cost,
            total_rows=query.top_k,
            uses_compression=True,
            is_approximate=False
        )
    
    def _optimize_anomaly(self, query: Query) -> ExecutionPlan:
        """Optimize anomaly detection query."""
        # Estimate selectivity from filters
        estimated_rows = self._estimate_selectivity(query.filters)
        
        root_step = None
        
        # Step 1: Scan with filters (if any)
        if query.filters:
            filter_cost = estimated_rows * self.costs['filter_per_row']
            filter_step = ExecutionStep(
                operator=OperatorType.FILTER,
                description=f"Apply SQL filters: {query.filters}",
                estimated_rows=estimated_rows,
                estimated_cost_ms=filter_cost
            )
            root_step = filter_step
        else:
            scan_cost = self.total_windows * self.costs['scan_per_row']
            scan_step = ExecutionStep(
                operator=OperatorType.SQL_SCAN,
                description="Sequential scan on timeseries",
                estimated_rows=self.total_windows,
                estimated_cost_ms=scan_cost
            )
            root_step = scan_step
        
        # Step 2: Decompress and compute reconstruction error
        decompress_cost = root_step.estimated_rows * self.costs['decompress_per_row']
        decompress_step = ExecutionStep(
            operator=OperatorType.DECOMPRESS,
            description="Decompress embeddings and compute reconstruction error",
            estimated_rows=root_step.estimated_rows,
            estimated_cost_ms=decompress_cost,
            children=[root_step],
            metadata={'compute_error': True}
        )
        root_step = decompress_step
        
        # Step 3: Filter by anomaly threshold
        anomaly_rows = int(root_step.estimated_rows * self.stats['anomaly_rate'])
        filter_cost = root_step.estimated_rows * self.costs['filter_per_row']
        anomaly_filter = ExecutionStep(
            operator=OperatorType.FILTER,
            description=f"Filter ANOMALY_SCORE > {query.anomaly_threshold}",
            estimated_rows=anomaly_rows,
            estimated_cost_ms=filter_cost,
            children=[root_step]
        )
        root_step = anomaly_filter
        
        total_cost = self._calculate_total_cost(root_step)
        
        return ExecutionPlan(
            query=query,
            root_step=root_step,
            total_cost_ms=total_cost,
            total_rows=anomaly_rows,
            uses_compression=False,  # Requires decompression
            is_approximate=False
        )
    
    def _optimize_hybrid(self, query: Query) -> ExecutionPlan:
        """Optimize hybrid SQL + similarity query."""
        if not query.similarity_ref:
            raise ValueError("Similarity reference required for hybrid query")
        
        # Estimate selectivity for SQL filters
        sql_selectivity = self._estimate_selectivity(query.filters)
        
        # Evaluate both approaches
        filter_first_cost, filter_first_rows = self._evaluate_filter_first(
            sql_selectivity, query
        )
        search_first_cost, search_first_rows = self._evaluate_search_first(query)
        
        # Choose better approach
        if filter_first_cost < search_first_cost:
            return self._build_filter_first_plan(sql_selectivity, query)
        else:
            return self._build_search_first_plan(query)
    
    def _evaluate_filter_first(self, sql_selectivity: int, query: Query) -> Tuple[float, int]:
        """Evaluate cost of filter-first approach."""
        filter_cost = sql_selectivity * self.costs['filter_per_row']
        search_cost = self._estimate_vector_search_cost(sql_selectivity, query.top_k)
        total_cost = filter_cost + search_cost
        return total_cost, query.top_k
    
    def _evaluate_search_first(self, query: Query) -> Tuple[float, int]:
        """Evaluate cost of search-first approach."""
        # Over-fetch to ensure we get enough results after filtering
        over_fetch_factor = 3  # Fetch 3x more than needed
        search_cost = self._estimate_vector_search_cost(
            self.total_windows,
            query.top_k * over_fetch_factor
        )
        filter_cost = query.top_k * over_fetch_factor * self.costs['filter_per_row']
        total_cost = search_cost + filter_cost
        return total_cost, query.top_k
    
    def _build_filter_first_plan(self, sql_selectivity: int, query: Query) -> ExecutionPlan:
        """Build filter-first execution plan."""
        # Filter step
        filter_step = ExecutionStep(
            operator=OperatorType.FILTER,
            description=f"Apply SQL filters: {query.filters}",
            estimated_rows=sql_selectivity,
            estimated_cost_ms=sql_selectivity * self.costs['filter_per_row']
        )
        
        # Search step
        search_cost = self._estimate_vector_search_cost(sql_selectivity, query.top_k)
        search_step = ExecutionStep(
            operator=OperatorType.VECTOR_SEARCH,
            description=f"Similarity search on filtered subset",
            estimated_rows=query.top_k,
            estimated_cost_ms=search_cost,
            children=[filter_step],
            metadata={'filtered_search': True}
        )
        
        root_step = search_step
        final_rows = query.top_k
        
        # Apply additional operations
        root_step = self._apply_additional_operations(root_step, query, final_rows)
        
        total_cost = self._calculate_total_cost(root_step)
        
        return ExecutionPlan(
            query=query,
            root_step=root_step,
            total_cost_ms=total_cost,
            total_rows=root_step.estimated_rows,
            uses_compression=True,
            is_approximate=False
        )
    
    def _build_search_first_plan(self, query: Query) -> ExecutionPlan:
        """Build search-first execution plan."""
        # Search step
        over_fetch_factor = 3
        search_cost = self._estimate_vector_search_cost(
            self.total_windows,
            query.top_k * over_fetch_factor
        )
        search_step = ExecutionStep(
            operator=OperatorType.VECTOR_SEARCH,
            description=f"Similarity search for '{query.similarity_ref}'",
            estimated_rows=query.top_k * over_fetch_factor,
            estimated_cost_ms=search_cost,
            metadata={'over_fetch': True}
        )
        
        # Filter step
        filter_cost = query.top_k * over_fetch_factor * self.costs['filter_per_row']
        filter_step = ExecutionStep(
            operator=OperatorType.FILTER,
            description=f"Apply SQL filters: {query.filters}",
            estimated_rows=query.top_k,
            estimated_cost_ms=filter_cost,
            children=[search_step]
        )
        
        root_step = filter_step
        final_rows = query.top_k
        
        # Apply additional operations
        root_step = self._apply_additional_operations(root_step, query, final_rows)
        
        total_cost = self._calculate_total_cost(root_step)
        
        return ExecutionPlan(
            query=query,
            root_step=root_step,
            total_cost_ms=total_cost,
            total_rows=root_step.estimated_rows,
            uses_compression=True,
            is_approximate=False
        )
    
    def _optimize_sql(self, query: Query) -> ExecutionPlan:
        """Optimize standard SQL query."""
        # Determine if we can use indexes
        has_index = self._can_use_index(query.filters)
        
        # Estimate rows after filtering
        estimated_rows = self._estimate_selectivity(query.filters)
        
        root_step = None
        
        if has_index:
            # Index lookup
            lookup_cost = self.costs['index_lookup'] + estimated_rows * self.costs['scan_per_row']
            root_step = ExecutionStep(
                operator=OperatorType.INDEX_LOOKUP,
                description=f"Index scan on {self._get_index_columns(query.filters)}",
                estimated_rows=estimated_rows,
                estimated_cost_ms=lookup_cost,
                metadata={'indexed': True}
            )
        else:
            # Full table scan
            scan_cost = self.total_windows * self.costs['scan_per_row']
            scan_step = ExecutionStep(
                operator=OperatorType.SQL_SCAN,
                description="Sequential scan on timeseries",
                estimated_rows=self.total_windows,
                estimated_cost_ms=scan_cost
            )
            
            if query.filters:
                filter_cost = self.total_windows * self.costs['filter_per_row']
                filter_step = ExecutionStep(
                    operator=OperatorType.FILTER,
                    description=f"Apply filters: {query.filters}",
                    estimated_rows=estimated_rows,
                    estimated_cost_ms=filter_cost,
                    children=[scan_step]
                )
                root_step = filter_step
            else:
                root_step = scan_step
        
        # Handle aggregations
        if query.aggregations:
            root_step, is_approximate = self._build_aggregation_plan(root_step, query)
        else:
            is_approximate = False
        
        # Apply additional operations
        root_step = self._apply_additional_operations(root_step, query, root_step.estimated_rows)
        
        total_cost = self._calculate_total_cost(root_step)
        
        return ExecutionPlan(
            query=query,
            root_step=root_step,
            total_cost_ms=total_cost,
            total_rows=root_step.estimated_rows,
            uses_compression=is_approximate,  # Uses compression only if approximate
            is_approximate=is_approximate
        )
    
    def _build_aggregation_plan(self, input_step: ExecutionStep, query: Query) -> Tuple[ExecutionStep, bool]:
        """Build aggregation sub-plan."""
        agg_cost = input_step.estimated_rows * self.costs['aggregate_per_row']
        distinct_aggs = len(set(query.aggregations))
        
        # Check if we can approximate on embeddings
        can_approximate = (
            self.enable_approximation and
            self._is_approximatable(query.aggregations)
        )
        
        if can_approximate:
            # Approximate aggregation on compressed embeddings
            agg_step = ExecutionStep(
                operator=OperatorType.AGGREGATE,
                description=f"Approximate aggregation on embeddings: {query.aggregations}",
                estimated_rows=distinct_aggs,
                estimated_cost_ms=agg_cost * 0.1,  # 10x faster
                children=[input_step],
                metadata={'approximate': True, 'aggregations': query.aggregations}
            )
            return agg_step, True
        else:
            # Need to decompress for exact aggregation
            decompress_cost = input_step.estimated_rows * self.costs['decompress_per_row']
            decompress_step = ExecutionStep(
                operator=OperatorType.DECOMPRESS,
                description="Decompress embeddings for exact aggregation",
                estimated_rows=input_step.estimated_rows,
                estimated_cost_ms=decompress_cost,
                children=[input_step]
            )
            
            agg_step = ExecutionStep(
                operator=OperatorType.AGGREGATE,
                description=f"Exact aggregation: {query.aggregations}",
                estimated_rows=distinct_aggs,
                estimated_cost_ms=agg_cost,
                children=[decompress_step],
                metadata={'approximate': False, 'aggregations': query.aggregations}
            )
            return agg_step, False
    
    def _apply_additional_operations(
        self, 
        root_step: ExecutionStep, 
        query: Query,
        current_rows: int
    ) -> ExecutionStep:
        """Apply limit, sort, and other operations to plan."""
        current_step = root_step
        
        # Apply sorting if needed
        if query.order_by:
            sort_cost = current_rows * self.costs['sort_per_row']
            sort_step = ExecutionStep(
                operator=OperatorType.SORT,
                description=f"Sort by {query.order_by}",
                estimated_rows=current_rows,
                estimated_cost_ms=sort_cost,
                children=[current_step]
            )
            current_step = sort_step
        
        # Apply limit if specified
        if query.limit and query.limit < current_rows:
            limit_step = ExecutionStep(
                operator=OperatorType.LIMIT,
                description=f"Limit to {query.limit} results",
                estimated_rows=query.limit,
                estimated_cost_ms=0.01,
                children=[current_step]
            )
            current_step = limit_step
            current_rows = query.limit
        
        # Project selected columns
        if query.columns and query.columns != ["*"]:
            project_cost = current_rows * self.costs['project_per_row']
            project_step = ExecutionStep(
                operator=OperatorType.PROJECT,
                description=f"Select columns: {', '.join(query.columns)}",
                estimated_rows=current_rows,
                estimated_cost_ms=project_cost,
                children=[current_step]
            )
            current_step = project_step
        
        return current_step
    
    def _calculate_total_cost(self, root_step: ExecutionStep) -> float:
        """Calculate total cost recursively."""
        if root_step.children:
            child_costs = sum(self._calculate_total_cost(c) for c in root_step.children)
            return root_step.estimated_cost_ms + child_costs
        return root_step.estimated_cost_ms
    
    def _estimate_vector_search_cost(self, n_vectors: int, k: int) -> float:
        """
        Estimate cost of HNSW vector search.
        
        HNSW has O(log n) complexity.
        """
        if n_vectors <= 0:
            return 0.0
        if k <= 0:
            return 0.0
            
        log_factor = math.log(max(n_vectors, 2), 2)
        return self.costs['vector_search_base'] + self.costs['vector_search_log'] * log_factor
    
    def _estimate_selectivity(self, filters: Optional[Dict]) -> int:
        """
        Estimate number of rows after applying filters.
        
        Simple heuristic:
        - symbol filter: 1/25 of data (25 symbols)
        - timestamp filter: depends on range
        - other filters: 0.1 selectivity
        """
        if not filters:
            return self.total_windows
        
        selectivity = 1.0
        
        for key, value in filters.items():
            if key == 'symbol':
                selectivity *= 1.0 / self.stats['total_symbols']
            elif key.startswith('timestamp'):
                # Assume timestamp range covers 30% of data
                selectivity *= 0.3
            else:
                selectivity *= self.stats['default_selectivity']
        
        estimated = int(self.total_windows * selectivity)
        return max(1, min(estimated, self.total_windows))
    
    def _can_use_index(self, filters: Optional[Dict]) -> bool:
        """Check if filters can use indexes."""
        if not filters:
            return False
        
        # We have indexes on: symbol, timestamp, symbol+timestamp
        indexed_columns = {'symbol', 'timestamp'}
        return any(key in indexed_columns for key in filters.keys())
    
    def _get_index_columns(self, filters: Optional[Dict]) -> str:
        """Get index column names."""
        if not filters:
            return ""
        
        indexed = [key for key in filters.keys() if key in {'symbol', 'timestamp'}]
        return ", ".join(sorted(indexed))
    
    def _is_approximatable(self, aggregations: List[str]) -> bool:
        """Check if aggregation can be approximated on embeddings."""
        if not aggregations:
            return False
        
        # Simple heuristic: COUNT can be approximated
        # SUM, AVG, etc. need exact values
        for agg in aggregations:
            agg_upper = agg.upper()
            if agg_upper not in {'COUNT', 'COUNT(*)', 'COUNT(1)'}:
                return False
        return True
    
    def _needs_decompression(self, query: Query) -> bool:
        """Check if query needs decompressed results."""
        # Always decompress for now
        # In production, could check if query needs original features
        return True


# ============================================================================
# SIMPLE QUERY PARSER FOR TESTING (if parser module is not available)
# ============================================================================

@dataclass
class SimpleQuery:
    """Simple query representation for testing."""
    type: QueryType
    similarity_ref: Optional[str] = None
    pattern_name: Optional[str] = None
    similarity_threshold: float = 0.8
    anomaly_threshold: float = 0.9
    top_k: int = 10
    filters: Optional[Dict] = None
    aggregations: Optional[List[str]] = None
    order_by: Optional[str] = None
    limit: Optional[int] = None
    columns: List[str] = None
    
    def __post_init__(self):
        if self.columns is None:
            self.columns = ["*"]


class SimpleQueryParser:
    """Simple parser for testing optimizer."""
    
    @staticmethod
    def parse(query_str: str) -> SimpleQuery:
        """Parse a simple query string."""
        query_str = query_str.strip().upper()
        
        # Default values
        query_type = QueryType.SQL
        similarity_ref = None
        pattern_name = None
        similarity_threshold = 0.8
        anomaly_threshold = 0.9
        top_k = 10
        filters = {}
        aggregations = []
        order_by = None
        limit = None
        columns = ["*"]
        
        # Parse SIMILAR_TO queries
        if "SIMILAR_TO" in query_str:
            query_type = QueryType.SIMILARITY
            # Extract reference
            import re
            match = re.search(r"SIMILAR_TO\('([^']+)'", query_str)
            if match:
                similarity_ref = match.group(1)
            
            # Extract threshold
            match = re.search(r"THRESHOLD=([0-9.]+)", query_str)
            if match:
                similarity_threshold = float(match.group(1))
        
        # Parse pattern queries
        elif "PATTERN" in query_str:
            query_type = QueryType.PATTERN
            import re
            match = re.search(r"PATTERN='([^']+)'", query_str)
            if match:
                pattern_name = match.group(1)
        
        # Parse anomaly queries
        elif "ANOMALY_SCORE" in query_str:
            query_type = QueryType.ANOMALY
            import re
            match = re.search(r"ANOMALY_SCORE\(\) > ([0-9.]+)", query_str)
            if match:
                anomaly_threshold = float(match.group(1))
        
        # Parse TOP
        import re
        match = re.search(r"TOP\s+(\d+)", query_str)
        if match:
            top_k = int(match.group(1))
            limit = top_k
        
        # Parse WHERE
        if "WHERE" in query_str:
            where_part = query_str.split("WHERE")[1].split("AND")[0].strip()
            if "SYMBOL =" in where_part:
                symbol_match = re.search(r"SYMBOL\s*=\s*'([^']+)'", where_part)
                if symbol_match:
                    filters['symbol'] = symbol_match.group(1)
        
        # Parse SELECT columns
        if query_str.startswith("SELECT"):
            cols_part = query_str.split("FROM")[0].replace("SELECT", "").strip()
            if cols_part != "*":
                columns = [col.strip() for col in cols_part.split(",")]
        
        return SimpleQuery(
            type=query_type,
            similarity_ref=similarity_ref,
            pattern_name=pattern_name,
            similarity_threshold=similarity_threshold,
            anomaly_threshold=anomaly_threshold,
            top_k=top_k,
            filters=filters,
            aggregations=aggregations,
            order_by=order_by,
            limit=limit,
            columns=columns
        )


# ============================================================================
# EXAMPLES & TESTS
# ============================================================================

def test_optimizer():
    """Test the query optimizer."""
    print("=" * 80)
    print("QueryOptimizer Unit Tests")
    print("=" * 80)
    
    # Create test queries directly
    optimizer = QueryOptimizer(total_windows=25000)
    
    # Test 1: Similarity search
    print("\nTest 1: Similarity search optimization")
    try:
        query = SimpleQuery(
            type=QueryType.SIMILARITY,
            similarity_ref="2020-03-16",
            similarity_threshold=0.85,
            top_k=10,
            columns=["*"]
        )
        plan = optimizer.optimize(query)
        print(plan.explain(verbose=False))
        assert plan.uses_compression
        print("✓ Similarity search plan generated")
    except Exception as e:
        print(f"✗ Similarity search failed: {e}")
    
    # Test 2: SQL with index
    print("\nTest 2: SQL with index optimization")
    try:
        query = SimpleQuery(
            type=QueryType.SQL,
            filters={'symbol': 'AAPL'},
            columns=["*"]
        )
        plan = optimizer.optimize(query)
        print(plan.explain(verbose=False))
        print("✓ SQL with index plan generated")
    except Exception as e:
        print(f"✗ SQL with index failed: {e}")
    
    # Test 3: Hybrid query
    print("\nTest 3: Hybrid query optimization")
    try:
        query = SimpleQuery(
            type=QueryType.HYBRID,
            similarity_ref="2020-03-16",
            filters={'symbol': 'AAPL'},
            top_k=5,
            columns=["*"]
        )
        plan = optimizer.optimize(query)
        print(plan.explain(verbose=False))
        assert plan.uses_compression
        print("✓ Hybrid plan generated")
    except Exception as e:
        print(f"✗ Hybrid query failed: {e}")
    
    # Test 4: Aggregation
    print("\nTest 4: Aggregation optimization")
    try:
        query = SimpleQuery(
            type=QueryType.SQL,
            aggregations=["COUNT(*)"],
            columns=["symbol", "COUNT(*)"]
        )
        plan = optimizer.optimize(query)
        print(plan.explain(verbose=False))
        print("✓ Aggregation plan generated")
    except Exception as e:
        print(f"✗ Aggregation failed: {e}")
    
    # Test 5: Anomaly detection
    print("\nTest 5: Anomaly detection optimization")
    try:
        query = SimpleQuery(
            type=QueryType.ANOMALY,
            anomaly_threshold=0.95,
            columns=["*"]
        )
        plan = optimizer.optimize(query)
        print(plan.explain(verbose=False))
        print("✓ Anomaly detection plan generated")
    except Exception as e:
        print(f"✗ Anomaly detection failed: {e}")
    
    print("\n" + "=" * 80)
    print("✓ All tests completed!")
    print("=" * 80)


def compare_plans():
    """Compare different execution strategies."""
    print("\n" + "=" * 80)
    print("Query Plan Comparison")
    print("=" * 80)
    
    optimizer = QueryOptimizer(total_windows=25000)
    
    queries = [
        ("Similarity Search", 
         SimpleQuery(type=QueryType.SIMILARITY, similarity_ref="2020-03-16", top_k=10, columns=["*"])),
        ("SQL Filter", 
         SimpleQuery(type=QueryType.SQL, filters={'symbol': 'AAPL'}, columns=["*"])),
        ("SQL Aggregation", 
         SimpleQuery(type=QueryType.SQL, aggregations=["COUNT(*)"], columns=["symbol", "COUNT(*)"])),
        ("Hybrid Query", 
         SimpleQuery(type=QueryType.HYBRID, similarity_ref="2020-03-16", 
                    filters={'symbol': 'AAPL'}, top_k=5, columns=["*"])),
    ]
    
    for name, query in queries:
        print(f"\n{name}:")
        print("-" * 80)
        try:
            plan = optimizer.optimize(query)
            print(f"Total Cost: {plan.total_cost_ms:.2f} ms")
            print(f"Estimated Rows: {plan.total_rows:,}")
            print(f"Uses Compression: {plan.uses_compression}")
            print(f"Is Approximate: {plan.is_approximate}")
        except Exception as e:
            print(f"Error: {e}")


def benchmark_optimizer():
    """Benchmark optimizer performance."""
    import time
    
    print("\n" + "=" * 80)
    print("Optimizer Benchmark")
    print("=" * 80)
    
    optimizer = QueryOptimizer(total_windows=1000000)  # Large dataset
    
    queries = [
        SimpleQuery(type=QueryType.SIMILARITY, similarity_ref="2020-03-16", top_k=10, columns=["*"]),
        SimpleQuery(type=QueryType.SQL, filters={'symbol': 'AAPL', 'timestamp': '>2020-01-01'}, columns=["*"]),
        SimpleQuery(type=QueryType.SQL, aggregations=["COUNT(*)"], columns=["symbol", "COUNT(*)"]),
        SimpleQuery(type=QueryType.HYBRID, similarity_ref="2020-03-16", 
                   filters={'symbol': 'AAPL'}, top_k=5, columns=["*"]),
    ]
    
    times = []
    
    for i, query in enumerate(queries, 1):
        start = time.perf_counter()
        plan = optimizer.optimize(query)
        end = time.perf_counter()
        
        times.append(end - start)
        print(f"Query {i}: {query.type.value}")
        print(f"  Optimization time: {(end - start) * 1000:.2f} ms")
        print(f"  Plan cost: {plan.total_cost_ms:.2f} ms")
        print()
    
    avg_time = sum(times) / len(times)
    print(f"Average optimization time: {avg_time * 1000:.2f} ms")


if __name__ == "__main__":
    # Run tests
    test_optimizer()
    
    # Show plan comparisons
    compare_plans()
    
    # Run benchmark
    benchmark_optimizer()