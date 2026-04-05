"""
FinanceDB - Query Parser
Parses SQL and semantic query strings into structured Query objects.

Author: FinanceDB Project
Module: 4 - Query Processing (Parser)
"""

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from enum import Enum

class QueryType(Enum):
    SQL = "sql"
    SIMILARITY = "similarity"
    PATTERN = "pattern"
    ANOMALY = "anomaly"
    HYBRID = "hybrid"

@dataclass
class Query:
    type: QueryType
    sql: Optional[str] = None
    filters: Optional[Dict[str, Any]] = None
    aggregations: Optional[List[str]] = None
    similarity_ref: Optional[str] = None
    similarity_threshold: float = 0.0
    top_k: int = 10
    pattern_name: Optional[str] = None
    anomaly_threshold: float = 0.95
    columns: Optional[List[str]] = None
    limit: Optional[int] = None
    order_by: Optional[Dict[str, str]] = None
    
    def __post_init__(self):
        """Set defaults for attributes."""
        if self.columns is None:
            self.columns = ["*"]
        if self.filters is None:
            self.filters = {}
        if self.aggregations is None:
            self.aggregations = []
        if self.order_by is None:
            self.order_by = {}

class QueryParser:
    def __init__(self):
        # Define regex patterns as strings
        self.patterns = {
            'similar_to': r"SIMILAR_TO\s*\(\s*['\"]([^'\"]+)['\"](?:\s*,\s*threshold\s*=\s*([\d.]+))?\s*\)",
            'pattern_match': r"PATTERN_MATCH\s*\(\s*['\"]([^'\"]+)['\"](?:\s*,\s*threshold\s*=\s*([\d.]+))?\s*\)",
            'anomaly_score': r"ANOMALY_SCORE\s*\(\s*\)\s*([><=]+)\s*([\d.]+)",
            'top_k': r"TOP\s+(\d+)"
        }

    def parse(self, query_string: str) -> Query:
        query_string = query_string.strip()
        
        # Use compiled patterns for searching
        similar_to_match = re.search(self.patterns['similar_to'], query_string, re.IGNORECASE)
        pattern_match_match = re.search(self.patterns['pattern_match'], query_string, re.IGNORECASE)
        anomaly_score_match = re.search(self.patterns['anomaly_score'], query_string, re.IGNORECASE)
        
        if similar_to_match:
            if self._has_sql_filters(query_string):
                return self._parse_hybrid(query_string)
            else:
                return self._parse_similarity(query_string)
        elif pattern_match_match:
            return self._parse_pattern(query_string)
        elif anomaly_score_match:
            return self._parse_anomaly(query_string)
        else:
            return self._parse_sql(query_string)

    def _has_sql_filters(self, query_string: str) -> bool:
        """
        Check if query contains SQL WHERE clause filters.
        
        Args:
            query_string: Raw query string
            
        Returns:
            True if contains SQL WHERE filters
        """
        # Remove similarity patterns to isolate SQL part
        cleaned = re.sub(self.patterns['similar_to'], '', query_string, flags=re.IGNORECASE)
        cleaned = re.sub(self.patterns['pattern_match'], '', cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(self.patterns['anomaly_score'], '', cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(self.patterns['top_k'], '', cleaned, flags=re.IGNORECASE)
        
        # Check for SQL WHERE keywords
        return bool(re.search(r'\bWHERE\b', cleaned, re.IGNORECASE))

    def _parse_sql(self, query_string: str) -> Query:
        """Parse standard SQL query."""
        return Query(
            type=QueryType.SQL,
            sql=query_string,
            filters=self._extract_sql_filters(query_string),
            aggregations=self._extract_aggregations(query_string),
            columns=self._extract_select_columns(query_string),
            limit=self._extract_limit(query_string),
            order_by=self._extract_order_by(query_string)
        )

    def _parse_similarity(self, query_string: str) -> Query:
        """Parse similarity search query."""
        match = re.search(self.patterns['similar_to'], query_string, re.IGNORECASE)
        if not match: 
            raise ValueError("Invalid SIMILAR_TO syntax")
        
        reference = match.group(1)
        threshold = float(match.group(2)) if match.group(2) else 0.0
        
        top_k = 10
        top_k_match = re.search(self.patterns['top_k'], query_string, re.IGNORECASE)
        if top_k_match: 
            top_k = int(top_k_match.group(1))
        
        # Try to extract columns if there's a SELECT clause
        columns = self._extract_select_columns(query_string)
        
        return Query(
            type=QueryType.SIMILARITY,
            similarity_ref=reference,
            similarity_threshold=threshold,
            top_k=top_k,
            columns=columns
        )

    def _parse_pattern(self, query_string: str) -> Query:
        """Parse pattern matching query."""
        match = re.search(self.patterns['pattern_match'], query_string, re.IGNORECASE)
        if not match: 
            raise ValueError("Invalid PATTERN_MATCH syntax")
        
        pattern_name = match.group(1)
        threshold = float(match.group(2)) if match.group(2) else 0.85
        
        top_k = 10
        top_k_match = re.search(self.patterns['top_k'], query_string, re.IGNORECASE)
        if top_k_match: 
            top_k = int(top_k_match.group(1))
        
        # Try to extract columns if there's a SELECT clause
        columns = self._extract_select_columns(query_string)
        
        return Query(
            type=QueryType.PATTERN,
            pattern_name=pattern_name,
            similarity_threshold=threshold,
            top_k=top_k,
            columns=columns
        )

    def _parse_anomaly(self, query_string: str) -> Query:
        """Parse anomaly detection query."""
        match = re.search(self.patterns['anomaly_score'], query_string, re.IGNORECASE)
        if not match: 
            raise ValueError("Invalid ANOMALY_SCORE syntax")
        
        operator = match.group(1)
        threshold = float(match.group(2))
        
        if operator not in ['>', '>=']:
            raise ValueError(f"Unsupported anomaly operator: {operator}")
        
        # Extract columns if there's a SELECT clause
        columns = self._extract_select_columns(query_string)
        
        return Query(
            type=QueryType.ANOMALY,
            anomaly_threshold=threshold,
            filters=self._extract_sql_filters(query_string),
            columns=columns
        )

    def _parse_hybrid(self, query_string: str) -> Query:
        """Parse hybrid SQL + similarity query."""
        match = re.search(self.patterns['similar_to'], query_string, re.IGNORECASE)
        if not match: 
            raise ValueError("Invalid SIMILAR_TO in hybrid query")
        
        reference = match.group(1)
        threshold = float(match.group(2)) if match.group(2) else 0.0
        
        # Clean the SQL part
        sql_part = re.sub(self.patterns['similar_to'], '', query_string, flags=re.IGNORECASE)
        sql_part = re.sub(self.patterns['top_k'], '', sql_part, flags=re.IGNORECASE)
        # Fix: Remove trailing 'AND' or 'OR' that might be left after SIMILAR_TO removal
        sql_part = re.sub(r'\s+(?:AND|OR)\s*$', '', sql_part.strip(), flags=re.IGNORECASE)
        
        # Extract columns from SQL part
        columns = self._extract_select_columns(sql_part)
        
        # Extract LIMIT from query_string (not sql_part) since LIMIT might be before SIMILAR_TO
        limit = self._extract_limit(query_string)
        
        # Extract ORDER BY from SQL part
        order_by = self._extract_order_by(sql_part)
        
        # Extract TOP K from query_string
        top_k = 10
        top_k_match = re.search(self.patterns['top_k'], query_string, re.IGNORECASE)
        if top_k_match: 
            top_k = int(top_k_match.group(1))
        
        return Query(
            type=QueryType.HYBRID,
            sql=sql_part,
            filters=self._extract_sql_filters(sql_part),
            aggregations=self._extract_aggregations(sql_part),
            similarity_ref=reference,
            similarity_threshold=threshold,
            top_k=top_k,
            columns=columns,
            limit=limit,
            order_by=order_by
        )

    def _extract_select_columns(self, query_string: str) -> List[str]:
        """Extract columns from SELECT clause."""
        # Match SELECT ... FROM pattern
        match = re.search(
            r'SELECT\s+(.+?)\s+FROM',
            query_string,
            re.IGNORECASE | re.DOTALL
        )
        
        if not match:
            # Check for SELECT without FROM (simple queries)
            match2 = re.search(
                r'SELECT\s+(.+?)(?:\s+WHERE|\s+ORDER|\s+LIMIT|$)',
                query_string,
                re.IGNORECASE | re.DOTALL
            )
            if not match2:
                return ["*"]
            columns_part = match2.group(1).strip()
        else:
            columns_part = match.group(1).strip()
        
        # Handle DISTINCT
        if columns_part.upper().startswith('DISTINCT'):
            columns_part = columns_part[8:].strip()
        
        # Handle TOP keyword
        if columns_part.upper().startswith('TOP'):
            # Remove TOP X
            columns_part = re.sub(r'TOP\s+\d+\s+', '', columns_part, flags=re.IGNORECASE)
        
        if columns_part.strip() == "*":
            return ["*"]
        
        # Split by comma, but not within parentheses
        columns = []
        current = ""
        paren_depth = 0
        
        for char in columns_part:
            if char == '(':
                paren_depth += 1
            elif char == ')':
                paren_depth -= 1
            
            if char == ',' and paren_depth == 0:
                col = current.strip()
                if col:
                    columns.append(col)
                current = ""
            else:
                current += char
        
        if current:
            col = current.strip()
            if col:
                columns.append(col)
        
        return columns if columns else ["*"]

    def _extract_sql_filters(self, query_string: str) -> Dict[str, Any]:
        filters = {}
        # Improved WHERE clause extraction
        where_match = re.search(
            r'\bWHERE\s+(.+?)(?:\s+(?:GROUP|ORDER|LIMIT|$))',
            query_string, 
            re.IGNORECASE | re.DOTALL
        )
        
        if not where_match: 
            return filters
        
        where_clause = where_match.group(1).strip()
        
        # Split by AND/OR but keep them for complex conditions
        # Simple extraction for basic comparisons
        # Pattern for column operator value
        pattern = re.compile(
            r'(\w+)\s*([=<>!]+)\s*(?:\'([^\']+)\'|"([^"]+)"|(\d+(?:\.\d+)?)|(\w+))',
            re.IGNORECASE
        )
        
        matches = pattern.finditer(where_clause)
        for match in matches:
            column = match.group(1)
            operator = match.group(2)
            # Get value from one of the capture groups
            value = match.group(3) or match.group(4) or match.group(5) or match.group(6)
            
            # Try to convert numeric values
            if value and value.replace('.', '').isdigit():
                if '.' in value:
                    value = float(value)
                else:
                    value = int(value)
            
            # Store filter
            if operator == '=':
                filters[column] = value
            else:
                # For other operators, store with operator suffix
                filters[f"{column}__{operator}"] = value
        
        return filters

    def _extract_aggregations(self, query_string: str) -> List[str]:
        group_by_match = re.search(
            r'\bGROUP\s+BY\s+(.+?)(?:\s+(?:ORDER|LIMIT|$))',
            query_string, 
            re.IGNORECASE | re.DOTALL
        )
        if group_by_match:
            return [col.strip() for col in group_by_match.group(1).split(',')]
        return []

    def _extract_limit(self, query_string: str) -> Optional[int]:
        """Extract LIMIT clause."""
        match = re.search(
            r'LIMIT\s+(\d+)',
            query_string,
            re.IGNORECASE
        )
        return int(match.group(1)) if match else None

    def _extract_order_by(self, query_string: str) -> Dict[str, str]:
        """Extract ORDER BY clause."""
        match = re.search(
            r'ORDER\s+BY\s+(.+?)(?:\s+(?:LIMIT|$))',
            query_string,
            re.IGNORECASE | re.DOTALL
        )
        
        if not match:
            return {}
        
        order_clause = match.group(1).strip()
        order_by = {}
        
        # Split by comma for multiple columns
        for part in order_clause.split(','):
            part = part.strip()
            if not part:
                continue
            
            # Check for ASC/DESC
            if part.upper().endswith(' ASC'):
                column = part[:-4].strip()
                order_by[column] = 'ASC'
            elif part.upper().endswith(' DESC'):
                column = part[:-5].strip()
                order_by[column] = 'DESC'
            else:
                # Default to ASC
                order_by[part] = 'ASC'
        
        return order_by


# Run unit tests
if __name__ == "__main__":
    parser = QueryParser()
    
    print("Testing Query Parser...")
    print("=" * 50)
    
    # Test 1: Standard SQL
    try:
        query = parser.parse("SELECT * FROM timeseries WHERE symbol = 'AAPL'")
        print(f"✓ Test 1 - SQL Query: type={query.type}, filters={query.filters}, columns={query.columns}")
        assert query.filters == {'symbol': 'AAPL'}
        assert query.columns == ["*"]
    except Exception as e:
        print(f"✗ Test 1 failed: {e}")
    
    # Test 2: Similarity Query
    try:
        query = parser.parse("SELECT TOP 5 SIMILAR_TO('2020-03-16')")
        print(f"✓ Test 2 - Similarity Query: type={query.type}, ref={query.similarity_ref}, top_k={query.top_k}, columns={query.columns}")
        assert query.type == QueryType.SIMILARITY
        assert query.similarity_ref == '2020-03-16'
        assert query.top_k == 5
        assert query.columns == ["*"]
    except Exception as e:
        print(f"✗ Test 2 failed: {e}")
    
    # Test 3: Hybrid Query
    try:
        query = parser.parse("SELECT TOP 3 * WHERE symbol = 'MSFT' AND SIMILAR_TO('crash_pattern')")
        print(f"✓ Test 3 - Hybrid Query: type={query.type}, ref={query.similarity_ref}, filters={query.filters}, columns={query.columns}, limit={query.limit}")
        assert query.type == QueryType.HYBRID
        assert query.similarity_ref == 'crash_pattern'
        assert query.filters['symbol'] == 'MSFT'
        assert query.columns == ["*"]
    except Exception as e:
        print(f"✗ Test 3 failed: {e}")
    
    # Test 4: SQL with columns
    try:
        query = parser.parse("SELECT symbol, price, volume FROM timeseries WHERE symbol = 'AAPL'")
        print(f"✓ Test 4 - SQL with columns: columns={query.columns}")
        assert query.columns == ["symbol", "price", "volume"]
    except Exception as e:
        print(f"✗ Test 4 failed: {e}")
    
    print("\n" + "=" * 50)
    print("Parser tests completed!")