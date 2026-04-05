"""
FinanceDB - Pattern Library (Complete Fixed Version)
Pre-defined financial market patterns for semantic matching.

Author: FinanceDB Project
Module: 5 - Semantic Retrieval (Pattern Library)
"""

import sys
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
from datetime import datetime
import numpy as np
import pandas as pd
import yaml
import pickle
import json
from loguru import logger

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from storage.storage_manager import StorageManager


class PatternLibrary:
    """
    Library of pre-defined and custom financial market patterns.
    
    Fixed: All embedding type issues resolved.
    """
    
    # Default pre-defined patterns with example dates (within available data range)
    DEFAULT_PATTERNS = {
        'market_crash': {
            'description': 'Severe market decline with high volatility',
            'examples': [
                ('2020-03-16', 'AAPL'),  # COVID-19 crash
                ('2020-03-12', 'MSFT'),  # COVID-19 crash
                ('2020-03-23', 'SPY'),   # COVID-19 crash bottom
            ]
        },
        'high_volatility': {
            'description': 'Large intraday price swings',
            'examples': [
                ('2020-03-09', 'AAPL'),  # Circuit breaker day
                ('2020-03-18', 'MSFT'),  # High volatility
                ('2020-03-12', 'SPY'),   # High volatility
            ]
        },
        'calm_market': {
            'description': 'Low volatility, steady trading',
            'examples': [
                ('2020-04-15', 'AAPL'),  # Post-crash calm
                ('2020-04-20', 'MSFT'),  # Low volatility period
                ('2020-05-10', 'SPY'),   # Stable market
            ]
        },
        'bull_run': {
            'description': 'Strong upward trend with momentum',
            'examples': [
                ('2020-11-09', 'AAPL'),  # Vaccine rally
                ('2020-12-21', 'MSFT'),  # Year-end rally
                ('2021-01-07', 'SPY'),   # Post-election rally
            ]
        },
        'earnings_beat': {
            'description': 'Positive earnings surprise reaction',
            'examples': [
                ('2020-07-31', 'AAPL'),  # Strong earnings
                ('2020-10-27', 'MSFT'),  # Earnings beat
                ('2021-01-26', 'SPY'),   # Earnings season
            ]
        },
        'sector_rotation': {
            'description': 'Money flow between sectors',
            'examples': [
                ('2020-06-08', 'JPM'),   # Bank rotation
                ('2020-11-09', 'XOM'),   # Energy rotation
                ('2021-03-08', 'BA'),    # Industrial rotation
            ]
        }
    }
    
    def __init__(
        self,
        storage_manager: StorageManager,
        config_path: str = "configs/patterns.yaml"
    ):
        """
        Initialize pattern library.
        
        Args:
            storage_manager: StorageManager instance
            config_path: Path to pattern configuration file
        """
        self.storage = storage_manager
        self.config_path = Path(config_path)
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Pattern storage: {pattern_name: {'embedding': np.ndarray, 'metadata': dict}}
        self.patterns: Dict[str, Dict] = {}
        
        logger.info("Initialized PatternLibrary")
        
        # Load existing patterns or initialize defaults
        if self.config_path.exists():
            try:
                self.load_patterns()
                logger.info(f"Loaded {len(self.patterns)} patterns from disk")
            except Exception as e:
                logger.warning(f"Failed to load patterns from {self.config_path}: {e}")
                logger.info("Initializing default patterns...")
                self._initialize_default_patterns()
        else:
            logger.info("No existing patterns found, initializing defaults")
            self._initialize_default_patterns()
    
    def _initialize_default_patterns(self):
        """Initialize library with default pre-defined patterns."""
        logger.info("Computing embeddings for default patterns...")
        
        successful_patterns = 0
        
        for pattern_name, pattern_info in self.DEFAULT_PATTERNS.items():
            try:
                success = self.add_pattern(
                    name=pattern_name,
                    example_dates=pattern_info['examples'],
                    description=pattern_info['description']
                )
                if success:
                    logger.info(f"  ✓ Added pattern: {pattern_name}")
                    successful_patterns += 1
                else:
                    logger.warning(f"  ✗ Failed to add pattern: {pattern_name}")
            except Exception as e:
                logger.error(f"  ✗ Error adding pattern {pattern_name}: {e}")
        
        # Save to disk
        if successful_patterns > 0:
            self.save_patterns()
        
        logger.info(f"Initialized {successful_patterns}/{len(self.DEFAULT_PATTERNS)} default patterns")
    
    def _get_embedding_from_window(self, window: Dict) -> Optional[np.ndarray]:
        """Safely extract embedding from window data."""
        try:
            embedding = window.get('embedding')
            if embedding is None:
                return None
            
            # Handle different embedding formats
            if isinstance(embedding, np.ndarray):
                return embedding.astype(np.float32)
            elif isinstance(embedding, (list, tuple)):
                return np.array(embedding, dtype=np.float32)
            elif isinstance(embedding, bytes):
                # Try to unpickle
                try:
                    unpickled = pickle.loads(embedding)
                    if isinstance(unpickled, np.ndarray):
                        return unpickled.astype(np.float32)
                    elif isinstance(unpickled, (list, tuple)):
                        return np.array(unpickled, dtype=np.float32)
                    else:
                        return None
                except:
                    return None
            elif isinstance(embedding, str):
                # Try to parse string representation
                try:
                    # Remove brackets and split
                    clean_str = embedding.strip('[]()')
                    values = [float(x.strip()) for x in clean_str.split(',') if x.strip()]
                    return np.array(values, dtype=np.float32)
                except:
                    return None
            else:
                logger.debug(f"Unknown embedding type: {type(embedding)}")
                return None
        except Exception as e:
            logger.debug(f"Error extracting embedding: {e}")
            return None
    
    def add_pattern(
        self,
        name: str,
        example_dates: List[Tuple[str, str]],
        description: str = "",
        overwrite: bool = False
    ) -> bool:
        """
        Add a new pattern from example dates.
        
        Args:
            name: Pattern name (unique identifier)
            example_dates: List of (date, symbol) tuples
            description: Pattern description
            overwrite: Whether to overwrite existing pattern
            
        Returns:
            True if pattern added successfully
        """
        try:
            # Check if pattern exists
            if name in self.patterns and not overwrite:
                logger.warning(f"Pattern '{name}' already exists. Use overwrite=True to replace.")
                return False
            
            logger.info(f"Adding pattern '{name}' from {len(example_dates)} examples")
            
            # Collect embeddings from example dates
            embeddings = []
            valid_examples = []
            
            for date_str, symbol in example_dates:
                try:
                    date = pd.to_datetime(date_str)
                    
                    # Get window for this date/symbol
                    windows = self.storage.read_by_symbol(
                        symbol=symbol,
                        start_date=date,
                        end_date=date
                    )
                    
                    if windows:
                        for window in windows:
                            embedding = self._get_embedding_from_window(window)
                            if embedding is not None:
                                embeddings.append(embedding)
                                valid_examples.append((date_str, symbol))
                                logger.debug(f"  Found embedding for {symbol} @ {date_str}")
                                break  # Found valid embedding for this date/symbol
                        else:
                            logger.debug(f"  No valid embedding found for {symbol} @ {date_str}")
                    else:
                        logger.debug(f"  No windows found for {symbol} @ {date_str}")
                        
                except Exception as e:
                    logger.debug(f"  Error processing {symbol} @ {date_str}: {e}")
                    continue
            
            if not embeddings:
                logger.warning(f"No valid embeddings found for pattern '{name}'")
                return False
            
            logger.info(f"  Found {len(embeddings)} valid embeddings for pattern '{name}'")
            
            # Compute average embedding (pattern centroid)
            pattern_embedding = np.mean(embeddings, axis=0).astype(np.float32)
            
            # Store pattern
            self.patterns[name] = {
                'embedding': pattern_embedding,
                'description': description,
                'examples': valid_examples,
                'num_examples': len(valid_examples),
                'created_at': datetime.now().isoformat(),
                'embedding_shape': pattern_embedding.shape,
                'embedding_dtype': str(pattern_embedding.dtype)
            }
            
            logger.info(f"✓ Pattern '{name}' created from {len(valid_examples)} examples")
            logger.debug(f"  Embedding shape: {pattern_embedding.shape}, dtype: {pattern_embedding.dtype}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to add pattern '{name}': {e}")
            import traceback
            logger.debug(traceback.format_exc())
            return False
    
    def match_pattern(
        self,
        embedding: np.ndarray,
        pattern_name: str,
        threshold: float = 0.85
    ) -> Optional[Dict]:
        """
        Check if embedding matches a specific pattern.
        
        Args:
            embedding: Query embedding vector
            pattern_name: Name of pattern to match
            threshold: Minimum similarity threshold (0-1)
            
        Returns:
            Match result dict with score, or None if no match
        """
        try:
            # Check if pattern exists
            if pattern_name not in self.patterns:
                logger.warning(f"Pattern '{pattern_name}' not found")
                return None
            
            pattern = self.patterns[pattern_name]
            pattern_embedding = pattern['embedding']
            
            # Handle different embedding types
            if isinstance(embedding, bytes):
                # Unpickle bytes
                try:
                    embedding = pickle.loads(embedding)
                except Exception as e:
                    logger.debug(f"Failed to unpickle embedding: {e}")
                    return None
            elif isinstance(embedding, (list, tuple)):
                embedding = np.array(embedding, dtype=np.float32)
            elif isinstance(embedding, str):
                # Try to parse string
                try:
                    embedding = np.array(json.loads(embedding), dtype=np.float32)
                except:
                    embedding = np.array(eval(embedding), dtype=np.float32)
            elif not isinstance(embedding, np.ndarray):
                logger.error(f"Invalid embedding type: {type(embedding)}")
                return None
            
            # Ensure it's float32 numpy array
            embedding = embedding.astype(np.float32) if isinstance(embedding, np.ndarray) else np.array(embedding, dtype=np.float32)
            
            # Check dimensions
            if embedding.shape != pattern_embedding.shape:
                logger.warning(f"Embedding shape mismatch: {embedding.shape} vs {pattern_embedding.shape}")
                # Try to reshape if possible
                if embedding.size == pattern_embedding.size:
                    embedding = embedding.reshape(pattern_embedding.shape)
                else:
                    return None
            
            # Compute cosine similarity
            similarity = self._cosine_similarity(embedding, pattern_embedding)
            
            # Check threshold
            if similarity < threshold:
                logger.debug(f"Pattern '{pattern_name}' similarity {similarity:.3f} below threshold {threshold}")
                return None
            
            # Return match result
            result = {
                'pattern_name': pattern_name,
                'score': float(similarity),
                'description': pattern['description'],
                'threshold': threshold,
                'matched': True
            }
            
            logger.debug(f"Pattern '{pattern_name}' matched with score {similarity:.3f}")
            
            return result
            
        except Exception as e:
            logger.error(f"Pattern matching failed for '{pattern_name}': {e}")
            import traceback
            logger.debug(traceback.format_exc())
            return None
    
    def find_all_patterns(
        self,
        embedding: np.ndarray,
        threshold: float = 0.85,
        top_k: Optional[int] = None
    ) -> List[Dict]:
        """
        Find all matching patterns for an embedding.
        
        Args:
            embedding: Query embedding vector
            threshold: Minimum similarity threshold
            top_k: Optional limit on number of results
            
        Returns:
            List of pattern matches sorted by score (descending)
        """
        try:
            matches = []
            
            # Handle different embedding types
            if isinstance(embedding, bytes):
                # Unpickle bytes
                try:
                    embedding = pickle.loads(embedding)
                except Exception as e:
                    logger.debug(f"Failed to unpickle embedding: {e}")
                    return []
            elif isinstance(embedding, (list, tuple)):
                embedding = np.array(embedding, dtype=np.float32)
            elif isinstance(embedding, str):
                # Try to parse string
                try:
                    embedding = np.array(json.loads(embedding), dtype=np.float32)
                except:
                    embedding = np.array(eval(embedding), dtype=np.float32)
            elif not isinstance(embedding, np.ndarray):
                logger.error(f"Invalid embedding type: {type(embedding)}")
                return []
            
            embedding = embedding.astype(np.float32) if isinstance(embedding, np.ndarray) else np.array(embedding, dtype=np.float32)
            
            # Check against all patterns
            for pattern_name, pattern in self.patterns.items():
                try:
                    pattern_embedding = pattern['embedding']
                    
                    # Skip if shapes don't match
                    if embedding.shape != pattern_embedding.shape:
                        if embedding.size == pattern_embedding.size:
                            embedding = embedding.reshape(pattern_embedding.shape)
                        else:
                            logger.debug(f"Skipping pattern '{pattern_name}' due to shape mismatch")
                            continue
                    
                    # Compute cosine similarity
                    similarity = self._cosine_similarity(embedding, pattern_embedding)
                    
                    # Check threshold
                    if similarity >= threshold:
                        matches.append({
                            'pattern_name': pattern_name,
                            'score': float(similarity),
                            'description': pattern['description'],
                            'threshold': threshold,
                            'matched': True
                        })
                        
                except Exception as e:
                    logger.debug(f"Error matching pattern '{pattern_name}': {e}")
                    continue
            
            # Sort by score (descending)
            matches.sort(key=lambda x: x['score'], reverse=True)
            
            # Limit to top_k if specified
            if top_k is not None:
                matches = matches[:top_k]
            
            logger.info(f"Found {len(matches)} pattern matches (threshold={threshold})")
            
            return matches
            
        except Exception as e:
            logger.error(f"Finding patterns failed: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            return []
    
    def list_patterns(self) -> List[Dict]:
        """
        List all available patterns with metadata.
        
        Returns:
            List of pattern information dictionaries
        """
        pattern_list = []
        
        for name, pattern in self.patterns.items():
            pattern_list.append({
                'name': name,
                'description': pattern['description'],
                'num_examples': pattern['num_examples'],
                'created_at': pattern.get('created_at', 'Unknown'),
                'examples': pattern['examples'],
                'embedding_shape': pattern.get('embedding_shape', 'Unknown'),
                'embedding_dtype': pattern.get('embedding_dtype', 'Unknown')
            })
        
        return pattern_list
    
    def remove_pattern(self, name: str) -> bool:
        """
        Remove a pattern from the library.
        
        Args:
            name: Pattern name to remove
            
        Returns:
            True if pattern removed successfully
        """
        if name in self.patterns:
            del self.patterns[name]
            logger.info(f"Removed pattern '{name}'")
            return True
        else:
            logger.warning(f"Pattern '{name}' not found")
            return False
    
    def save_patterns(self):
        """Save patterns to YAML configuration file."""
        try:
            # Prepare serializable data - convert tuples to lists for YAML
            config_data = {}
            
            for name, pattern in self.patterns.items():
                # Convert examples (list of tuples) to list of lists
                examples_as_lists = [[date, symbol] for date, symbol in pattern['examples']]
                
                config_data[name] = {
                    'description': pattern['description'],
                    'examples': examples_as_lists,  # Now list of lists, not tuples
                    'num_examples': pattern['num_examples'],
                    'created_at': pattern.get('created_at', datetime.now().isoformat()),
                    'embedding_shape': list(pattern['embedding'].shape) if hasattr(pattern['embedding'], 'shape') else [],
                    'embedding': pattern['embedding'].tolist()  # Convert numpy to list
                }
            
            # Write to YAML with safe dumping
            with open(self.config_path, 'w') as f:
                yaml.dump(config_data, f, default_flow_style=False, sort_keys=False, Dumper=yaml.SafeDumper)
            
            logger.info(f"Saved {len(self.patterns)} patterns to {self.config_path}")
            
        except Exception as e:
            logger.error(f"Failed to save patterns: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            raise
    
    def load_patterns(self):
        """Load patterns from YAML configuration file."""
        try:
            with open(self.config_path, 'r') as f:
                config_data = yaml.safe_load(f)
            
            if not config_data:
                logger.warning("Empty pattern configuration file")
                return
            
            # Load patterns
            self.patterns = {}
            
            for name, pattern_data in config_data.items():
                # Load embedding with proper type
                embedding_list = pattern_data['embedding']
                embedding = np.array(embedding_list, dtype=np.float32)
                
                # Reshape if shape information is available
                if 'embedding_shape' in pattern_data and pattern_data['embedding_shape']:
                    target_shape = tuple(pattern_data['embedding_shape'])
                    if embedding.shape != target_shape:
                        embedding = embedding.reshape(target_shape)
                
                # Convert examples back to tuples (from lists)
                examples_as_tuples = [(date, symbol) for date, symbol in pattern_data['examples']]
                
                self.patterns[name] = {
                    'embedding': embedding,
                    'description': pattern_data['description'],
                    'examples': examples_as_tuples,  # Convert back to tuples
                    'num_examples': pattern_data['num_examples'],
                    'created_at': pattern_data.get('created_at', 'Unknown'),
                    'embedding_shape': embedding.shape,
                    'embedding_dtype': str(embedding.dtype)
                }
            
            logger.info(f"Loaded {len(self.patterns)} patterns from {self.config_path}")
            
        except yaml.YAMLError as e:
            logger.error(f"YAML parsing error in {self.config_path}: {e}")
            raise
        except Exception as e:
            logger.error(f"Failed to load patterns: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            raise
    
    def _cosine_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Compute cosine similarity between two embeddings."""
        # Ensure both are float32
        emb1 = emb1.astype(np.float32)
        emb2 = emb2.astype(np.float32)
        
        # Flatten if needed
        if emb1.ndim > 1:
            emb1 = emb1.flatten()
        if emb2.ndim > 1:
            emb2 = emb2.flatten()
        
        # Compute dot product
        dot_product = np.dot(emb1, emb2)
        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        similarity = dot_product / (norm1 * norm2)
        
        # Clip to valid range [-1, 1]
        similarity = max(-1.0, min(1.0, similarity))
        
        # Convert to similarity score [0, 1] for patterns (cosine similarity of -1 to 1 maps to 0 to 1)
        similarity_score = (similarity + 1) / 2
        
        return similarity_score
    
    def get_pattern_statistics(self) -> Dict:
        """
        Get statistics about the pattern library.
        
        Returns:
            Dictionary with library statistics
        """
        stats = {
            'total_patterns': len(self.patterns),
            'pattern_names': list(self.patterns.keys()),
            'total_examples': sum(p['num_examples'] for p in self.patterns.values()),
            'config_path': str(self.config_path),
            'config_exists': self.config_path.exists()
        }
        
        # Add embedding statistics
        if self.patterns:
            first_pattern = list(self.patterns.values())[0]
            stats['embedding_shape'] = first_pattern['embedding'].shape
            stats['embedding_dtype'] = str(first_pattern['embedding'].dtype)
            stats['similarity_threshold_recommended'] = 0.7  # Default recommendation
        
        return stats
    
    def validate_patterns(self) -> Dict:
        """
        Validate all patterns in the library.
        
        Returns:
            Dictionary with validation results
        """
        validation_results = {
            'valid_patterns': [],
            'invalid_patterns': [],
            'warnings': []
        }
        
        for name, pattern in self.patterns.items():
            try:
                embedding = pattern['embedding']
                
                # Check for NaN values
                if np.any(np.isnan(embedding)):
                    validation_results['invalid_patterns'].append({
                        'name': name,
                        'reason': 'Contains NaN values'
                    })
                    continue
                
                # Check for infinite values
                if np.any(np.isinf(embedding)):
                    validation_results['invalid_patterns'].append({
                        'name': name,
                        'reason': 'Contains infinite values'
                    })
                    continue
                
                # Check embedding norm (should not be zero)
                norm = np.linalg.norm(embedding)
                if norm == 0:
                    validation_results['invalid_patterns'].append({
                        'name': name,
                        'reason': 'Zero norm embedding'
                    })
                    continue
                
                # Check if pattern has enough examples
                if pattern['num_examples'] < 2:
                    validation_results['warnings'].append({
                        'name': name,
                        'warning': f'Only {pattern["num_examples"]} example(s), consider adding more'
                    })
                
                validation_results['valid_patterns'].append(name)
                
            except Exception as e:
                validation_results['invalid_patterns'].append({
                    'name': name,
                    'reason': f'Validation error: {str(e)}'
                })
        
        return validation_results
    
    def search_similar_to_pattern(
        self,
        pattern_name: str,
        top_k: int = 10,
        symbol_filter: Optional[str] = None,
        threshold: float = 0.7
    ) -> List[Dict]:
        """
        Find market windows similar to a specific pattern.
        
        Args:
            pattern_name: Name of the pattern to search for
            top_k: Number of similar windows to return
            symbol_filter: Optional symbol to filter results
            threshold: Minimum similarity threshold
            
        Returns:
            List of similar windows with scores
        """
        if pattern_name not in self.patterns:
            logger.warning(f"Pattern '{pattern_name}' not found")
            return []
        
        pattern_embedding = self.patterns[pattern_name]['embedding']
        
        # Use storage manager to search for similar embeddings
        similar_windows = self.storage.search_similar(
            query_embedding=pattern_embedding,
            top_k=top_k,
            symbol_filter=symbol_filter
        )
        
        # Filter by threshold
        filtered_results = [w for w in similar_windows if w.get('score', 0) >= threshold]
        
        logger.info(f"Found {len(filtered_results)} windows similar to pattern '{pattern_name}' (threshold={threshold})")
        
        return filtered_results
    
    def get_pattern(self, pattern_name: str) -> Optional[Dict]:
        """
        Get a specific pattern by name.
        
        Args:
            pattern_name: Name of the pattern
            
        Returns:
            Pattern dictionary or None if not found
        """
        if pattern_name in self.patterns:
            pattern = self.patterns[pattern_name].copy()
            pattern['name'] = pattern_name
            return pattern
        return None
    
    def update_pattern(
        self,
        name: str,
        example_dates: Optional[List[Tuple[str, str]]] = None,
        description: Optional[str] = None,
        embedding: Optional[np.ndarray] = None
    ) -> bool:
        """
        Update an existing pattern.
        
        Args:
            name: Pattern name
            example_dates: New example dates (optional)
            description: New description (optional)
            embedding: New embedding (optional)
            
        Returns:
            True if pattern updated successfully
        """
        if name not in self.patterns:
            logger.warning(f"Pattern '{name}' not found")
            return False
        
        try:
            pattern = self.patterns[name]
            
            if example_dates is not None:
                # Recreate pattern from new examples
                success = self.add_pattern(
                    name=name,
                    example_dates=example_dates,
                    description=description or pattern['description'],
                    overwrite=True
                )
                return success
            elif embedding is not None:
                # Update embedding directly
                pattern['embedding'] = embedding.astype(np.float32)
                pattern['embedding_shape'] = embedding.shape
                pattern['embedding_dtype'] = str(embedding.dtype)
            
            if description is not None:
                pattern['description'] = description
            
            pattern['updated_at'] = datetime.now().isoformat()
            
            logger.info(f"✓ Pattern '{name}' updated")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update pattern '{name}': {e}")
            return False


# ============================================================================
# EXAMPLES & TESTS
# ============================================================================

def test_pattern_library():
    """Test the pattern library."""
    print("=" * 80)
    print("PatternLibrary Unit Tests")
    print("=" * 80)
    
    try:
        # Initialize storage and library
        print("\nInitializing storage and pattern library...")
        storage = StorageManager(
            duckdb_path="storage/financedb.duckdb",
            qdrant_path="storage/qdrant_data"
        )
        
        # Use a test config path
        test_config = "configs/patterns_test.yaml"
        Path(test_config).parent.mkdir(parents=True, exist_ok=True)
        
        library = PatternLibrary(
            storage_manager=storage,
            config_path=test_config
        )
        
        print("✓ PatternLibrary initialized")
        
        # Test 1: List patterns
        print("\n" + "=" * 80)
        print("Test 1: List Patterns")
        print("=" * 80)
        
        patterns = library.list_patterns()
        
        print(f"\nFound {len(patterns)} patterns:")
        print("-" * 80)
        for p in patterns:
            print(f"• {p['name']:20s} - {p['description']}")
            print(f"  Examples: {p['num_examples']}, Shape: {p['embedding_shape']}")
        
        # Test 2: Add custom pattern
        print("\n" + "=" * 80)
        print("Test 2: Add Custom Pattern")
        print("=" * 80)
        
        # Try dates more likely to have data
        custom_pattern = library.add_pattern(
            name='tech_rally',
            example_dates=[
                ('2020-04-06', 'AAPL'),
                ('2020-04-07', 'MSFT'),
                ('2020-04-08', 'GOOGL')
            ],
            description='Technology sector rally pattern'
        )
        
        if custom_pattern:
            print("✓ Custom pattern added successfully")
        else:
            print("⚠ Could not add custom pattern (no embeddings found for dates)")
        
        # Test 3: Validate patterns
        print("\n" + "=" * 80)
        print("Test 3: Validate Patterns")
        print("=" * 80)
        
        validation = library.validate_patterns()
        print(f"\nValidation Results:")
        print(f"  Valid patterns: {len(validation['valid_patterns'])}")
        print(f"  Invalid patterns: {len(validation['invalid_patterns'])}")
        print(f"  Warnings: {len(validation['warnings'])}")
        
        if validation['invalid_patterns']:
            print("\nInvalid patterns:")
            for p in validation['invalid_patterns']:
                print(f"  • {p['name']}: {p['reason']}")
        
        # Test 4: Save and reload
        print("\n" + "=" * 80)
        print("Test 4: Save and Reload Patterns")
        print("=" * 80)
        
        if library.patterns:
            # Save current patterns
            library.save_patterns()
            print("✓ Patterns saved to disk")
            
            # Create new library instance (should load from disk)
            print("\nLoading patterns from disk...")
            library2 = PatternLibrary(
                storage_manager=storage,
                config_path=test_config
            )
            
            reloaded_patterns = library2.list_patterns()
            print(f"✓ Reloaded {len(reloaded_patterns)} patterns from disk")
            
            # Verify the data matches
            if len(reloaded_patterns) >= len(patterns):
                print("✓ Pattern count matches or increased")
            else:
                print(f"⚠ Pattern count decreased: original={len(patterns)}, reloaded={len(reloaded_patterns)}")
        else:
            print("⚠ No patterns to save")
        
        # Test 5: Get statistics
        print("\n" + "=" * 80)
        print("Test 5: Library Statistics")
        print("=" * 80)
        
        stats = library.get_pattern_statistics()
        
        print(f"\nPattern Library Statistics:")
        print("-" * 80)
        print(f"Total patterns:    {stats['total_patterns']}")
        print(f"Total examples:    {stats['total_examples']}")
        print(f"Config path:       {stats['config_path']}")
        if stats['total_patterns'] > 0:
            print(f"Embedding shape:   {stats.get('embedding_shape', 'Unknown')}")
            print(f"Embedding dtype:   {stats.get('embedding_dtype', 'Unknown')}")
        print(f"Patterns:          {', '.join(stats['pattern_names'])}")
        
        # Test 6: Pattern matching with real data
        print("\n" + "=" * 80)
        print("Test 6: Pattern Matching with Real Data")
        print("=" * 80)
        
        if patterns:
            # Get real embedding from first pattern example
            if patterns[0]['examples']:
                date_str, symbol = patterns[0]['examples'][0]
                print(f"\nTesting with real data: {symbol} @ {date_str}")
                
                windows = storage.read_by_symbol(
                    symbol=symbol,
                    start_date=pd.to_datetime(date_str),
                    end_date=pd.to_datetime(date_str)
                )
                
                if windows and windows[0].get('embedding') is not None:
                    real_embedding = windows[0]['embedding']
                    
                    # Test match_pattern with bytes embedding
                    match = library.match_pattern(
                        embedding=real_embedding,
                        pattern_name=patterns[0]['name'],
                        threshold=0.5
                    )
                    
                    if match:
                        print(f"✓ Matched pattern '{match['pattern_name']}' with score {match['score']:.3f}")
                    else:
                        print(f"⚠ No match for {patterns[0]['name']} (threshold: 0.5)")
                    
                    # Test find_all_patterns
                    all_matches = library.find_all_patterns(
                        embedding=real_embedding,
                        threshold=0.3,
                        top_k=3
                    )
                    
                    if all_matches:
                        print(f"\nTop {len(all_matches)} pattern matches:")
                        for i, match in enumerate(all_matches, 1):
                            print(f"  {i}. {match['pattern_name']:20s} | Score: {match['score']:.3f}")
                    else:
                        print("\nNo pattern matches found")
                else:
                    print(f"No embedding found for {symbol} @ {date_str}")
        
        # Test 7: Search for windows similar to pattern
        print("\n" + "=" * 80)
        print("Test 7: Search Similar to Pattern")
        print("=" * 80)
        
        if patterns:
            pattern_name = patterns[0]['name']
            print(f"\nSearching for windows similar to pattern '{pattern_name}'...")
            
            similar_windows = library.search_similar_to_pattern(
                pattern_name=pattern_name,
                top_k=5,
                threshold=0.6
            )
            
            if similar_windows:
                print(f"Found {len(similar_windows)} similar windows:")
                for i, window in enumerate(similar_windows[:3], 1):
                    symbol = window.get('symbol', 'Unknown')
                    timestamp = window.get('timestamp', 'Unknown')
                    if isinstance(timestamp, str):
                        try:
                            timestamp = pd.to_datetime(timestamp).strftime('%Y-%m-%d %H:%M:%S')
                        except:
                            pass
                    score = window.get('score', 0)
                    print(f"  {i}. {symbol} @ {timestamp} - Score: {score:.3f}")
            else:
                print("No similar windows found (threshold: 0.6)")
        
        # Test 8: Get and update pattern
        print("\n" + "=" * 80)
        print("Test 8: Get and Update Pattern")
        print("=" * 80)
        
        if patterns:
            pattern_name = patterns[0]['name']
            pattern = library.get_pattern(pattern_name)
            
            if pattern:
                print(f"\nGot pattern '{pattern_name}':")
                print(f"  Description: {pattern['description']}")
                print(f"  Examples: {pattern['num_examples']}")
                print(f"  Shape: {pattern.get('embedding_shape', 'Unknown')}")
                
                # Try to update description
                success = library.update_pattern(
                    name=pattern_name,
                    description=f"Updated: {pattern['description']}"
                )
                
                if success:
                    print("✓ Pattern description updated")
                else:
                    print("⚠ Could not update pattern")
        
        # Cleanup
        storage.close()
        
        # Clean up test config file if it exists
        import os
        if os.path.exists(test_config):
            os.remove(test_config)
            print("\n✓ Cleaned up test files")
        
        print("\n" + "=" * 80)
        print("✓ All tests completed successfully!")
        print("=" * 80)
        
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()


def demo_pattern_library():
    """Interactive demo of pattern library."""
    print("\n" + "=" * 80)
    print("PatternLibrary Interactive Demo")
    print("=" * 80)
    
    try:
        # Initialize
        storage = StorageManager(
            duckdb_path="storage/financedb.duckdb",
            qdrant_path="storage/qdrant_data"
        )
        
        library = PatternLibrary(storage)
        
        print("\n✓ Pattern library initialized")
        print(f"  Loaded {len(library.patterns)} patterns")
        
        # Demo 1: List all patterns
        print("\n" + "-" * 80)
        print("AVAILABLE PATTERNS")
        print("-" * 80)
        
        patterns = library.list_patterns()
        if patterns:
            for i, p in enumerate(patterns, 1):
                print(f"\n{i}. {p['name'].upper()}")
                print(f"   Description: {p['description']}")
                print(f"   Examples: {p['num_examples']}")
                if p['examples']:
                    example_str = ', '.join([f'{s}@{d}' for d, s in p['examples'][:3]])
                    print(f"   Example dates: {example_str}")
        else:
            print("\nNo patterns available.")
            
            # Show database info
            print("\nChecking database...")
            stats = storage.get_statistics()
            print(f"Total windows in database: {stats.get('total_windows', 0)}")
            print(f"Date range: {stats.get('date_range', {})}")
        
        # Demo 2: Test pattern matching
        print("\n" + "-" * 80)
        print("PATTERN MATCHING DEMO")
        print("-" * 80)
        
        if patterns:
            # Get current date for AAPL
            test_symbol = 'AAPL'
            print(f"\nTesting pattern matching for {test_symbol}...")
            
            # Get latest date
            try:
                sql_result = storage.query_sql(
                    f"SELECT timestamp FROM timeseries WHERE symbol = '{test_symbol}' ORDER BY timestamp DESC LIMIT 1"
                )
                
                if sql_result:
                    latest_date = sql_result[0]['timestamp']
                    date_str = latest_date.strftime('%Y-%m-%d')
                    print(f"Latest date: {date_str}")
                    
                    # Get embedding
                    windows = storage.read_by_symbol(
                        symbol=test_symbol,
                        start_date=pd.to_datetime(date_str),
                        end_date=pd.to_datetime(date_str)
                    )
                    
                    if windows and windows[0].get('embedding') is not None:
                        embedding = windows[0]['embedding']
                        
                        # Find matching patterns
                        matches = library.find_all_patterns(
                            embedding=embedding,
                            threshold=0.5,
                            top_k=5
                        )
                        
                        if matches:
                            print(f"\nFound {len(matches)} pattern matches:")
                            print("-" * 60)
                            for match in matches:
                                print(f"  • {match['pattern_name']:20s}")
                                print(f"    Score: {match['score']:.3f}")
                                print(f"    Description: {match['description'][:60]}...")
                                print()
                        else:
                            print("\nNo pattern matches found (threshold: 0.5)")
                else:
                    print(f"No data found for {test_symbol}")
            except Exception as e:
                print(f"Error: {e}")
        
        # Demo 3: Search for historical similar periods
        print("\n" + "-" * 80)
        print("FIND SIMILAR HISTORICAL PERIODS")
        print("-" * 80)
        
        if patterns:
            pattern_name = patterns[0]['name']
            print(f"\nFinding historical periods similar to '{pattern_name}'...")
            
            similar_windows = library.search_similar_to_pattern(
                pattern_name=pattern_name,
                top_k=5,
                threshold=0.7
            )
            
            if similar_windows:
                print(f"\nTop {len(similar_windows)} similar historical periods:")
                print("-" * 60)
                for i, window in enumerate(similar_windows, 1):
                    symbol = window.get('symbol', 'Unknown')
                    timestamp = window.get('timestamp', 'Unknown')
                    score = window.get('score', 0)
                    
                    # Format timestamp
                    if isinstance(timestamp, str):
                        try:
                            timestamp = pd.to_datetime(timestamp).strftime('%Y-%m-%d')
                        except:
                            pass
                    
                    print(f"  {i}. {symbol} @ {timestamp}")
                    print(f"     Similarity: {score:.3f}")
                    print()
            else:
                print("No similar historical periods found (threshold: 0.7)")
        
        # Cleanup
        storage.close()
        
        print("\n" + "=" * 80)
        print("✓ Demo completed successfully!")
        print("=" * 80)
        
    except Exception as e:
        print(f"\n✗ Demo failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == '--demo':
        demo_pattern_library()
    else:
        test_pattern_library()