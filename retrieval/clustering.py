"""
FinanceDB - Pattern Discovery Module
Unsupervised clustering to discover financial patterns using K-Means.

Author: FinanceDB Project
Module: 5 - Semantic Retrieval (Clustering)
"""

import sys
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
from datetime import datetime, timedelta
import pickle
import numpy as np
import pandas as pd
import matplotlib
# Use non-interactive backend to avoid Tcl/Tk issues
matplotlib.use('Agg')  # Use Agg backend for non-interactive plotting
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.decomposition import PCA
from tqdm import tqdm
from loguru import logger

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from storage.storage_manager import StorageManager
from retrieval.patterns import PatternLibrary


class PatternDiscovery:
    """
    Discover market patterns through unsupervised clustering.
    
    Features:
    - K-Means clustering on embeddings
    - Automatic pattern discovery
    - t-SNE visualization
    - Export to PatternLibrary
    - Manual cluster labeling
    """
    
    def __init__(
        self,
        storage_manager: StorageManager,
        random_state: int = 42
    ):
        """
        Initialize pattern discovery engine.
        
        Args:
            storage_manager: StorageManager instance
            random_state: Random seed for reproducibility
        """
        self.storage = storage_manager
        self.random_state = random_state
        
        # Clustering state
        self.embeddings = None
        self.window_ids = None
        self.symbols = None
        self.timestamps = None
        self.kmeans = None
        self.cluster_labels = None
        self.cluster_centroids = None
        self.cluster_names = {}  # Manual labels
        
        logger.info("Initialized PatternDiscovery")
        logger.info(f"  Random state: {random_state}")
    
    def load_embeddings(self, sample_size: Optional[int] = None, normalize: bool = True):
        """
        Load all embeddings from storage.
        
        Args:
            sample_size: Optional limit on number of embeddings
            normalize: Whether to normalize embeddings (recommended)
        """
        try:
            logger.info("Loading embeddings from storage...")
            
            # Query all windows
            if sample_size:
                sql = f"SELECT window_id, symbol, timestamp, embedding FROM timeseries ORDER BY RANDOM() LIMIT {sample_size}"
            else:
                sql = "SELECT window_id, symbol, timestamp, embedding FROM timeseries"
            
            results = self.storage.duckdb.query(sql)
            
            # Extract data
            embeddings_list = []
            self.window_ids = []
            self.symbols = []
            self.timestamps = []
            
            failed_count = 0
            for result in tqdm(results, desc="Loading embeddings"):
                # Handle different embedding formats
                embedding_data = result['embedding']
                
                # Try to deserialize pickled data
                try:
                    if isinstance(embedding_data, (bytes, str)):
                        if isinstance(embedding_data, str):
                            embedding_data = embedding_data.encode('latin-1')
                        
                        # Try pickle
                        try:
                            embedding = pickle.loads(embedding_data)
                        except:
                            # Try to parse as bytes directly
                            embedding = np.frombuffer(embedding_data, dtype=np.float32)
                    elif hasattr(embedding_data, '__array__'):
                        embedding = np.array(embedding_data, dtype=np.float32)
                    else:
                        # Try to convert whatever it is
                        embedding = np.array(embedding_data, dtype=np.float32)
                    
                    # Ensure it's a numpy array
                    if not isinstance(embedding, np.ndarray):
                        embedding = np.array(embedding, dtype=np.float32)
                    
                    # Flatten if needed
                    if embedding.ndim > 1:
                        embedding = embedding.flatten()
                    
                    # Check if embedding is valid
                    if embedding.size == 0 or np.isnan(embedding).any() or np.isinf(embedding).any():
                        failed_count += 1
                        continue
                        
                    embeddings_list.append(embedding)
                    self.window_ids.append(result['window_id'])
                    self.symbols.append(result['symbol'])
                    self.timestamps.append(result['timestamp'])
                    
                except Exception as e:
                    failed_count += 1
                    continue
            
            if not embeddings_list:
                raise ValueError("No valid embeddings found")
            
            if failed_count > 0:
                logger.warning(f"Failed to parse {failed_count} embeddings")
            
            # Convert to numpy array
            self.embeddings = np.vstack(embeddings_list)
            
            # Normalize embeddings if requested
            if normalize and len(self.embeddings) > 0:
                logger.info("Normalizing embeddings...")
                scaler = StandardScaler()
                self.embeddings = scaler.fit_transform(self.embeddings)
                logger.info("✓ Normalization complete")
            
            logger.info(f"Loaded {len(self.embeddings)} embeddings")
            logger.info(f"  Embedding shape: {self.embeddings.shape}")
            logger.info(f"  Embedding dtype: {self.embeddings.dtype}")
            
        except Exception as e:
            logger.error(f"Failed to load embeddings: {e}")
            raise
    
    def discover_patterns(self, k: int = 10, n_init: int = 10) -> Dict[str, Any]:
        """
        Discover patterns using K-Means clustering.
        
        Args:
            k: Number of clusters
            n_init: Number of K-Means initializations
            
        Returns:
            Dictionary with clustering results
            
        Example:
            results = discovery.discover_patterns(k=10)
        """
        try:
            if self.embeddings is None:
                raise ValueError("No embeddings loaded. Call load_embeddings() first.")
            
            logger.info(f"Running K-Means clustering with k={k}...")
            
            # Run K-Means
            self.kmeans = KMeans(
                n_clusters=k,
                random_state=self.random_state,
                n_init=n_init,
                max_iter=300,
                verbose=0
            )
            
            self.cluster_labels = self.kmeans.fit_predict(self.embeddings)
            self.cluster_centroids = self.kmeans.cluster_centers_
            
            # Compute cluster statistics
            cluster_sizes = np.bincount(self.cluster_labels)
            
            # Get inertia (within-cluster sum of squares)
            inertia = self.kmeans.inertia_
            
            # Compute silhouette score if possible
            silhouette = None
            davies_bouldin = None
            if k > 1 and len(self.embeddings) > k:
                try:
                    silhouette = silhouette_score(self.embeddings, self.cluster_labels)
                except:
                    pass
                
                try:
                    davies_bouldin = davies_bouldin_score(self.embeddings, self.cluster_labels)
                except:
                    pass
            
            results = {
                'n_clusters': k,
                'cluster_sizes': cluster_sizes.tolist(),
                'cluster_centroids': self.cluster_centroids,
                'inertia': inertia,
                'silhouette_score': silhouette,
                'davies_bouldin_score': davies_bouldin,
                'labels': self.cluster_labels.tolist()
            }
            
            logger.info(f"✓ Clustering complete")
            logger.info(f"  Inertia: {inertia:.2f}")
            if silhouette is not None:
                logger.info(f"  Silhouette score: {silhouette:.3f}")
            if davies_bouldin is not None:
                logger.info(f"  Davies-Bouldin score: {davies_bouldin:.3f}")
            logger.info(f"  Cluster sizes: {cluster_sizes.tolist()}")
            
            return results
            
        except Exception as e:
            logger.error(f"Clustering failed: {e}")
            raise
    
    def find_optimal_k(self, max_k: int = 20) -> Dict[str, Any]:
        """
        Find optimal number of clusters using elbow method.
        
        Args:
            max_k: Maximum k to try
            
        Returns:
            Dictionary with inertia values and recommended k
        """
        try:
            if self.embeddings is None:
                raise ValueError("No embeddings loaded. Call load_embeddings() first.")
            
            logger.info(f"Finding optimal k up to {max_k}...")
            
            inertias = []
            silhouette_scores = []
            davies_bouldin_scores = []
            k_values = range(2, max_k + 1)
            
            for k in tqdm(k_values, desc="Testing k values"):
                kmeans = KMeans(
                    n_clusters=k,
                    random_state=self.random_state,
                    n_init=5,
                    max_iter=200
                )
                kmeans.fit(self.embeddings)
                inertias.append(kmeans.inertia_)
                
                # Compute silhouette score
                if k > 1:
                    try:
                        labels = kmeans.labels_
                        silhouette = silhouette_score(self.embeddings, labels)
                        silhouette_scores.append(silhouette)
                    except:
                        silhouette_scores.append(None)
                    
                    try:
                        davies_bouldin = davies_bouldin_score(self.embeddings, labels)
                        davies_bouldin_scores.append(davies_bouldin)
                    except:
                        davies_bouldin_scores.append(None)
            
            # Find elbow point
            diff = np.diff(inertias)
            diff_ratio = diff[:-1] / diff[1:]
            elbow_k = np.argmax(diff_ratio) + 3  # +2 for starting at k=2, +1 for index
            
            # Find max silhouette score
            valid_silhouettes = [s for s in silhouette_scores if s is not None]
            if valid_silhouettes:
                silhouette_k = np.argmax(valid_silhouettes) + 2
            else:
                silhouette_k = None
            
            # Find min Davies-Bouldin score
            valid_davies = [s for s in davies_bouldin_scores if s is not None]
            if valid_davies:
                davies_k = np.argmin(valid_davies) + 2
            else:
                davies_k = None
            
            results = {
                'k_values': list(k_values),
                'inertias': inertias,
                'silhouette_scores': silhouette_scores,
                'davies_bouldin_scores': davies_bouldin_scores,
                'elbow_k': min(elbow_k, max_k),
                'silhouette_k': silhouette_k,
                'davies_k': davies_k
            }
            
            logger.info(f"✓ Optimal k analysis complete")
            logger.info(f"  Recommended k (elbow): {results['elbow_k']}")
            if silhouette_k:
                logger.info(f"  Recommended k (silhouette): {silhouette_k}")
            if davies_k:
                logger.info(f"  Recommended k (davies-bouldin): {davies_k}")
            
            return results
            
        except Exception as e:
            logger.error(f"Optimal k search failed: {e}")
            raise
    
    def get_cluster_exemplars(
        self,
        cluster_id: int,
        n: int = 5
    ) -> List[Dict]:
        """
        Get windows closest to cluster centroid (exemplars).
        
        Args:
            cluster_id: Cluster identifier (0 to k-1)
            n: Number of exemplars to return
            
        Returns:
            List of exemplar windows with distances
            
        Example:
            exemplars = discovery.get_cluster_exemplars(cluster_id=0, n=5)
        """
        try:
            if self.cluster_labels is None:
                raise ValueError("No clustering results. Call discover_patterns() first.")
            
            # Get centroid
            centroid = self.cluster_centroids[cluster_id]
            
            # Find windows in this cluster
            cluster_mask = self.cluster_labels == cluster_id
            cluster_indices = np.where(cluster_mask)[0]
            
            if len(cluster_indices) == 0:
                logger.warning(f"Cluster {cluster_id} is empty")
                return []
            
            # Compute distances to centroid
            cluster_embeddings = self.embeddings[cluster_indices]
            distances = np.linalg.norm(cluster_embeddings - centroid, axis=1)
            
            # Get top n closest
            sorted_indices = np.argsort(distances)[:n]
            
            # Build exemplar list
            exemplars = []
            for idx in sorted_indices:
                global_idx = cluster_indices[idx]
                exemplars.append({
                    'window_id': self.window_ids[global_idx],
                    'symbol': self.symbols[global_idx],
                    'timestamp': self.timestamps[global_idx],
                    'distance_to_centroid': float(distances[idx]),
                    'cluster_id': cluster_id
                })
            
            logger.info(f"Retrieved {len(exemplars)} exemplars for cluster {cluster_id}")
            
            return exemplars
            
        except Exception as e:
            logger.error(f"Failed to get exemplars: {e}")
            raise
    
    def get_cluster_summary(self, cluster_id: int) -> Dict[str, Any]:
        """
        Get summary statistics for a cluster.
        
        Args:
            cluster_id: Cluster identifier
            
        Returns:
            Dictionary with cluster statistics
        """
        try:
            if self.cluster_labels is None:
                raise ValueError("No clustering results. Call discover_patterns() first.")
            
            # Get cluster members
            cluster_mask = self.cluster_labels == cluster_id
            cluster_indices = np.where(cluster_mask)[0]
            
            if len(cluster_indices) == 0:
                return {'cluster_id': cluster_id, 'size': 0, 'error': 'Empty cluster'}
            
            # Get symbols and timestamps
            cluster_symbols = [self.symbols[i] for i in cluster_indices]
            cluster_timestamps = [self.timestamps[i] for i in cluster_indices]
            
            # Symbol distribution
            symbol_counts = pd.Series(cluster_symbols).value_counts().to_dict()
            
            # Temporal distribution
            try:
                dates = pd.to_datetime(cluster_timestamps)
                date_range = {
                    'start': dates.min().strftime('%Y-%m-%d'),
                    'end': dates.max().strftime('%Y-%m-%d'),
                    'span_days': (dates.max() - dates.min()).days
                }
                # Year-month distribution
                month_counts = dates.strftime('%Y-%m').value_counts().to_dict()
            except:
                date_range = {'start': 'Unknown', 'end': 'Unknown', 'span_days': 0}
                month_counts = {}
            
            # Manual label
            cluster_label = self.cluster_names.get(cluster_id, f"Cluster {cluster_id}")
            
            summary = {
                'cluster_id': cluster_id,
                'label': cluster_label,
                'size': len(cluster_indices),
                'symbol_distribution': symbol_counts,
                'month_distribution': month_counts,
                'date_range': date_range,
                'top_symbols': list(symbol_counts.keys())[:5],
                'percentage': len(cluster_indices) / len(self.cluster_labels) * 100
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"Failed to get cluster summary: {e}")
            return {'cluster_id': cluster_id, 'error': str(e)}
    
    def label_cluster(self, cluster_id: int, label: str):
        """
        Manually assign a semantic label to a cluster.
        
        Args:
            cluster_id: Cluster identifier
            label: Semantic label (e.g., "Bull Market", "Crash Pattern")
            
        Example:
            discovery.label_cluster(0, "High Volatility")
            discovery.label_cluster(3, "Bull Run")
        """
        self.cluster_names[cluster_id] = label
        logger.info(f"Labeled cluster {cluster_id} as '{label}'")
    
    def auto_label_clusters(self):
        """
        Automatically label clusters based on characteristics.
        This is a simple heuristic-based labeling.
        """
        try:
            if self.cluster_labels is None:
                raise ValueError("No clustering results. Call discover_patterns() first.")
            
            n_clusters = len(self.cluster_centroids)
            
            for cluster_id in range(n_clusters):
                summary = self.get_cluster_summary(cluster_id)
                
                if summary['size'] == 0:
                    continue
                
                # Analyze cluster characteristics
                size_percentage = summary['percentage']
                symbol_concentration = len(summary['symbol_distribution'])
                top_symbol = summary['top_symbols'][0] if summary['top_symbols'] else "Unknown"
                
                # Simple heuristics for labeling
                if size_percentage > 20:
                    label = "Common Pattern"
                elif size_percentage < 5:
                    label = "Rare Pattern"
                elif symbol_concentration < 3:
                    label = f"{top_symbol} Dominant"
                elif '2020-03' in summary.get('month_distribution', {}):
                    # Check if contains March 2020 (COVID crash)
                    if any('2020-03' in month for month in summary.get('month_distribution', {})):
                        label = "COVID Crash Pattern"
                    else:
                        label = f"Mixed Pattern {cluster_id}"
                else:
                    label = f"Mixed Pattern {cluster_id}"
                
                self.label_cluster(cluster_id, label)
            
            logger.info(f"Auto-labeled {n_clusters} clusters")
            
        except Exception as e:
            logger.error(f"Auto-labeling failed: {e}")
    
    def find_matching_windows(self, symbol: str, target_date: datetime, days_tolerance: int = 3) -> List[Dict]:
        """
        Find windows matching a symbol and date (with tolerance).
        
        Args:
            symbol: Stock symbol
            target_date: Target date
            days_tolerance: Days tolerance for matching
            
        Returns:
            List of matching windows
        """
        try:
            # Calculate date range
            start_date = target_date - timedelta(days=days_tolerance)
            end_date = target_date + timedelta(days=days_tolerance)
            
            # Use DuckDB's EXTRACT(EPOCH FROM ...) for date difference
            sql = f"""
            SELECT window_id, symbol, timestamp, embedding 
            FROM timeseries 
            WHERE symbol = '{symbol}' 
            AND timestamp >= '{start_date.strftime('%Y-%m-%d')}' 
            AND timestamp <= '{end_date.strftime('%Y-%m-%d')}'
            ORDER BY ABS(EXTRACT(EPOCH FROM timestamp) - EXTRACT(EPOCH FROM TIMESTAMP '{target_date.strftime('%Y-%m-%d %H:%M:%S')}'))
            LIMIT 3
            """
            
            results = self.storage.duckdb.query(sql)
            
            matches = []
            for result in results:
                matches.append({
                    'window_id': result['window_id'],
                    'symbol': result['symbol'],
                    'timestamp': result['timestamp'],
                    'date_str': result['timestamp'].strftime('%Y-%m-%d') if hasattr(result['timestamp'], 'strftime') else str(result['timestamp'])
                })
            
            return matches
            
        except Exception as e:
            logger.error(f"Failed to find matching windows: {e}")
            # Fallback to simpler query
            try:
                sql = f"""
                SELECT window_id, symbol, timestamp, embedding 
                FROM timeseries 
                WHERE symbol = '{symbol}' 
                AND timestamp >= '{start_date.strftime('%Y-%m-%d')}' 
                AND timestamp <= '{end_date.strftime('%Y-%m-%d')}'
                LIMIT 3
                """
                
                results = self.storage.duckdb.query(sql)
                matches = []
                for result in results:
                    matches.append({
                        'window_id': result['window_id'],
                        'symbol': result['symbol'],
                        'timestamp': result['timestamp'],
                        'date_str': result['timestamp'].strftime('%Y-%m-%d') if hasattr(result['timestamp'], 'strftime') else str(result['timestamp'])
                    })
                return matches
            except:
                return []
    
    def export_to_pattern_library(
        self,
        pattern_library: PatternLibrary,
        cluster_ids: Optional[List[int]] = None,
        use_labels: bool = True,
        days_tolerance: int = 7  # Increased tolerance for better matching
    ) -> List[str]:
        """
        Export discovered clusters as patterns to PatternLibrary.
        
        Args:
            pattern_library: PatternLibrary instance
            cluster_ids: Optional list of cluster IDs to export (default: all)
            use_labels: Use manual labels as pattern names
            days_tolerance: Days tolerance for matching dates
            
        Returns:
            List of exported pattern names
            
        Example:
            library = PatternLibrary(storage)
            exported = discovery.export_to_pattern_library(library, cluster_ids=[0, 3])
        """
        try:
            if self.cluster_labels is None:
                raise ValueError("No clustering results. Call discover_patterns() first.")
            
            # Determine which clusters to export
            if cluster_ids is None:
                cluster_ids = range(len(self.cluster_centroids))
            
            exported_patterns = []
            
            for cluster_id in cluster_ids:
                # Get exemplars
                exemplars = self.get_cluster_exemplars(cluster_id, n=5)
                
                if not exemplars:
                    logger.warning(f"Cluster {cluster_id} is empty, skipping")
                    continue
                
                # Generate pattern name
                if use_labels and cluster_id in self.cluster_names:
                    pattern_name = f"discovered_{self.cluster_names[cluster_id].lower().replace(' ', '_')}"
                else:
                    pattern_name = f"discovered_cluster_{cluster_id}"
                
                # Get description
                summary = self.get_cluster_summary(cluster_id)
                description = (
                    f"Auto-discovered pattern (Cluster {cluster_id}, "
                    f"{summary['size']} windows, "
                    f"top symbols: {', '.join(summary['top_symbols'][:3])})"
                )
                
                # Prepare example dates with tolerance
                example_dates = []
                dates_used = set()  # Track used dates to avoid duplicates
                
                for e in exemplars[:3]:  # Use top 3 exemplars
                    timestamp = e['timestamp']
                    
                    # Find matching windows with tolerance
                    matching_windows = self.find_matching_windows(
                        symbol=e['symbol'],
                        target_date=timestamp,
                        days_tolerance=days_tolerance
                    )
                    
                    if matching_windows:
                        # Use the first match that hasn't been used yet
                        for match in matching_windows:
                            date_key = f"{match['date_str']}_{match['symbol']}"
                            if date_key not in dates_used:
                                example_dates.append((match['date_str'], match['symbol']))
                                dates_used.add(date_key)
                                break
                        else:
                            # All matches already used, use exact date
                            if hasattr(timestamp, 'strftime'):
                                date_str = timestamp.strftime('%Y-%m-%d')
                            else:
                                date_str = str(timestamp).split()[0]
                            example_dates.append((date_str, e['symbol']))
                    else:
                        # Fallback to exact date
                        if hasattr(timestamp, 'strftime'):
                            date_str = timestamp.strftime('%Y-%m-%d')
                        else:
                            date_str = str(timestamp).split()[0]
                        example_dates.append((date_str, e['symbol']))
                
                # Skip if we don't have enough example dates
                if len(example_dates) < 1:
                    logger.warning(f"Cluster {cluster_id}: Not enough matching dates found")
                    continue
                
                # Add to pattern library
                success = pattern_library.add_pattern(
                    name=pattern_name,
                    example_dates=example_dates,
                    description=description,
                    overwrite=True
                )
                
                if success:
                    exported_patterns.append(pattern_name)
                    logger.info(f"✓ Exported cluster {cluster_id} as '{pattern_name}'")
                else:
                    logger.warning(f"Failed to export cluster {cluster_id}")
            
            # Save pattern library
            pattern_library.save_patterns()
            
            logger.info(f"Exported {len(exported_patterns)} patterns to library")
            
            return exported_patterns
            
        except Exception as e:
            logger.error(f"Export to pattern library failed: {e}")
            raise
    
    # ========================================================================
    # VISUALIZATION METHODS
    # ========================================================================
    
    def visualize_clusters(
        self,
        perplexity: int = 30,
        figsize: Tuple[int, int] = (12, 8),
        save_path: Optional[str] = None
    ):
        """
        Visualize clusters using t-SNE dimensionality reduction.
        
        Args:
            perplexity: t-SNE perplexity parameter (5-50)
            figsize: Figure size
            save_path: Optional path to save figure
        """
        try:
            if self.cluster_labels is None:
                raise ValueError("No clustering results. Call discover_patterns() first.")
            
            logger.info("Running t-SNE dimensionality reduction...")
            
            # Use PCA first for speed if high dimensional
            if self.embeddings.shape[1] > 50:
                logger.info("Using PCA pre-reduction for speed...")
                pca = PCA(n_components=50, random_state=self.random_state)
                embeddings_reduced = pca.fit_transform(self.embeddings)
            else:
                embeddings_reduced = self.embeddings
            
            # Run t-SNE with compatible parameters
            try:
                # Try with newer API
                tsne = TSNE(
                    n_components=2,
                    perplexity=min(perplexity, len(self.embeddings) - 1),
                    random_state=self.random_state,
                    learning_rate=200,
                    n_iter=1000
                )
            except TypeError:
                # Fallback for older versions
                tsne = TSNE(
                    n_components=2,
                    perplexity=min(perplexity, len(self.embeddings) - 1),
                    random_state=self.random_state,
                    learning_rate=200
                )
            
            embeddings_2d = tsne.fit_transform(embeddings_reduced)
            
            # Create plot
            fig, ax = plt.subplots(figsize=figsize)
            
            # Get unique clusters
            unique_clusters = np.unique(self.cluster_labels)
            n_clusters = len(unique_clusters)
            
            # Color map
            colors = plt.cm.tab20(np.linspace(0, 1, n_clusters))
            
            # Plot each cluster
            for i, cluster_id in enumerate(unique_clusters):
                cluster_mask = self.cluster_labels == cluster_id
                cluster_points = embeddings_2d[cluster_mask]
                
                # Get cluster label
                if cluster_id in self.cluster_names:
                    label = self.cluster_names[cluster_id]
                else:
                    label = f"Cluster {cluster_id}"
                
                # Plot points
                ax.scatter(
                    cluster_points[:, 0],
                    cluster_points[:, 1],
                    c=[colors[i]],
                    label=f"{label} ({np.sum(cluster_mask)})",
                    alpha=0.6,
                    s=30,
                    edgecolors='black',
                    linewidth=0.5
                )
            
            # Styling
            ax.set_xlabel('t-SNE Component 1', fontsize=12)
            ax.set_ylabel('t-SNE Component 2', fontsize=12)
            ax.set_title(
                f'Discovered Market Patterns (K-Means, k={n_clusters})',
                fontsize=14,
                fontweight='bold'
            )
            ax.legend(
                loc='upper left',
                bbox_to_anchor=(1.02, 1),
                fontsize=9,
                framealpha=0.9
            )
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Saved cluster visualization to {save_path}")
                plt.close(fig)  # Close the figure to free memory
            else:
                # Save to default location
                default_path = 'docs/cluster_visualization.png'
                plt.savefig(default_path, dpi=300, bbox_inches='tight')
                logger.info(f"Saved cluster visualization to {default_path}")
                plt.close(fig)
            
        except Exception as e:
            logger.error(f"Cluster visualization failed: {e}")
            plt.close('all')  # Close any open figures
    
    def plot_elbow_method(self, max_k: int = 20, figsize: Tuple[int, int] = (14, 5), 
                         save_path: Optional[str] = None):
        """
        Plot elbow method to determine optimal k.
        
        Args:
            max_k: Maximum k to try
            figsize: Figure size
            save_path: Optional path to save figure
        """
        try:
            if self.embeddings is None:
                raise ValueError("No embeddings loaded. Call load_embeddings() first.")
            
            results = self.find_optimal_k(max_k=max_k)
            
            fig, axs = plt.subplots(1, 3, figsize=figsize)
            
            # Plot inertia
            axs[0].plot(results['k_values'], results['inertias'], 'bo-', linewidth=2)
            axs[0].axvline(x=results['elbow_k'], color='r', linestyle='--', alpha=0.5, 
                          label=f'Elbow k={results["elbow_k"]}')
            axs[0].set_xlabel('Number of clusters (k)', fontsize=12)
            axs[0].set_ylabel('Inertia', fontsize=12)
            axs[0].set_title('Elbow Method', fontsize=14, fontweight='bold')
            axs[0].legend()
            axs[0].grid(True, alpha=0.3)
            
            # Plot silhouette scores
            valid_scores = [(k, s) for k, s in zip(results['k_values'], results['silhouette_scores']) if s is not None]
            if valid_scores:
                k_vals, scores = zip(*valid_scores)
                axs[1].plot(k_vals, scores, 'go-', linewidth=2)
                if results['silhouette_k']:
                    axs[1].axvline(x=results['silhouette_k'], color='r', linestyle='--', alpha=0.5,
                                 label=f'Silhouette k={results["silhouette_k"]}')
                    axs[1].legend()
                axs[1].set_xlabel('Number of clusters (k)', fontsize=12)
                axs[1].set_ylabel('Silhouette Score', fontsize=12)
                axs[1].set_title('Silhouette Analysis', fontsize=14, fontweight='bold')
                axs[1].grid(True, alpha=0.3)
            
            # Plot Davies-Bouldin scores
            valid_davies = [(k, s) for k, s in zip(results['k_values'], results['davies_bouldin_scores']) if s is not None]
            if valid_davies:
                k_vals, scores = zip(*valid_davies)
                axs[2].plot(k_vals, scores, 'mo-', linewidth=2)
                if results['davies_k']:
                    axs[2].axvline(x=results['davies_k'], color='r', linestyle='--', alpha=0.5,
                                 label=f'Davies-Bouldin k={results["davies_k"]}')
                    axs[2].legend()
                axs[2].set_xlabel('Number of clusters (k)', fontsize=12)
                axs[2].set_ylabel('Davies-Bouldin Score', fontsize=12)
                axs[2].set_title('Davies-Bouldin Index', fontsize=14, fontweight='bold')
                axs[2].grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Saved elbow plot to {save_path}")
                plt.close(fig)
            else:
                # Save to default location
                default_path = 'docs/elbow_analysis.png'
                plt.savefig(default_path, dpi=300, bbox_inches='tight')
                logger.info(f"Saved elbow plot to {default_path}")
                plt.close(fig)
            
            logger.info(f"Optimal k: {results['elbow_k']} (elbow method)")
            if results['silhouette_k']:
                logger.info(f"Optimal k: {results['silhouette_k']} (silhouette method)")
            if results['davies_k']:
                logger.info(f"Optimal k: {results['davies_k']} (davies-bouldin method)")
            
        except Exception as e:
            logger.error(f"Elbow plot failed: {e}")
            plt.close('all')
    
    def plot_cluster_distribution(self, figsize: Tuple[int, int] = (10, 6), 
                                 save_path: Optional[str] = None):
        """
        Plot cluster size distribution.
        
        Args:
            figsize: Figure size
            save_path: Optional path to save figure
        """
        try:
            if self.cluster_labels is None:
                raise ValueError("No clustering results. Call discover_patterns() first.")
            
            cluster_sizes = np.bincount(self.cluster_labels)
            cluster_ids = np.arange(len(cluster_sizes))
            
            fig, ax = plt.subplots(figsize=figsize)
            
            # Get cluster names for labels
            labels = [self.cluster_names.get(i, f"Cluster {i}") for i in cluster_ids]
            
            colors = plt.cm.tab20(np.linspace(0, 1, len(cluster_ids)))
            bars = ax.bar(cluster_ids, cluster_sizes, color=colors)
            
            # Add labels
            for i, (bar, size) in enumerate(zip(bars, cluster_sizes)):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + max(cluster_sizes)*0.01,
                       f'{size}\n({size/len(self.cluster_labels)*100:.1f}%)',
                       ha='center', va='bottom', fontsize=9)
            
            ax.set_xlabel('Cluster ID', fontsize=12)
            ax.set_ylabel('Number of Windows', fontsize=12)
            ax.set_title(f'Cluster Distribution (Total: {len(self.cluster_labels)} windows)', 
                        fontsize=14, fontweight='bold')
            ax.set_xticks(cluster_ids)
            ax.set_xticklabels([f"{i}" for i in cluster_ids], fontsize=10)
            
            # Add legend with cluster names
            legend_labels = [f"{i}: {label}" for i, label in enumerate(labels)]
            ax.legend(bars, legend_labels, title="Cluster Labels", 
                     bbox_to_anchor=(1.05, 1), loc='upper left')
            
            ax.grid(True, alpha=0.3, axis='y')
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Saved cluster distribution to {save_path}")
                plt.close(fig)
            else:
                # Save to default location
                default_path = 'docs/cluster_distribution.png'
                plt.savefig(default_path, dpi=300, bbox_inches='tight')
                logger.info(f"Saved cluster distribution to {default_path}")
                plt.close(fig)
            
        except Exception as e:
            logger.error(f"Cluster distribution plot failed: {e}")
            plt.close('all')
    
    def plot_top_symbols_per_cluster(self, n_symbols: int = 3, figsize: Tuple[int, int] = (12, 8),
                                    save_path: Optional[str] = None):
        """
        Plot top symbols for each cluster.
        
        Args:
            n_symbols: Number of top symbols to show per cluster
            figsize: Figure size
            save_path: Optional path to save figure
        """
        try:
            if self.cluster_labels is None:
                raise ValueError("No clustering results. Call discover_patterns() first.")
            
            n_clusters = len(self.cluster_centroids)
            
            # Calculate number of rows needed
            n_rows = (n_clusters + 2) // 3  # 3 columns per row
            fig, axes = plt.subplots(n_rows, 3, figsize=figsize)
            
            # Flatten axes array if needed
            if n_rows == 1:
                axes = axes.reshape(1, -1)
            
            for cluster_id in range(n_clusters):
                row = cluster_id // 3
                col = cluster_id % 3
                
                if n_rows == 1:
                    ax = axes[col]
                else:
                    ax = axes[row, col]
                
                summary = self.get_cluster_summary(cluster_id)
                
                if 'error' in summary:
                    ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
                    ax.set_title(f'Cluster {cluster_id}')
                    continue
                
                # Get top symbols
                symbol_counts = summary['symbol_distribution']
                top_symbols = list(symbol_counts.items())[:n_symbols]
                
                if not top_symbols:
                    ax.text(0.5, 0.5, 'No symbols', ha='center', va='center', transform=ax.transAxes)
                    continue
                
                symbols, counts = zip(*top_symbols)
                
                # Plot
                colors = plt.cm.tab10(np.linspace(0, 1, len(symbols)))
                bars = ax.bar(symbols, counts, color=colors)
                
                # Add labels
                for i, (symbol, count) in enumerate(top_symbols):
                    percentage = count / summary['size'] * 100
                    ax.text(i, count + max(counts)*0.02, f'{percentage:.1f}%', 
                           ha='center', va='bottom', fontsize=8)
                
                cluster_label = self.cluster_names.get(cluster_id, f"Cluster {cluster_id}")
                ax.set_title(f'{cluster_label} (n={summary["size"]})', fontsize=10, fontweight='bold')
                ax.set_ylabel('Count')
                ax.tick_params(axis='x', rotation=45)
                ax.grid(True, alpha=0.3, axis='y')
            
            # Hide unused subplots
            for cluster_id in range(n_clusters, n_rows * 3):
                row = cluster_id // 3
                col = cluster_id % 3
                if n_rows == 1:
                    axes[col].axis('off')
                else:
                    axes[row, col].axis('off')
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Saved top symbols plot to {save_path}")
                plt.close(fig)
            else:
                # Save to default location
                default_path = 'docs/top_symbols_per_cluster.png'
                plt.savefig(default_path, dpi=300, bbox_inches='tight')
                logger.info(f"Saved top symbols plot to {default_path}")
                plt.close(fig)
            
        except Exception as e:
            logger.error(f"Top symbols plot failed: {e}")
            plt.close('all')
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive clustering statistics.
        
        Returns:
            Dictionary with statistics
        """
        if self.cluster_labels is None:
            return {'error': 'No clustering results available'}
        
        try:
            cluster_sizes = np.bincount(self.cluster_labels)
            
            stats = {
                'n_embeddings': len(self.embeddings),
                'embedding_dim': self.embeddings.shape[1],
                'n_clusters': len(self.cluster_centroids),
                'cluster_sizes': cluster_sizes.tolist(),
                'labeled_clusters': len(self.cluster_names),
                'cluster_labels': self.cluster_names,
                'inertia': self.kmeans.inertia_ if self.kmeans else None,
                'average_cluster_size': np.mean(cluster_sizes),
                'std_cluster_size': np.std(cluster_sizes),
                'min_cluster_size': np.min(cluster_sizes),
                'max_cluster_size': np.max(cluster_sizes),
                'size_ratio': np.max(cluster_sizes) / np.min(cluster_sizes) if np.min(cluster_sizes) > 0 else float('inf')
            }
            
            # Add silhouette score if possible
            if len(np.unique(self.cluster_labels)) > 1:
                try:
                    stats['silhouette_score'] = silhouette_score(self.embeddings, self.cluster_labels)
                except:
                    stats['silhouette_score'] = None
                
                try:
                    stats['davies_bouldin_score'] = davies_bouldin_score(self.embeddings, self.cluster_labels)
                except:
                    stats['davies_bouldin_score'] = None
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get statistics: {e}")
            return {'error': str(e)}
    
    def generate_report(self, report_path: str = 'docs/clustering_report.txt'):
        """
        Generate a comprehensive clustering report.
        
        Args:
            report_path: Path to save the report
        """
        try:
            if self.cluster_labels is None:
                raise ValueError("No clustering results. Call discover_patterns() first.")
            
            stats = self.get_statistics()
            
            with open(report_path, 'w') as f:
                f.write("=" * 80 + "\n")
                f.write("PATTERN DISCOVERY CLUSTERING REPORT\n")
                f.write("=" * 80 + "\n\n")
                
                f.write("SUMMARY\n")
                f.write("-" * 40 + "\n")
                f.write(f"Total windows analyzed: {stats['n_embeddings']:,}\n")
                f.write(f"Number of clusters: {stats['n_clusters']}\n")
                f.write(f"Inertia: {stats['inertia']:.2f}\n")
                if stats.get('silhouette_score'):
                    f.write(f"Silhouette score: {stats['silhouette_score']:.3f}\n")
                if stats.get('davies_bouldin_score'):
                    f.write(f"Davies-Bouldin score: {stats['davies_bouldin_score']:.3f}\n")
                f.write(f"Average cluster size: {stats['average_cluster_size']:.1f}\n")
                f.write(f"Cluster size range: {stats['min_cluster_size']} - {stats['max_cluster_size']}\n")
                f.write(f"Cluster size ratio: {stats['size_ratio']:.1f}\n")
                f.write(f"Standard deviation: {stats['std_cluster_size']:.1f}\n\n")
                
                f.write("CLUSTER DETAILS\n")
                f.write("-" * 40 + "\n")
                
                for cluster_id in range(stats['n_clusters']):
                    summary = self.get_cluster_summary(cluster_id)
                    exemplars = self.get_cluster_exemplars(cluster_id, n=2)
                    
                    f.write(f"\nCluster {cluster_id}: {summary['label']}\n")
                    f.write(f"  Size: {summary['size']} windows ({summary['percentage']:.1f}%)\n")
                    f.write(f"  Top 3 symbols: {', '.join(summary['top_symbols'][:3])}\n")
                    f.write(f"  Date range: {summary['date_range']['start']} to {summary['date_range']['end']}\n")
                    
                    if exemplars:
                        f.write("  Top exemplars:\n")
                        for ex in exemplars:
                            f.write(f"    - {ex['symbol']} @ {ex['timestamp']} (dist: {ex['distance_to_centroid']:.3f})\n")
                
                f.write("\n" + "=" * 80 + "\n")
                f.write("CLUSTERING QUALITY ASSESSMENT\n")
                f.write("-" * 40 + "\n")
                
                if stats.get('silhouette_score'):
                    silhouette = stats['silhouette_score']
                    if silhouette > 0.7:
                        quality = "Excellent (strong structure)"
                    elif silhouette > 0.5:
                        quality = "Good (reasonable structure)"
                    elif silhouette > 0.25:
                        quality = "Fair (weak structure)"
                    elif silhouette > 0:
                        quality = "Poor (no substantial structure)"
                    else:
                        quality = "Very poor (incorrect clustering)"
                    
                    f.write(f"Silhouette score: {silhouette:.3f}\n")
                    f.write(f"Interpretation: {quality}\n")
                
                if stats.get('davies_bouldin_score'):
                    db = stats['davies_bouldin_score']
                    if db < 0.5:
                        quality = "Excellent (well-separated)"
                    elif db < 1.0:
                        quality = "Good (reasonable separation)"
                    elif db < 2.0:
                        quality = "Fair (some overlap)"
                    else:
                        quality = "Poor (high overlap)"
                    
                    f.write(f"Davies-Bouldin score: {db:.3f}\n")
                    f.write(f"Interpretation: {quality}\n")
                
                # Balance assessment
                size_ratio = stats['size_ratio']
                if size_ratio < 3:
                    balance = "Well-balanced clusters"
                elif size_ratio < 10:
                    balance = "Moderately imbalanced"
                else:
                    balance = "Highly imbalanced - consider removing small clusters"
                
                f.write(f"\nCluster balance:\n")
                f.write(f"  Size ratio (max/min): {size_ratio:.1f}\n")
                f.write(f"  Assessment: {balance}\n")
                
                # Recommendations
                f.write("\n" + "=" * 80 + "\n")
                f.write("RECOMMENDATIONS\n")
                f.write("-" * 40 + "\n")
                
                if stats.get('silhouette_score') and stats['silhouette_score'] < 0.3:
                    f.write("1. Consider trying different clustering algorithms (DBSCAN, Gaussian Mixture)\n")
                    f.write("2. Try different feature representations or dimensionality reduction\n")
                    f.write("3. Check if embeddings capture meaningful market patterns\n")
                
                if size_ratio > 10:
                    f.write("1. Consider removing very small clusters (size < 5% of data)\n")
                    f.write("2. Try hierarchical clustering with minimum cluster size constraint\n")
                    f.write("3. Adjust k to get more balanced clusters\n")
                
                # Timestamp
                from datetime import datetime
                f.write(f"\nReport generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            
            logger.info(f"Generated clustering report at {report_path}")
            
        except Exception as e:
            logger.error(f"Failed to generate report: {e}")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def test_pattern_discovery():
    """Test the pattern discovery module."""
    print("=" * 80)
    print("PatternDiscovery Unit Tests")
    print("=" * 80)
    
    try:
        # Initialize
        print("\nInitializing storage and pattern discovery...")
        storage = StorageManager(
            duckdb_path="storage/financedb.duckdb",
            qdrant_path="storage/qdrant_data"
        )
        
        discovery = PatternDiscovery(storage)
        print("✓ PatternDiscovery initialized")
        
        # Test 1: Load embeddings
        print("\n" + "=" * 80)
        print("Test 1: Load Embeddings")
        print("=" * 80)
        
        discovery.load_embeddings(sample_size=1000)  # Sample for speed
        
        assert discovery.embeddings is not None
        print(f"✓ Loaded {len(discovery.embeddings)} embeddings")
        print(f"  Shape: {discovery.embeddings.shape}")
        print(f"  Dtype: {discovery.embeddings.dtype}")
        
        # Test 2: Discover patterns
        print("\n" + "=" * 80)
        print("Test 2: Discover Patterns")
        print("=" * 80)
        
        # First find optimal k
        optimal_k_results = discovery.find_optimal_k(max_k=10)
        k = optimal_k_results['elbow_k']
        print(f"Optimal k determined: {k}")
        
        # Run clustering
        results = discovery.discover_patterns(k=k)
        
        assert results['n_clusters'] == k
        print(f"✓ Discovered {results['n_clusters']} patterns")
        print(f"  Cluster sizes: {results['cluster_sizes']}")
        print(f"  Inertia: {results['inertia']:.2f}")
        if results.get('silhouette_score'):
            print(f"  Silhouette score: {results['silhouette_score']:.3f}")
        if results.get('davies_bouldin_score'):
            print(f"  Davies-Bouldin score: {results['davies_bouldin_score']:.3f}")
        
        # Test 3: Get exemplars
        print("\n" + "=" * 80)
        print("Test 3: Get Cluster Exemplars")
        print("=" * 80)
        
        exemplars = discovery.get_cluster_exemplars(cluster_id=0, n=3)
        
        assert len(exemplars) > 0
        print(f"\nCluster 0 exemplars:")
        for i, ex in enumerate(exemplars, 1):
            print(f"  {i}. {ex['symbol']:6s} @ {ex['timestamp']} | "
                  f"Distance: {ex['distance_to_centroid']:.4f}")
        
        print("\n✓ Exemplar retrieval successful")
        
        # Test 4: Cluster summary
        print("\n" + "=" * 80)
        print("Test 4: Cluster Summary")
        print("=" * 80)
        
        summary = discovery.get_cluster_summary(cluster_id=0)
        
        print(f"\nCluster 0 Summary:")
        print(f"  Size: {summary['size']}")
        print(f"  Percentage: {summary['percentage']:.1f}%")
        print(f"  Top symbols: {summary['top_symbols'][:3]}")
        print(f"  Date range: {summary['date_range']['start']} to {summary['date_range']['end']}")
        
        print("\n✓ Cluster summary successful")
        
        # Test 5: Label clusters
        print("\n" + "=" * 80)
        print("Test 5: Manual Cluster Labeling")
        print("=" * 80)
        
        discovery.label_cluster(0, "High Volatility")
        discovery.label_cluster(1, "Steady Growth")
        discovery.label_cluster(2, "Market Correction")
        
        print("✓ Labeled 3 clusters")
        
        # Test 6: Auto-label clusters
        print("\n" + "=" * 80)
        print("Test 6: Auto-label Clusters")
        print("=" * 80)
        
        discovery.auto_label_clusters()
        print(f"✓ Auto-labeled all clusters")
        
        # Test 7: Statistics
        print("\n" + "=" * 80)
        print("Test 7: Get Statistics")
        print("=" * 80)
        
        stats = discovery.get_statistics()
        
        print(f"\nClustering Statistics:")
        print(f"  Embeddings: {stats['n_embeddings']:,}")
        print(f"  Clusters: {stats['n_clusters']}")
        print(f"  Labeled: {stats['labeled_clusters']}")
        print(f"  Inertia: {stats['inertia']:.2f}")
        if stats.get('silhouette_score'):
            print(f"  Silhouette: {stats['silhouette_score']:.3f}")
        if stats.get('davies_bouldin_score'):
            print(f"  Davies-Bouldin: {stats['davies_bouldin_score']:.3f}")
        
        print("\n✓ Statistics retrieved")
        
        # Test 8: Generate visualizations (non-interactive)
        print("\n" + "=" * 80)
        print("Test 8: Generate Visualizations")
        print("=" * 80)
        
        print("\nGenerating elbow method plot...")
        discovery.plot_elbow_method(max_k=10, save_path='docs/elbow_method.png')
        print("✓ Elbow plot saved to docs/elbow_method.png")
        
        print("\nGenerating cluster distribution plot...")
        discovery.plot_cluster_distribution(save_path='docs/cluster_distribution.png')
        print("✓ Cluster distribution saved to docs/cluster_distribution.png")
        
        print("\nGenerating t-SNE visualization...")
        discovery.visualize_clusters(
            perplexity=20,
            save_path='docs/cluster_visualization.png'
        )
        print("✓ Cluster visualization saved to docs/cluster_visualization.png")
        
        print("\nGenerating top symbols plot...")
        discovery.plot_top_symbols_per_cluster(save_path='docs/top_symbols.png')
        print("✓ Top symbols plot saved to docs/top_symbols.png")
        
        print("\nGenerating clustering report...")
        discovery.generate_report(report_path='docs/clustering_report.txt')
        print("✓ Clustering report saved to docs/clustering_report.txt")
        
        # Test 9: Export to pattern library (with date tolerance)
        print("\n" + "=" * 80)
        print("Test 9: Export to Pattern Library")
        print("=" * 80)
        
        library = PatternLibrary(
            storage_manager=storage,
            config_path="configs/patterns_test.yaml"
        )
        
        exported = discovery.export_to_pattern_library(
            library,
            cluster_ids=[0, 1, 2],
            use_labels=True,
            days_tolerance=7
        )
        
        print(f"\n✓ Exported {len(exported)} patterns:")
        for pattern in exported:
            print(f"  - {pattern}")
        
        # Cleanup
        storage.close()
        
        import os
        if os.path.exists("configs/patterns_test.yaml"):
            os.remove("configs/patterns_test.yaml")
        
        # Create docs directory if it doesn't exist
        os.makedirs('docs', exist_ok=True)
        
        print("\n" + "=" * 80)
        print("✓ All tests passed!")
        print("=" * 80)
        
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()


def demo_pattern_discovery():
    """Interactive demo of pattern discovery."""
    print("\n" + "=" * 80)
    print("PatternDiscovery Interactive Demo")
    print("=" * 80)
    
    try:
        # Initialize
        storage = StorageManager(
            duckdb_path="storage/financedb.duckdb",
            qdrant_path="storage/qdrant_data"
        )
        
        discovery = PatternDiscovery(storage)
        
        print("\n✓ Pattern discovery initialized")
        
        # Load embeddings
        print("\nLoading embeddings...")
        discovery.load_embeddings(sample_size=2000)  # Use sample for demo
        print(f"✓ Loaded {len(discovery.embeddings):,} embeddings")
        
        # Find optimal k
        print("\nFinding optimal number of clusters...")
        optimal_results = discovery.find_optimal_k(max_k=15)
        k = optimal_results['elbow_k']
        print(f"✓ Optimal k: {k}")
        
        # Discover patterns
        print(f"\nDiscovering patterns with K-Means (k={k})...")
        results = discovery.discover_patterns(k=k)
        
        print(f"\n✓ Discovered {results['n_clusters']} patterns")
        print(f"  Cluster sizes: {results['cluster_sizes']}")
        if results.get('silhouette_score'):
            print(f"  Silhouette score: {results['silhouette_score']:.3f}")
        if results.get('davies_bouldin_score'):
            print(f"  Davies-Bouldin score: {results['davies_bouldin_score']:.3f}")
        
        # Auto-label clusters
        print("\nAuto-labeling clusters...")
        discovery.auto_label_clusters()
        
        # Show cluster summaries
        print("\n" + "-" * 80)
        print("CLUSTER SUMMARIES")
        print("-" * 80)
        
        for cluster_id in range(min(5, results['n_clusters'])):
            summary = discovery.get_cluster_summary(cluster_id)
            print(f"\nCluster {cluster_id} - {summary['label']}:")
            print(f"  Size: {summary['size']} windows ({summary['percentage']:.1f}%)")
            print(f"  Top symbols: {summary['top_symbols'][:3]}")
            print(f"  Date range: {summary['date_range']['start']} to {summary['date_range']['end']}")
        
        # Generate visualizations
        print("\n" + "-" * 80)
        print("GENERATING VISUALIZATIONS")
        print("-" * 80)
        
        print("\nCreating visualizations...")
        
        # Create docs directory
        import os
        os.makedirs('docs', exist_ok=True)
        
        # Generate all plots
        discovery.plot_elbow_method(max_k=15, save_path='docs/elbow_method_demo.png')
        discovery.plot_cluster_distribution(save_path='docs/cluster_distribution_demo.png')
        discovery.visualize_clusters(save_path='docs/cluster_visualization_demo.png')
        discovery.plot_top_symbols_per_cluster(save_path='docs/top_symbols_demo.png')
        discovery.generate_report(report_path='docs/clustering_report_demo.txt')
        
        print("\n✓ All visualizations saved to docs/ folder:")
        print("  - elbow_method_demo.png")
        print("  - cluster_distribution_demo.png")
        print("  - cluster_visualization_demo.png")
        print("  - top_symbols_demo.png")
        print("  - clustering_report_demo.txt")
        
        # Export to pattern library
        print("\n" + "-" * 80)
        print("EXPORTING DISCOVERED PATTERNS")
        print("-" * 80)
        
        library = PatternLibrary(storage)
        exported = discovery.export_to_pattern_library(
            library,
            cluster_ids=range(min(5, results['n_clusters'])),
            use_labels=True,
            days_tolerance=7
        )
        
        print(f"\n✓ Exported {len(exported)} discovered patterns:")
        for pattern in exported:
            print(f"  - {pattern}")
        
        # Show statistics
        print("\n" + "-" * 80)
        print("CLUSTERING STATISTICS")
        print("-" * 80)
        
        stats = discovery.get_statistics()
        print(f"\nOverall statistics:")
        print(f"  Total windows analyzed: {stats['n_embeddings']:,}")
        print(f"  Clusters discovered: {stats['n_clusters']}")
        print(f"  Average cluster size: {stats['average_cluster_size']:.0f}")
        print(f"  Cluster size range: {stats['min_cluster_size']} - {stats['max_cluster_size']}")
        if stats.get('silhouette_score'):
            silhouette = stats['silhouette_score']
            if silhouette > 0.5:
                quality = "Good"
            elif silhouette > 0.25:
                quality = "Fair"
            else:
                quality = "Poor"
            print(f"  Clustering quality: {quality} (silhouette={silhouette:.3f})")
        
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
    
    if len(sys.argv) > 1:
        if sys.argv[1] == '--demo':
            demo_pattern_discovery()
        elif sys.argv[1] == '--test':
            test_pattern_discovery()
        elif sys.argv[1] == '--analyze':
            # Quick analysis
            print("\nRunning quick analysis...")
            storage = StorageManager(
                duckdb_path="storage/financedb.duckdb",
                qdrant_path="storage/qdrant_data"
            )
            discovery = PatternDiscovery(storage)
            discovery.load_embeddings(sample_size=1000)
            optimal = discovery.find_optimal_k(max_k=12)
            print(f"\nOptimal k analysis:")
            print(f"  Elbow method: k={optimal['elbow_k']}")
            if optimal['silhouette_k']:
                print(f"  Silhouette method: k={optimal['silhouette_k']}")
            if optimal['davies_k']:
                print(f"  Davies-Bouldin method: k={optimal['davies_k']}")
            storage.close()
        else:
            print("Usage:")
            print("  python clustering.py           # Run unit tests")
            print("  python clustering.py --demo    # Run interactive demo")
            print("  python clustering.py --test    # Unit tests")
            print("  python clustering.py --analyze # Quick analysis")
    else:
        test_pattern_discovery()