"""
FinanceDB - Anomaly Detection Module (Production Ready)
Detect unusual market behavior using reconstruction error from compressed embeddings.

Author: FinanceDB Project
Module: 5 - Semantic Retrieval (Anomaly Detection)
"""

import sys
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
from datetime import datetime
import numpy as np
import pandas as pd
import pickle
import warnings
from tqdm import tqdm
from loguru import logger

# Set matplotlib to use non-interactive backend to avoid Tkinter issues
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

# Suppress warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from storage.storage_manager import StorageManager
from models.compression_model import CompressionModel


class AnomalyDetector:
    """
    Detect anomalous market behavior using reconstruction error.
    
    Principle: Normal patterns compress well (low reconstruction error).
               Anomalous patterns compress poorly (high reconstruction error).
    
    Features:
    - Score individual windows (0-1 anomaly score)
    - Detect anomalies across entire database
    - Explain which features contribute to anomaly
    - Visualize anomaly timelines and feature errors
    - Save/load calibration for production use
    """
    
    # Feature names (from feature_extractor.py)
    FEATURE_NAMES = [
        'returns', 'volatility', 'volume_change', 'hl_spread', 'close_position',
        'rsi', 'macd', 'macd_signal', 'macd_histogram',
        'bb_upper', 'bb_lower', 'bb_width', 'bb_position',
        'momentum_5d', 'momentum_10d', 'momentum_20d',
        'volume_momentum', 'sma_ratio', 'ema_ratio', 'atr'
    ]
    
    def __init__(
        self,
        storage_manager: StorageManager,
        compression_model: CompressionModel,
        device: str = "cpu",
        auto_calibrate: bool = False,
        calibration_path: str = "configs/anomaly_calibration.yaml"
    ):
        """
        Initialize anomaly detector.
        
        Args:
            storage_manager: StorageManager instance
            compression_model: Trained compression model
            device: Device for model inference ('cpu' or 'cuda')
            auto_calibrate: If True, try to load calibration or calibrate
            calibration_path: Path to calibration file
        """
        self.storage = storage_manager
        self.model = compression_model
        self.device = device
        self.calibration_path = calibration_path
        
        # Setup model
        self.model.eval()
        self.model.to(device)
        
        # Normalization statistics (computed on first use)
        self.error_mean = None
        self.error_std = None
        self.is_calibrated = False
        
        # Try to load calibration if requested
        if auto_calibrate:
            if not self.load_calibration():
                logger.info("Calibration not found, will calibrate on first use")
        
        logger.info("Initialized AnomalyDetector")
        logger.info(f"  Device: {device}")
        logger.info(f"  Model: {compression_model.__class__.__name__}")
        logger.info(f"  Calibrated: {self.is_calibrated}")
    
    def _bytes_to_numpy(self, embedding_data: Any) -> Optional[np.ndarray]:
        """
        Convert embedding data from bytes or other formats to numpy array.
        
        Args:
            embedding_data: Embedding data in various formats
            
        Returns:
            Numpy array or None if conversion fails
        """
        try:
            if embedding_data is None:
                return None
            
            # Handle bytes (pickled)
            if isinstance(embedding_data, bytes):
                unpickled = pickle.loads(embedding_data)
                if isinstance(unpickled, np.ndarray):
                    return unpickled.astype(np.float32)
                elif isinstance(unpickled, (list, tuple)):
                    return np.array(unpickled, dtype=np.float32)
                else:
                    logger.debug(f"Unpickled type not array: {type(unpickled)}")
                    return None
            
            # Handle numpy array
            elif isinstance(embedding_data, np.ndarray):
                return embedding_data.astype(np.float32)
            
            # Handle list/tuple
            elif isinstance(embedding_data, (list, tuple)):
                return np.array(embedding_data, dtype=np.float32)
            
            # Handle string (JSON or literal)
            elif isinstance(embedding_data, str):
                try:
                    # Try JSON first
                    import json as json_module
                    data = json_module.loads(embedding_data)
                    return np.array(data, dtype=np.float32)
                except:
                    # Try literal evaluation
                    import ast
                    data = ast.literal_eval(embedding_data)
                    return np.array(data, dtype=np.float32)
            
            else:
                logger.debug(f"Unknown embedding type: {type(embedding_data)}")
                return None
                
        except Exception as e:
            logger.debug(f"Error converting embedding: {e}")
            return None
    
    def score(self, embedding: Any) -> float:
        """
        Compute anomaly score for a single embedding.
        
        Args:
            embedding: Embedding vector in any format (bytes, numpy, list, etc.)
            
        Returns:
            Anomaly score (0-1, higher = more anomalous)
            
        Note:
            Requires calibration (call calibrate() first)
        """
        try:
            # Convert embedding to numpy
            embedding_np = self._bytes_to_numpy(embedding)
            if embedding_np is None:
                return 0.0
            
            # Decode embedding
            import torch
            with torch.no_grad():
                emb_tensor = torch.FloatTensor(embedding_np).unsqueeze(0).to(self.device)
                reconstructed = self.model.decode(emb_tensor)
            
            # Re-encode the reconstruction and measure embedding distance
            with torch.no_grad():
                re_encoded = self.model.encode(reconstructed)
            
            # Compute MSE between original and re-encoded embedding
            mse = torch.mean((emb_tensor - re_encoded) ** 2).item()
            
            # Normalize to 0-1 score
            if self.is_calibrated:
                # Z-score normalization
                z_score = (mse - self.error_mean) / (self.error_std + 1e-8)
                # Convert to 0-1 (using sigmoid)
                score = 1.0 / (1.0 + np.exp(-z_score))
            else:
                # Without calibration, just return raw MSE
                score = min(1.0, mse)
            
            return float(score)
            
        except Exception as e:
            logger.error(f"Anomaly scoring failed: {e}")
            return 0.0
    
    def score_with_features(
        self,
        embedding: Any,
        original_features: Optional[np.ndarray] = None
    ) -> Tuple[float, Optional[np.ndarray]]:
        """
        Compute anomaly score with per-feature error breakdown.
        
        Args:
            embedding: Embedding vector in any format
            original_features: Optional original features [20, 20]
            
        Returns:
            Tuple of (overall_score, feature_errors [20])
            
        Note:
            If original_features not provided, returns None for feature_errors
        """
        try:
            # Convert embedding to numpy
            embedding_np = self._bytes_to_numpy(embedding)
            if embedding_np is None:
                return 0.0, None
            
            # Decode embedding
            import torch
            with torch.no_grad():
                emb_tensor = torch.FloatTensor(embedding_np).unsqueeze(0).to(self.device)
                reconstructed = self.model.decode(emb_tensor)
                reconstructed_np = reconstructed.cpu().numpy().squeeze()
            
            if original_features is None:
                # Without original features, use embedding distance
                overall_score = self.score(embedding_np)
                return overall_score, None
            
            # Compute per-feature MSE
            feature_errors = np.mean((reconstructed_np - original_features) ** 2, axis=0)
            
            # Overall score (mean of feature errors)
            overall_mse = np.mean(feature_errors)
            
            # Normalize to 0-1
            if self.is_calibrated:
                z_score = (overall_mse - self.error_mean) / (self.error_std + 1e-8)
                overall_score = 1.0 / (1.0 + np.exp(-z_score))
            else:
                overall_score = min(1.0, overall_mse)
            
            return float(overall_score), feature_errors
            
        except Exception as e:
            logger.error(f"Feature scoring failed: {e}")
            return 0.0, None
    
    def calibrate(self, sample_size: int = 1000, force: bool = False) -> bool:
        """
        Calibrate anomaly scores using a random sample of windows.
        
        Args:
            sample_size: Number of windows to sample for calibration
            force: If True, recalibrate even if already calibrated
            
        Returns:
            True if calibration successful
        """
        try:
            if self.is_calibrated and not force:
                logger.info("Already calibrated, use force=True to recalibrate")
                return True
            
            logger.info(f"Calibrating anomaly detector on {sample_size} samples...")
            
            # Get random sample of windows
            total_windows = self.storage.duckdb.count()
            sample_size = min(sample_size, total_windows)
            
            # Query random windows
            sql = f"SELECT window_id, embedding FROM timeseries ORDER BY RANDOM() LIMIT {sample_size}"
            results = self.storage.duckdb.query(sql)
            
            # Compute reconstruction errors
            errors = []
            
            for result in tqdm(results, desc="Calibrating"):
                embedding_data = result['embedding']
                
                # Convert to numpy
                embedding_np = self._bytes_to_numpy(embedding_data)
                if embedding_np is None:
                    continue
                
                # Decode and re-encode
                import torch
                with torch.no_grad():
                    emb_tensor = torch.FloatTensor(embedding_np).unsqueeze(0).to(self.device)
                    reconstructed = self.model.decode(emb_tensor)
                    re_encoded = self.model.encode(reconstructed)
                    
                    mse = torch.mean((emb_tensor - re_encoded) ** 2).item()
                    errors.append(mse)
            
            # Check if we got enough samples
            if len(errors) < 10:
                logger.warning(f"Only {len(errors)} valid embeddings found, calibration may be unreliable")
                return False
            
            # Compute normalization statistics
            self.error_mean = np.mean(errors) if errors else 0.1
            self.error_std = np.std(errors) if errors else 0.05
            self.is_calibrated = True
            
            # Save calibration
            self.save_calibration()
            
            logger.info(f"✓ Calibration complete")
            logger.info(f"  Mean error: {self.error_mean:.6f}")
            logger.info(f"  Std error: {self.error_std:.6f}")
            logger.info(f"  Samples used: {len(errors)}/{sample_size}")
            
            return True
            
        except Exception as e:
            logger.error(f"Calibration failed: {e}")
            # Set default values if calibration fails
            self.error_mean = 0.1
            self.error_std = 0.05
            self.is_calibrated = True
            logger.warning("Using default calibration values")
            return False
    
    def detect_anomalies(
        self,
        threshold: float = 0.95,
        limit: Optional[int] = None,
        symbol: Optional[str] = None
    ) -> List[Dict]:
        """
        Scan windows and detect anomalies.
        
        Args:
            threshold: Minimum anomaly score (0-1)
            limit: Optional limit on number of results
            symbol: Optional symbol filter
            
        Returns:
            List of anomalous windows with scores
        """
        try:
            if not self.is_calibrated:
                logger.warning("Detector not calibrated, calibrating now...")
                if not self.calibrate(sample_size=1000):
                    logger.warning("Calibration failed, using default values")
            
            logger.info(f"Scanning for anomalies (threshold={threshold})...")
            
            # Build SQL query
            if symbol:
                sql = f"SELECT window_id, symbol, timestamp, embedding FROM timeseries WHERE symbol = '{symbol}'"
            else:
                sql = "SELECT window_id, symbol, timestamp, embedding FROM timeseries"
            
            results = self.storage.duckdb.query(sql)
            
            anomalies = []
            
            for result in tqdm(results, desc="Scanning"):
                window_id = result['window_id']
                symbol = result['symbol']
                timestamp = result['timestamp']
                embedding_data = result['embedding']
                
                # Convert to numpy
                embedding_np = self._bytes_to_numpy(embedding_data)
                if embedding_np is None:
                    continue
                
                # Compute anomaly score
                score = self.score(embedding_np)
                
                # Check threshold
                if score >= threshold:
                    anomalies.append({
                        'window_id': window_id,
                        'symbol': symbol,
                        'timestamp': timestamp,
                        'anomaly_score': float(score),
                        'embedding': embedding_np
                    })
            
            # Sort by score (descending)
            anomalies.sort(key=lambda x: x['anomaly_score'], reverse=True)
            
            # Apply limit
            if limit is not None:
                anomalies = anomalies[:limit]
            
            logger.info(f"Found {len(anomalies)} anomalies (threshold={threshold})")
            
            return anomalies
            
        except Exception as e:
            logger.error(f"Anomaly detection failed: {e}")
            return []
    
    def explain_anomaly(self, window_id: str) -> Dict[str, Any]:
        """
        Explain why a window is anomalous by showing feature-level errors.
        
        Args:
            window_id: Window identifier
            
        Returns:
            Dictionary with explanation
        """
        try:
            # Get window data
            window = self.storage.read_by_id(window_id)
            
            if not window:
                logger.warning(f"Window {window_id} not found")
                return {'error': 'Window not found'}
            
            embedding_data = window['embedding']
            
            # Convert to numpy
            embedding_np = self._bytes_to_numpy(embedding_data)
            if embedding_np is None:
                return {'error': 'Invalid embedding format'}
            
            # Decode to get reconstruction
            import torch
            with torch.no_grad():
                emb_tensor = torch.FloatTensor(embedding_np).unsqueeze(0).to(self.device)
                reconstructed = self.model.decode(emb_tensor)
                reconstructed_np = reconstructed.cpu().numpy().squeeze()
            
            # Without original features, we can only analyze reconstruction variance
            # Compute variance across timesteps for each feature
            feature_variances = np.var(reconstructed_np, axis=0)
            
            # Create feature error dict
            feature_errors = {}
            for i, feature_name in enumerate(self.FEATURE_NAMES):
                if i < len(feature_variances):
                    feature_errors[feature_name] = float(feature_variances[i])
            
            # Get top contributing features
            sorted_features = sorted(
                feature_errors.items(),
                key=lambda x: x[1],
                reverse=True
            )
            
            # Overall score
            overall_score = self.score(embedding_np)
            
            explanation = {
                'window_id': window_id,
                'symbol': window['symbol'],
                'timestamp': str(window['timestamp']),
                'overall_score': float(overall_score),
                'feature_errors': feature_errors,
                'top_features': sorted_features[:5],  # Top 5 contributors
                'feature_count': len(feature_errors)
            }
            
            logger.info(f"Explained anomaly for window {window_id}")
            
            return explanation
            
        except Exception as e:
            logger.error(f"Anomaly explanation failed: {e}")
            return {'error': str(e)}
    
    def save_calibration(self, path: Optional[str] = None) -> bool:
        """
        Save calibration statistics to disk.
        
        Args:
            path: Optional custom path, uses default if None
            
        Returns:
            True if save successful
        """
        try:
            # Try to import yaml
            try:
                import yaml
            except ImportError:
                logger.error("PyYAML not installed. Install with: pip install pyyaml")
                return False
            
            if path is None:
                path = self.calibration_path
            
            # Create directory if needed
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            
            data = {
                'error_mean': float(self.error_mean) if self.error_mean is not None else 0.0,
                'error_std': float(self.error_std) if self.error_std is not None else 0.0,
                'is_calibrated': self.is_calibrated,
                'calibrated_at': datetime.now().isoformat()
            }
            
            with open(path, 'w') as f:
                yaml.dump(data, f, default_flow_style=False)
            
            logger.info(f"Saved calibration to {path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save calibration: {e}")
            return False
    
    def load_calibration(self, path: Optional[str] = None) -> bool:
        """
        Load calibration statistics from disk.
        
        Args:
            path: Optional custom path, uses default if None
            
        Returns:
            True if load successful
        """
        try:
            # Try to import yaml
            try:
                import yaml
            except ImportError:
                logger.error("PyYAML not installed. Install with: pip install pyyaml")
                return False
            
            if path is None:
                path = self.calibration_path
            
            calibration_file = Path(path)
            if not calibration_file.exists():
                logger.warning(f"Calibration file {path} not found")
                return False
            
            with open(path, 'r') as f:
                data = yaml.safe_load(f)
            
            self.error_mean = data['error_mean']
            self.error_std = data['error_std']
            self.is_calibrated = data['is_calibrated']
            
            logger.info(f"Loaded calibration from {path}")
            logger.info(f"  Mean error: {self.error_mean:.6f}")
            logger.info(f"  Std error: {self.error_std:.6f}")
            logger.info(f"  Calibrated at: {data.get('calibrated_at', 'Unknown')}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to load calibration: {e}")
            return False
    
    # ========================================================================
    # VISUALIZATION METHODS
    # ========================================================================
    
    def plot_anomaly_timeline(
        self,
        symbol: str,
        figsize: Tuple[int, int] = (15, 6),
        save_path: Optional[str] = None,
        show_plot: bool = False
    ) -> bool:
        """
        Plot timeline of anomaly scores for a symbol.
        
        Args:
            symbol: Stock ticker symbol
            figsize: Figure size
            save_path: Optional path to save figure
            show_plot: Whether to display the plot
            
        Returns:
            True if plot generated successfully
        """
        try:
            logger.info(f"Generating anomaly timeline for {symbol}")
            
            # Get all windows for symbol
            windows = self.storage.read_by_symbol(symbol)
            
            if not windows:
                logger.warning(f"No windows found for {symbol}")
                return False
            
            # Compute anomaly scores
            timestamps = []
            scores = []
            
            for window in tqdm(windows, desc="Computing scores"):
                timestamps.append(pd.to_datetime(window['timestamp']))
                embedding_np = self._bytes_to_numpy(window['embedding'])
                if embedding_np is not None:
                    score = self.score(embedding_np)
                    scores.append(score)
                else:
                    scores.append(0.0)
            
            # Create plot
            fig, ax = plt.subplots(figsize=figsize)
            
            # Plot timeline
            ax.plot(timestamps, scores, linewidth=1.5, color='steelblue', alpha=0.7)
            ax.fill_between(timestamps, scores, alpha=0.3, color='steelblue')
            
            # Highlight anomalies (score > 0.95)
            anomaly_threshold = 0.95
            anomaly_mask = np.array(scores) > anomaly_threshold
            
            if anomaly_mask.any():
                ax.scatter(
                    np.array(timestamps)[anomaly_mask],
                    np.array(scores)[anomaly_mask],
                    color='red',
                    s=100,
                    marker='o',
                    zorder=5,
                    label=f'Anomalies (>{anomaly_threshold})'
                )
            
            # Styling
            ax.axhline(y=anomaly_threshold, color='red', linestyle='--', 
                      linewidth=1, alpha=0.5, label=f'Threshold ({anomaly_threshold})')
            ax.set_xlabel('Date', fontsize=12)
            ax.set_ylabel('Anomaly Score', fontsize=12)
            ax.set_title(f'Anomaly Timeline - {symbol}', fontsize=14, fontweight='bold')
            ax.set_ylim([0, 1])
            ax.grid(True, alpha=0.3)
            ax.legend()
            
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            
            # Save if requested
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Saved anomaly timeline to {save_path}")
            
            # Show if requested
            if show_plot:
                plt.show()
            else:
                plt.close()
            
            return True
            
        except Exception as e:
            logger.error(f"Anomaly timeline plotting failed: {e}")
            return False
    
    def plot_anomaly_features(
        self,
        window_id: str,
        figsize: Tuple[int, int] = (12, 6),
        save_path: Optional[str] = None,
        show_plot: bool = False
    ) -> bool:
        """
        Plot feature-level error contributions for an anomaly.
        
        Args:
            window_id: Window identifier
            figsize: Figure size
            save_path: Optional path to save figure
            show_plot: Whether to display the plot
            
        Returns:
            True if plot generated successfully
        """
        try:
            logger.info(f"Plotting anomaly features for {window_id}")
            
            # Get explanation
            explanation = self.explain_anomaly(window_id)
            
            if 'error' in explanation:
                logger.warning(f"Cannot plot: {explanation['error']}")
                return False
            
            # Extract feature errors
            feature_errors = explanation['feature_errors']
            
            # Sort by error
            sorted_items = sorted(feature_errors.items(), key=lambda x: x[1], reverse=True)
            features = [item[0] for item in sorted_items]
            errors = [item[1] for item in sorted_items]
            
            # Create plot
            fig, ax = plt.subplots(figsize=figsize)
            
            # Color map (red for high error, green for low)
            colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(features)))
            
            # Bar chart
            bars = ax.barh(range(len(features)), errors, color=colors, 
                          alpha=0.7, edgecolor='black', linewidth=0.5)
            
            # Styling
            ax.set_yticks(range(len(features)))
            ax.set_yticklabels(features, fontsize=10)
            ax.set_xlabel('Error Contribution (Variance)', fontsize=12)
            ax.set_title(
                f"Anomaly Feature Analysis\n{explanation['symbol']} @ {explanation['timestamp']}\n"
                f"Overall Score: {explanation['overall_score']:.3f}",
                fontsize=14,
                fontweight='bold'
            )
            ax.grid(True, alpha=0.3, axis='x')
            
            # Add value labels
            for i, (bar, error) in enumerate(zip(bars, errors)):
                ax.text(
                    error + max(errors) * 0.01,
                    bar.get_y() + bar.get_height() / 2,
                    f'{error:.4f}',
                    va='center',
                    fontsize=8
                )
            
            plt.tight_layout()
            
            # Save if requested
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Saved feature analysis to {save_path}")
            
            # Show if requested
            if show_plot:
                plt.show()
            else:
                plt.close()
            
            return True
            
        except Exception as e:
            logger.error(f"Feature plotting failed: {e}")
            return False
    
    def analyze_anomaly_clusters(
        self,
        anomalies: List[Dict],
        n_clusters: int = 5,
        save_path: Optional[str] = None,
        show_plot: bool = False
    ) -> Dict:
        """
        Cluster anomalies to find common anomaly patterns.
        
        Args:
            anomalies: List of anomaly dictionaries
            n_clusters: Number of clusters to find
            save_path: Optional path to save visualization
            show_plot: Whether to display the plot
            
        Returns:
            Dictionary with cluster statistics
        """
        try:
            if not anomalies:
                logger.warning("No anomalies to cluster")
                return {}
            
            logger.info(f"Clustering {len(anomalies)} anomalies...")
            
            # Extract embeddings
            embeddings = [anomaly['embedding'] for anomaly in anomalies]
            
            # Convert to numpy array
            embedding_matrix = np.vstack(embeddings)
            
            # Use simple KMeans clustering
            try:
                from sklearn.cluster import KMeans
                from sklearn.decomposition import PCA
                from collections import Counter
            except ImportError:
                logger.error("scikit-learn not installed. Install with: pip install scikit-learn")
                return {}
            
            kmeans = KMeans(n_clusters=min(n_clusters, len(anomalies)), random_state=42)
            clusters = kmeans.fit_predict(embedding_matrix)
            
            # Assign clusters to anomalies
            for i, anomaly in enumerate(anomalies):
                anomaly['cluster'] = int(clusters[i])
            
            # Analyze clusters
            cluster_stats = {}
            for cluster_id in range(min(n_clusters, len(anomalies))):
                cluster_anomalies = [a for a, c in zip(anomalies, clusters) if c == cluster_id]
                cluster_scores = [a['anomaly_score'] for a in cluster_anomalies]
                cluster_symbols = [a['symbol'] for a in cluster_anomalies]
                
                # Find most common symbol in cluster
                symbol_counts = Counter(cluster_symbols)
                most_common_symbol = symbol_counts.most_common(1)[0][0]
                
                cluster_stats[cluster_id] = {
                    'size': len(cluster_anomalies),
                    'avg_score': float(np.mean(cluster_scores)),
                    'max_score': float(np.max(cluster_scores)),
                    'most_common_symbol': most_common_symbol,
                    'symbol_distribution': dict(symbol_counts)
                }
            
            # Print cluster statistics
            print(f"\nAnomaly Clusters Analysis:")
            print("=" * 80)
            for cluster_id, stats in cluster_stats.items():
                print(f"\nCluster {cluster_id}:")
                print(f"  Size: {stats['size']} anomalies")
                print(f"  Avg Score: {stats['avg_score']:.4f}")
                print(f"  Max Score: {stats['max_score']:.4f}")
                print(f"  Most Common Symbol: {stats['most_common_symbol']}")
                print(f"  Top 3 Symbols: {list(stats['symbol_distribution'].items())[:3]}")
            
            # Create visualization if requested and we have enough dimensions
            if save_path and embedding_matrix.shape[1] >= 2:
                # Reduce dimensionality for visualization
                pca = PCA(n_components=2)
                embedding_2d = pca.fit_transform(embedding_matrix)
                
                fig, ax = plt.subplots(figsize=(10, 8))
                
                # Scatter plot
                scatter = ax.scatter(
                    embedding_2d[:, 0],
                    embedding_2d[:, 1],
                    c=clusters,
                    cmap='tab10',
                    s=100,
                    alpha=0.7,
                    edgecolors='black',
                    linewidth=0.5
                )
                
                # Styling
                ax.set_xlabel('PCA Component 1', fontsize=12)
                ax.set_ylabel('PCA Component 2', fontsize=12)
                ax.set_title(f'Anomaly Clusters (n={len(anomalies)})', fontsize=14, fontweight='bold')
                ax.grid(True, alpha=0.3)
                
                # Add legend
                legend_elements = []
                for cluster_id, stats in cluster_stats.items():
                    legend_elements.append(
                        plt.Line2D([0], [0], marker='o', color='w', 
                                  label=f'Cluster {cluster_id}: {stats["size"]} anomalies',
                                  markerfacecolor=scatter.cmap(cluster_id / max(1, len(cluster_stats)-1)),
                                  markersize=10)
                    )
                ax.legend(handles=legend_elements, title='Clusters', loc='best')
                
                plt.tight_layout()
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Saved cluster analysis to {save_path}")
                
                if show_plot:
                    plt.show()
                else:
                    plt.close()
            
            return cluster_stats
            
        except Exception as e:
            logger.error(f"Clustering failed: {e}")
            return {}
    
    def generate_report(
        self,
        anomalies: List[Dict],
        output_dir: str = "reports",
        symbol: Optional[str] = None
    ) -> bool:
        """
        Generate comprehensive anomaly report with visualizations.
        
        Args:
            anomalies: List of anomalies to report on
            output_dir: Directory to save report
            symbol: Optional symbol for timeline plot
            
        Returns:
            True if report generated successfully
        """
        try:
            # Create output directory
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            logger.info(f"Generating anomaly report in {output_dir}")
            
            # Generate report file
            report_path = output_path / "anomaly_report.md"
            with open(report_path, 'w') as f:
                f.write("# FinanceDB Anomaly Detection Report\n\n")
                f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                
                f.write("## Summary\n\n")
                f.write(f"- **Total Anomalies Found**: {len(anomalies)}\n")
                if anomalies:
                    f.write(f"- **Highest Score**: {anomalies[0]['anomaly_score']:.4f}\n")
                    f.write(f"- **Average Score**: {np.mean([a['anomaly_score'] for a in anomalies]):.4f}\n")
                
                f.write("\n## Top Anomalies\n\n")
                f.write("| Rank | Symbol | Timestamp | Score |\n")
                f.write("|------|--------|-----------|-------|\n")
                for i, anomaly in enumerate(anomalies[:20], 1):
                    f.write(f"| {i} | {anomaly['symbol']} | {anomaly['timestamp']} | {anomaly['anomaly_score']:.4f} |\n")
                
                # Add symbol distribution
                if anomalies:
                    from collections import Counter
                    symbol_counts = Counter([a['symbol'] for a in anomalies])
                    f.write("\n## Symbol Distribution\n\n")
                    f.write("| Symbol | Count | Percentage |\n")
                    f.write("|--------|-------|------------|\n")
                    for symbol, count in symbol_counts.most_common():
                        percentage = (count / len(anomalies)) * 100
                        f.write(f"| {symbol} | {count} | {percentage:.1f}% |\n")
            
            logger.info(f"Report saved to {report_path}")
            
            # Generate visualizations if we have anomalies
            if anomalies:
                # Feature analysis for top anomaly
                top_anomaly = anomalies[0]
                self.plot_anomaly_features(
                    window_id=top_anomaly['window_id'],
                    save_path=str(output_path / "top_anomaly_features.png"),
                    show_plot=False
                )
                
                # Cluster analysis
                self.analyze_anomaly_clusters(
                    anomalies=anomalies,
                    n_clusters=min(5, len(anomalies)),
                    save_path=str(output_path / "anomaly_clusters.png"),
                    show_plot=False
                )
                
                # Timeline for most common symbol
                if symbol:
                    self.plot_anomaly_timeline(
                        symbol=symbol,
                        save_path=str(output_path / f"{symbol}_timeline.png"),
                        show_plot=False
                    )
            
            return True
            
        except Exception as e:
            logger.error(f"Report generation failed: {e}")
            return False


# ============================================================================
# MAIN ENTRY POINT (FIXED)
# ============================================================================

def main():
    """Main entry point for anomaly detection."""
    import argparse
    
    parser = argparse.ArgumentParser(description="FinanceDB Anomaly Detection")
    parser.add_argument("--demo", action="store_true", help="Run interactive demo")
    parser.add_argument("--calibrate", action="store_true", help="Calibrate detector")
    parser.add_argument("--detect", action="store_true", help="Detect anomalies")
    parser.add_argument("--symbol", type=str, help="Symbol to analyze")
    parser.add_argument("--threshold", type=float, default=0.9, help="Anomaly threshold")
    parser.add_argument("--limit", type=int, default=50, help="Maximum anomalies to show")
    parser.add_argument("--report", action="store_true", help="Generate report")
    parser.add_argument("--timeline", action="store_true", help="Generate timeline plot")
    
    args = parser.parse_args()
    
    try:
        print("=" * 80)
        print("FinanceDB Anomaly Detection")
        print("=" * 80)
        
        # Initialize storage manager once
        storage = StorageManager(
            duckdb_path="storage/financedb.duckdb",
            qdrant_path="storage/qdrant_data"
        )
        
        model_path = Path("models/checkpoints/best_model.pt")
        if not model_path.exists():
            print("✗ Model not found. Please train the model first.")
            storage.close()
            return
        
        model, _ = CompressionModel.load_checkpoint(str(model_path))
        
        detector = AnomalyDetector(
            storage_manager=storage,
            compression_model=model,
            device='cpu',
            auto_calibrate=True
        )
        
        print("✓ AnomalyDetector initialized")
        
        # Run different modes based on arguments
        if args.calibrate:
            print("\n" + "=" * 80)
            print("Calibrating Detector")
            print("=" * 80)
            detector.calibrate(sample_size=1000, force=True)
        
        elif args.detect:
            print("\n" + "=" * 80)
            print(f"Detecting Anomalies (threshold={args.threshold})")
            print("=" * 80)
            
            anomalies = detector.detect_anomalies(
                threshold=args.threshold,
                limit=args.limit,
                symbol=args.symbol
            )
            
            print(f"\nFound {len(anomalies)} anomalies:")
            print("-" * 80)
            for i, anomaly in enumerate(anomalies[:10], 1):
                print(f"{i:2d}. {anomaly['symbol']:6s} @ {anomaly['timestamp']} | "
                      f"Score: {anomaly['anomaly_score']:.4f}")
            
            if anomalies and args.report:
                detector.generate_report(
                    anomalies=anomalies,
                    output_dir="reports",
                    symbol=args.symbol or anomalies[0]['symbol']
                )
        
        elif args.timeline:
            if not args.symbol:
                print("Error: --timeline requires --symbol")
                storage.close()
                return
            
            print("\n" + "=" * 80)
            print(f"Generating Timeline for {args.symbol}")
            print("=" * 80)
            
            detector.plot_anomaly_timeline(
                symbol=args.symbol,
                save_path=f"reports/{args.symbol}_timeline.png",
                show_plot=True
            )
        
        elif args.demo:
            # Interactive demo mode
            print("\n" + "=" * 80)
            print("Interactive Demo")
            print("=" * 80)
            
            # Calibrate if needed
            if not detector.is_calibrated:
                print("\nCalibrating detector...")
                detector.calibrate(sample_size=1000)
            
            # Detect anomalies
            print("\nDetecting anomalies...")
            anomalies = detector.detect_anomalies(threshold=0.9, limit=50)
            
            if anomalies:
                print(f"\nFound {len(anomalies)} anomalies:")
                for i, anomaly in enumerate(anomalies[:5], 1):
                    print(f"{i}. {anomaly['symbol']} @ {anomaly['timestamp']} - Score: {anomaly['anomaly_score']:.4f}")
                
                # Explain top anomaly
                print("\n" + "-" * 80)
                print("Explaining Top Anomaly")
                print("-" * 80)
                
                explanation = detector.explain_anomaly(anomalies[0]['window_id'])
                if 'error' not in explanation:
                    print(f"\nSymbol: {explanation['symbol']}")
                    print(f"Date: {explanation['timestamp']}")
                    print(f"Overall Score: {explanation['overall_score']:.4f}")
                    print("\nTop 5 Contributing Features:")
                    for i, (feature, error) in enumerate(explanation['top_features'], 1):
                        print(f"  {i}. {feature:20s}: {error:.6f}")
                
                # Generate visualizations
                print("\n" + "-" * 80)
                print("Generating Visualizations")
                print("-" * 80)
                
                # Create reports directory
                Path("reports").mkdir(exist_ok=True)
                
                # Feature analysis
                print("\nGenerating feature analysis...")
                detector.plot_anomaly_features(
                    anomalies[0]['window_id'],
                    save_path="reports/top_anomaly_features.png",
                    show_plot=True
                )
                
                # Cluster analysis
                if len(anomalies) >= 5:
                    print("\nPerforming cluster analysis...")
                    detector.analyze_anomaly_clusters(
                        anomalies=anomalies[:20],
                        n_clusters=min(5, len(anomalies)),
                        save_path="reports/anomaly_clusters.png",
                        show_plot=True
                    )
                
                # Timeline for most common symbol
                if anomalies:
                    from collections import Counter
                    symbol_counts = Counter([a['symbol'] for a in anomalies])
                    most_common_symbol = symbol_counts.most_common(1)[0][0]
                    
                    print(f"\nGenerating timeline for {most_common_symbol}...")
                    detector.plot_anomaly_timeline(
                        symbol=most_common_symbol,
                        save_path=f"reports/{most_common_symbol}_timeline.png",
                        show_plot=True
                    )
            else:
                print("\nNo anomalies found with threshold 0.9")
        
        else:
            # Default: run simplified test without creating new storage instance
            print("\n" + "=" * 80)
            print("Running Quick Test")
            print("=" * 80)
            
            # Run quick test using existing detector
            if not detector.is_calibrated:
                print("Calibrating detector...")
                detector.calibrate(sample_size=500)
            
            # Test scoring
            sql = "SELECT window_id, symbol, timestamp, embedding FROM timeseries WHERE symbol = 'AAPL' LIMIT 1"
            test_results = storage.query_sql(sql)
            
            if test_results:
                score = detector.score(test_results[0]['embedding'])
                print(f"\nAnomaly score for AAPL: {score:.4f}")
            
            # Detect a few anomalies
            print("\nDetecting top anomalies...")
            anomalies = detector.detect_anomalies(threshold=0.9, limit=5)
            
            if anomalies:
                print(f"\nFound {len(anomalies)} anomalies:")
                for i, anomaly in enumerate(anomalies, 1):
                    print(f"{i}. {anomaly['symbol']} @ {anomaly['timestamp']} - Score: {anomaly['anomaly_score']:.4f}")
        
        # Cleanup
        storage.close()
        
        print("\n" + "=" * 80)
        print("✓ Completed successfully!")
        print("=" * 80)
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()


# ============================================================================
# RUN MAIN
# ============================================================================

if __name__ == "__main__":
    main()