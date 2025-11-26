"""
Manual Feature Engineering for Crypto Crime Detection
Implements comprehensive graph-based and behavioral features as per project specification

Experiment 1 (Elliptic): Graph-structural features
Experiment 2 (Mendeley): Behavioral/pattern features
"""

import pandas as pd
import numpy as np
import networkx as nx
from sklearn.preprocessing import StandardScaler, RobustScaler
from joblib import Parallel, delayed
import multiprocessing
import threading
import time
import sys
import warnings
warnings.filterwarnings('ignore')

# Try to import IPython display for notebook progress
try:
    from IPython.display import display, clear_output
    IPYTHON_AVAILABLE = True
except ImportError:
    IPYTHON_AVAILABLE = False

# Detect if running in Jupyter notebook
def _is_notebook():
    """Detect if code is running in a Jupyter notebook"""
    try:
        from IPython import get_ipython
        ipython = get_ipython()
        if ipython is None:
            return False
        # Check if it's a notebook (not a terminal or other frontend)
        return 'IPKernelApp' in ipython.config or 'notebook' in str(type(ipython)).lower()
    except:
        return False

IS_NOTEBOOK = _is_notebook()

# Progress tracking class that works reliably in notebooks
class NotebookProgress:
    """Progress tracker that works in Jupyter notebooks with joblib.Parallel"""
    def __init__(self, total, desc="Progress", update_interval=1.0):
        self.total = total
        self.desc = desc
        self.completed = 0
        self.start_time = time.time()
        self.lock = threading.Lock()
        self.running = True
        self.update_interval = update_interval
        self.last_display_time = time.time()
        self.use_clear_output = IS_NOTEBOOK and IPYTHON_AVAILABLE
        
        # Start update thread
        self.update_thread = threading.Thread(target=self._periodic_update, daemon=True)
        self.update_thread.start()
        
        # Initial display
        self._update_display()
    
    def _update_display(self, force=False):
        """Update the progress display"""
        if not self.running and not force:
            return
        
        now = time.time()
        if not force and (now - self.last_display_time) < self.update_interval:
            return
        
        self.last_display_time = now
        
        with self.lock:
            completed = self.completed
            elapsed = time.time() - self.start_time
            rate = completed / elapsed if elapsed > 0 else 0
            remaining = (self.total - completed) / rate if rate > 0 else 0
            pct = (completed / self.total) * 100 if self.total > 0 else 0
        
        # Format progress message
        progress_msg = (f"    {self.desc}: {completed}/{self.total} ({pct:.1f}%) | "
                       f"Elapsed: {elapsed:.1f}s | Rate: {rate:.1f}/s | "
                       f"Remaining: ~{remaining:.0f}s")
        
        # Use clear_output for notebooks, simple print for terminals
        if self.use_clear_output:
            clear_output(wait=True)
            print(progress_msg, flush=True)
        else:
            # For terminals, use simple print (no carriage return)
            print(progress_msg, flush=True)
    
    def _periodic_update(self):
        """Periodically update display in background thread"""
        while self.running:
            time.sleep(self.update_interval)
            if self.running:
                self._update_display(force=True)
    
    def update(self, n=1):
        """Update progress counter"""
        with self.lock:
            self.completed += n
            # Update display immediately if significant change (every 1%)
            if self.completed % max(1, self.total // 100) == 0:
                self._update_display(force=True)
    
    def finish(self):
        """Mark as complete"""
        self.running = False
        if self.update_thread.is_alive():
            self.update_thread.join(timeout=0.5)
        elapsed = time.time() - self.start_time
        
        # Final message
        final_msg = (f"    {self.desc}: {self.total}/{self.total} (100.0%) | "
                    f"Completed in {elapsed:.1f}s")
        
        if self.use_clear_output:
            clear_output(wait=True)
            print(final_msg, flush=True)
        else:
            print(final_msg, flush=True)


# ============================================================================
# EXPERIMENT 1: ELLIPTIC BITCOIN DATASET - GRAPH-BASED FEATURES
# ============================================================================

class EllipticFeatureEngineer:
    """
    Graph-structural feature engineering for Elliptic Bitcoin dataset.
    Computes three feature sets: Transaction-only (A), Graph-only (B), Combined (C)
    """
    
    def __init__(self, G, elliptic_features_df=None, n_jobs=-1):
        """
        Parameters:
        -----------
        G : networkx.DiGraph
            Transaction graph (nodes = transactions, edges = value transfers)
        elliptic_features_df : pd.DataFrame, optional
            Elliptic's pre-computed 166 features (used for transaction-only baseline)
        n_jobs : int, default=-1
            Number of parallel workers. -1 means use all available cores.
        """
        self.G = G
        self.elliptic_features = elliptic_features_df
        self.n_jobs = n_jobs if n_jobs > 0 else multiprocessing.cpu_count()
        print(f"  Using {self.n_jobs} parallel workers")
        self.pagerank_cache = None
        self.betweenness_cache = None
        self.closeness_cache = None
        self.clustering_cache = None
        self.communities_cache = None
        
    def compute_graph_features(self, nodes, use_sampling=True, sample_size=10000):
        """
        Compute comprehensive graph-structural features for each node.

        Core Graph Features (as per specification):
        - Degree (total, in-degree, out-degree)
        - PageRank
        - Betweenness centrality
        - Closeness centrality
        - Clustering coefficient
        - Average neighbor degree
        - Number of unique neighbors
        - Community membership (Louvain)
        """
        print("\n" + "="*70)
        print("COMPUTING GRAPH-STRUCTURAL FEATURES")
        print("="*70)
        print(f"Total nodes to process: {len(nodes):,}")
        print(f"Parallel workers: {self.n_jobs}")
        print(f"Graph size: {self.G.number_of_nodes():,} nodes, {self.G.number_of_edges():,} edges")
        print("="*70 + "\n")
        
        # Compute expensive features once on a sample if needed
        if use_sampling and len(nodes) > sample_size:
            print(f"  Using sampling for expensive computations (sample size: {sample_size})")
            sample_nodes = np.random.choice(
                list(set(nodes).intersection(self.G.nodes())), 
                size=min(sample_size, len(set(nodes).intersection(self.G.nodes()))),
                replace=False
            )
            subgraph = self.G.subgraph(sample_nodes)
        else:
            subgraph = self.G
            sample_nodes = list(set(nodes).intersection(self.G.nodes()))
        
        # Compute PageRank
        if self.pagerank_cache is None:
            print("  Computing PageRank...", end=' ', flush=True)
            self.pagerank_cache = nx.pagerank(subgraph, max_iter=50, tol=1e-4)
            print("✓")
        
        # Compute Betweenness Centrality (expensive - use sampling)
        if self.betweenness_cache is None:
            print("  Computing Betweenness Centrality (sampled)...", end=' ', flush=True)
            k = min(100, len(sample_nodes))  # Sample for betweenness
            self.betweenness_cache = nx.betweenness_centrality(subgraph, k=k, normalized=True)
            print("✓")
        
        # Compute Closeness Centrality (expensive - use batch computation on subgraph)
        if self.closeness_cache is None:
            print("  Computing Closeness Centrality (batch mode - much faster)...")
            # Use batch computation on the subgraph (MUCH faster than per-node)
            print(f"    Processing {len(sample_nodes)} nodes in batch...")
            try:
                # Batch compute on subgraph is orders of magnitude faster
                self.closeness_cache = nx.closeness_centrality(subgraph)
                print("    ✓ Closeness centrality complete")
            except Exception as e:
                print(f"    Warning: Batch closeness failed ({e}), using fallback...")
                # Fallback: assign zero to all
                self.closeness_cache = {node: 0 for node in sample_nodes}
                print("    ✓ Closeness centrality complete (fallback)")
        
        # Compute Clustering Coefficient (batch on subgraph for speed)
        if self.clustering_cache is None:
            print("  Computing Clustering Coefficients (batch mode)...", end=' ', flush=True)
            # Compute on subgraph only for speed, then fill in zeros for missing nodes
            subgraph_clustering = nx.clustering(subgraph.to_undirected())
            # Fill in zeros for nodes not in subgraph
            self.clustering_cache = {node: 0 for node in self.G.nodes()}
            self.clustering_cache.update(subgraph_clustering)
            print("✓")
        
        # Compute Communities (Louvain)
        if self.communities_cache is None:
            print("  Computing Community Structure...", end=' ', flush=True)
            try:
                import community as community_louvain
                undirected_subgraph = subgraph.to_undirected()
                self.communities_cache = community_louvain.best_partition(undirected_subgraph)
                print("✓ (Louvain)")
            except:
                # Fallback: use connected components
                print("(Using connected components as fallback)...", end=' ', flush=True)
                undirected_subgraph = subgraph.to_undirected()
                components = list(nx.connected_components(undirected_subgraph))
                self.communities_cache = {}
                for i, component in enumerate(components):
                    for node in component:
                        self.communities_cache[node] = i
                print(f"✓ ({len(components)} components)")
        
        # Compute features for each node (parallelized with MAXIMUM CPU usage)
        print("  Computing per-node features (MAXIMIZING CPU USAGE)...")
        print(f"    Processing {len(nodes):,} nodes for feature extraction...")
        print(f"    Using {self.n_jobs} parallel workers")

        def compute_node_features_batch(batch_nodes):
            """Process a batch of nodes - reduces overhead"""
            return [self._compute_node_features(node) if node in self.G else self._empty_features(node)
                    for node in batch_nodes]

        # Split nodes into batches for better CPU utilization
        batch_size = max(100, len(nodes) // (self.n_jobs * 4))  # 4 batches per worker
        node_batches = [nodes[i:i+batch_size] for i in range(0, len(nodes), batch_size)]

        print(f"    Split into {len(node_batches)} batches of ~{batch_size} nodes each")
        print(f"    Starting parallel processing with verbose output...")

        # Use verbose to see progress, loky backend for max CPU
        batched_features = Parallel(
            n_jobs=self.n_jobs,
            backend='loky',
            verbose=10,  # Show progress bar from joblib
            max_nbytes=None
        )(
            delayed(compute_node_features_batch)(batch)
            for batch in node_batches
        )

        # Flatten batched results
        features = [feat for batch in batched_features for feat in batch]

        print("    ✓ Node feature extraction complete")

        result_df = pd.DataFrame(features)

        print("\n" + "="*70)
        print("FEATURE COMPUTATION COMPLETE!")
        print("="*70)
        print(f"Generated {len(features):,} feature vectors")
        print(f"Features per node: {len(features[0])-1}")
        print(f"Total DataFrame shape: {result_df.shape}")
        print("="*70 + "\n")

        return result_df
    
    def _compute_node_features(self, node):
        """Compute all graph features for a single node (optimized)"""
        # Fast degree lookup
        in_deg = self.G.in_degree(node)
        out_deg = self.G.out_degree(node)
        total_deg = in_deg + out_deg

        # Get neighbors efficiently
        predecessors = list(self.G.predecessors(node))
        successors = list(self.G.successors(node))
        unique_neighbors = len(set(predecessors + successors))

        # Average neighbor degree (optimized)
        all_neighbors = predecessors + successors
        if all_neighbors:
            # Use list comprehension for speed
            neighbor_degrees = [self.G.degree(n) for n in all_neighbors]
            avg_neighbor_degree = sum(neighbor_degrees) / len(neighbor_degrees)  # Faster than np.mean
            max_neighbor_degree = max(neighbor_degrees)
            min_neighbor_degree = min(neighbor_degrees)
        else:
            avg_neighbor_degree = 0
            max_neighbor_degree = 0
            min_neighbor_degree = 0

        # Cached centrality measures (fast dictionary lookups)
        pagerank = self.pagerank_cache.get(node, 0)
        betweenness = self.betweenness_cache.get(node, 0)
        closeness = self.closeness_cache.get(node, 0)
        clustering = self.clustering_cache.get(node, 0)
        community_id = self.communities_cache.get(node, -1)

        # Derived features (fast arithmetic)
        degree_ratio = out_deg / (in_deg + 1)
        in_out_ratio = in_deg / (out_deg + 1)
        flow_imbalance = abs(in_deg - out_deg) / (total_deg + 1)
        
        return {
            'txId': node,
            # Core degree features
            'in_degree': in_deg,
            'out_degree': out_deg,
            'total_degree': total_deg,
            'degree_ratio': degree_ratio,
            'in_out_ratio': in_out_ratio,
            'flow_imbalance': flow_imbalance,
            
            # Neighborhood features
            'n_predecessors': len(predecessors),
            'n_successors': len(successors),
            'unique_neighbors': unique_neighbors,
            'avg_neighbor_degree': avg_neighbor_degree,
            'max_neighbor_degree': max_neighbor_degree,
            'min_neighbor_degree': min_neighbor_degree,
            
            # Centrality measures (core specification)
            'pagerank': pagerank,
            'betweenness_centrality': betweenness,
            'closeness_centrality': closeness,
            'clustering_coefficient': clustering,
            
            # Community structure
            'community_id': community_id,
            
            # Hub/Authority indicators
            'is_hub': int(out_deg > 10),  # High out-degree
            'is_authority': int(in_deg > 10),  # High in-degree
        }
    
    def _empty_features(self, node):
        """Return zero features for nodes not in graph"""
        return {
            'txId': node,
            'in_degree': 0, 'out_degree': 0, 'total_degree': 0,
            'degree_ratio': 0, 'in_out_ratio': 0, 'flow_imbalance': 0,
            'n_predecessors': 0, 'n_successors': 0, 'unique_neighbors': 0,
            'avg_neighbor_degree': 0, 'max_neighbor_degree': 0, 'min_neighbor_degree': 0,
            'pagerank': 0, 'betweenness_centrality': 0, 
            'closeness_centrality': 0, 'clustering_coefficient': 0,
            'community_id': -1, 'is_hub': 0, 'is_authority': 0
        }
    
    def get_transaction_features(self, nodes, feature_subset=None):
        """
        Extract transaction-level features from Elliptic's pre-computed features.
        Uses a small subset to avoid bloating the feature space.
        
        Parameters:
        -----------
        nodes : array-like
            Transaction IDs
        feature_subset : list, optional
            Specific columns to extract. If None, uses first 10 features.
        """
        if self.elliptic_features is None:
            print("Warning: No Elliptic features provided. Returning empty transaction features.")
            return pd.DataFrame({'txId': nodes})
        
        # First column is txId
        txid_col = self.elliptic_features.columns[0]
        
        # Select subset of features
        if feature_subset is None:
            # Use first 10 features as a representative subset
            feature_cols = self.elliptic_features.columns[1:11].tolist()
        else:
            feature_cols = feature_subset
        
        # Extract features for requested nodes
        tx_features = self.elliptic_features[self.elliptic_features[txid_col].isin(nodes)]
        tx_features = tx_features[[txid_col] + feature_cols].rename(columns={txid_col: 'txId'})
        
        # Rename columns for clarity
        tx_features.columns = ['txId'] + [f'tx_feat_{i}' for i in range(len(feature_cols))]
        
        return tx_features
    
    def create_feature_sets(self, nodes, include_transaction_features=False):
        """
        Create three feature sets as per specification:
        - Feature Set A: Transaction-only features
        - Feature Set B: Graph-only features
        - Feature Set C: Combined (transaction + graph)
        
        Returns:
        --------
        dict with keys 'A', 'B', 'C' containing respective feature dataframes
        """
        print("\n" + "="*70)
        print("CREATING FEATURE SETS")
        print("="*70)
        
        # Feature Set B: Graph-only
        print("\n[Feature Set B] Graph-structural features")
        graph_features = self.compute_graph_features(nodes)
        
        feature_sets = {'B': graph_features}
        
        if include_transaction_features and self.elliptic_features is not None:
            # Feature Set A: Transaction-only
            print("\n[Feature Set A] Transaction-only features")
            tx_features = self.get_transaction_features(nodes)
            feature_sets['A'] = tx_features
            
            # Feature Set C: Combined
            print("\n[Feature Set C] Combined (transaction + graph)")
            combined = tx_features.merge(graph_features, on='txId', how='left')
            feature_sets['C'] = combined
        else:
            print("\n[Note] Transaction features not included (use Graph-only for experiments)")
            feature_sets['A'] = pd.DataFrame({'txId': nodes})
            feature_sets['C'] = graph_features
        
        print("="*70)
        return feature_sets


# ============================================================================
# EXPERIMENT 2: MENDELEY CRYPTOCURRENCY SCAM DATASET - BEHAVIORAL FEATURES
# ============================================================================

class MendeleyFeatureEngineer:
    """
    Behavioral feature engineering for Mendeley Scam Dataset.
    Focus on transaction and wallet activity patterns (NO graph features).
    """
    
    def __init__(self, df):
        """
        Parameters:
        -----------
        df : pd.DataFrame
            Raw Mendeley dataset with columns like Transaction_Value, Wallet_Age_Days, etc.
        """
        self.df = df.copy()
        
    def engineer_features(self):
        """
        Create behavioral and pattern-based features as per specification.
        
        Core behavioral features:
        - Transaction value and fees
        - Wallet age and balance
        - Transaction velocity
        - Number of inputs/outputs
        - Gas price and exchange rate
        
        Engineered features:
        - Value/fee ratio
        - Velocity normalized by wallet age
        - Average value per input/output
        """
        print("Engineering behavioral features for Mendeley dataset...")
        
        df = self.df.copy()
        
        # Drop RL-related columns (Action, Reward, Predicted_Action)
        print("  Removing RL-related columns (Action, Reward, Predicted_Action)...")
        rl_cols = ['Action', 'Reward', 'Predicted_Action']
        df = df.drop(columns=[col for col in rl_cols if col in df.columns], errors='ignore')
        
        # Handle missing values
        print("  Handling missing values...")
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df[col].isna().sum() > 0:
                df[col].fillna(df[col].median(), inplace=True)
        
        # Engineered features
        print("  Creating engineered features...")
        
        # 1. Value/Fee Ratio (efficiency metric)
        df['value_fee_ratio'] = df['Transaction_Value'] / (df['Transaction_Fees'] + 1e-6)
        
        # 2. Velocity normalized by wallet age (activity intensity)
        df['velocity_per_day'] = df['Transaction_Velocity'] / (df['Wallet_Age_Days'] + 1)
        
        # 3. Average value per input/output
        df['avg_value_per_input'] = df['Transaction_Value'] / (df['Number_of_Inputs'] + 1)
        df['avg_value_per_output'] = df['Transaction_Value'] / (df['Number_of_Outputs'] + 1)
        
        # 4. Input/Output ratio (transaction structure)
        df['input_output_ratio'] = df['Number_of_Inputs'] / (df['Number_of_Outputs'] + 1)
        
        # 5. Fee as percentage of value
        df['fee_percentage'] = (df['Transaction_Fees'] / (df['Transaction_Value'] + 1e-6)) * 100
        
        # 6. Wallet balance to value ratio (relative transaction size)
        df['balance_value_ratio'] = df['Wallet_Balance'] / (df['Transaction_Value'] + 1)
        
        # 7. Gas price adjusted value
        if 'Gas_Price' in df.columns:
            df['gas_adjusted_value'] = df['Transaction_Value'] / (df['Gas_Price'] + 1)
        
        # 8. Total transaction activity (inputs + outputs)
        df['total_tx_activity'] = df['Number_of_Inputs'] + df['Number_of_Outputs']
        
        # 9. Velocity intensity (high velocity + young wallet = suspicious)
        df['velocity_intensity'] = df['Transaction_Velocity'] * np.log1p(1 / (df['Wallet_Age_Days'] + 1))
        
        # 10. Exchange rate adjusted value
        if 'Exchange_Rate' in df.columns:
            df['exchange_adjusted_value'] = df['Transaction_Value'] * df['Exchange_Rate']
        
        # Log transformations for skewed features
        print("  Applying log transformations...")
        skewed_features = ['Transaction_Value', 'Transaction_Fees', 'Wallet_Balance', 
                          'Transaction_Velocity', 'Wallet_Age_Days']
        for feat in skewed_features:
            if feat in df.columns:
                df[f'{feat}_log'] = np.log1p(df[feat])
        
        print(f"  Final feature count: {len(df.columns)} features")
        return df
    
    def get_feature_columns(self, include_target=False):
        """
        Get list of feature columns (excluding target variable).
        """
        exclude_cols = ['Is_Scam'] if not include_target else []
        feature_cols = [col for col in self.df.columns if col not in exclude_cols]
        return feature_cols
    
    def prepare_for_modeling(self, target_col='Is_Scam', scale=True):
        """
        Prepare features for modeling: separate X and y, optional scaling.
        
        Returns:
        --------
        X : pd.DataFrame
            Feature matrix
        y : pd.Series
            Target variable
        scaler : StandardScaler or None
            Fitted scaler if scale=True
        """
        df = self.engineer_features()
        
        # Separate features and target
        y = df[target_col]
        X = df.drop(columns=[target_col])
        
        scaler = None
        if scale:
            print("  Scaling features...")
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            X = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
        
        return X, y, scaler


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def load_elliptic_data(classes_path, edges_path, features_path=None):
    """
    Load Elliptic Bitcoin dataset.
    
    Returns:
    --------
    labeled_classes : pd.DataFrame
        Labeled transactions (illicit/licit)
    unknown_classes : pd.DataFrame
        Unknown transactions to predict
    G : networkx.DiGraph
        Transaction graph
    features_df : pd.DataFrame or None
        Pre-computed Elliptic features (if path provided)
    """
    print("Loading Elliptic dataset...")
    classes = pd.read_csv(classes_path)
    edges = pd.read_csv(edges_path)
    
    # Separate labeled and unknown
    labeled = classes[classes['class'] != 'unknown'].copy()
    labeled['label'] = (labeled['class'] == '1').astype(int)
    unknown = classes[classes['class'] == 'unknown'].copy()
    
    # Build graph
    G = nx.from_pandas_edgelist(edges, 'txId1', 'txId2', create_using=nx.DiGraph())
    
    # Load features if provided
    features_df = None
    if features_path:
        print("  Loading Elliptic pre-computed features...")
        features_df = pd.read_csv(features_path)
    
    print(f"  Labeled: {len(labeled)} (Illicit: {labeled['label'].sum()})")
    print(f"  Unknown: {len(unknown)}")
    print(f"  Graph: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")
    
    return labeled, unknown, G, features_df


def load_mendeley_data(data_path):
    """
    Load Mendeley Cryptocurrency Scam Dataset.
    
    Returns:
    --------
    df : pd.DataFrame
        Raw Mendeley dataset
    """
    print("Loading Mendeley dataset...")
    df = pd.read_csv(data_path)
    print(f"  Shape: {df.shape}")
    print(f"  Scam rate: {df['Is_Scam'].mean():.2%}")
    return df


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print(" MANUAL FEATURE ENGINEERING - CRYPTO CRIME DETECTION")
    print("="*70)
    
    # Example for Elliptic
    print("\n--- EXPERIMENT 1: ELLIPTIC BITCOIN ---")
    labeled, unknown, G, features = load_elliptic_data(
        classes_path='data/hugging/elliptic_txs_classes.csv',
        edges_path='data/hugging/elliptic_txs_edgelist.csv',
        features_path='data/hugging/elliptic_txs_features.csv'  # Optional
    )
    
    engineer = EllipticFeatureEngineer(G, features)
    feature_sets = engineer.create_feature_sets(
        nodes=labeled['txId'].values[:1000],  # Sample for demo
        include_transaction_features=False  # Set True to include Elliptic features
    )
    
    print("\nFeature Set B (Graph-only) shape:", feature_sets['B'].shape)
    print("Sample features:\n", feature_sets['B'].head())
    
    # Example for Mendeley
    print("\n--- EXPERIMENT 2: MENDELEY SCAM DETECTION ---")
    mendeley_df = load_mendeley_data(
        'data/Cryptocurrency_Scam_Dataset_for_DQN_Models/Cryptocurrency_Scam_Dataset_for_DQN_Models.csv'
    )
    
    mendeley_engineer = MendeleyFeatureEngineer(mendeley_df)
    X, y, scaler = mendeley_engineer.prepare_for_modeling()
    
    print("\nFeature matrix shape:", X.shape)
    print("Target distribution:\n", y.value_counts())
    print("\nSample engineered features:\n", X.columns.tolist()[:15])
    
    print("\n" + "="*70)
    print(" FEATURE ENGINEERING COMPLETE")
    print("="*70)
