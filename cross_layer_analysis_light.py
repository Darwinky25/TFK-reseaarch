#!/usr/bin/env python3
"""Cross-Layer Neuron Analysis - Lightweight Version"""

import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.manifold import TSNE
import logging

# Setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
MODEL_NAME = 'all-MiniLM-L6-v2'
NUM_LAYERS = 6
OUTPUT_DIR = 'cross_layer_results'
os.makedirs(OUTPUT_DIR, exist_ok=True)

class CrossLayerAnalyzer:
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = None
        self.activations = {}
        self.correlations = {}
    
    def load_model(self):
        logger.info("Loading model...")
        self.model = SentenceTransformer(MODEL_NAME).to(self.device)
        self.model._modules['0'].auto_model.config.output_hidden_states = True
    
    def get_concepts(self):
        df = pd.read_csv('tsr_final_forces.csv')
        return list(set(df['concept1'].tolist() + df['concept2'].tolist()))
    
    def extract_activations(self, concepts, max_concepts=20):
        """Extract activations for a subset of concepts"""
        logger.info(f"Extracting activations for {min(max_concepts, len(concepts))} concepts...")
        
        for concept in tqdm(concepts[:max_concepts]):
            try:
                inputs = self.model.tokenize([concept])
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                with torch.no_grad():
                    outputs = self.model._modules['0'].auto_model(**inputs, output_hidden_states=True)
                
                self.activations[concept] = [
                    h.squeeze(0).mean(0).cpu().numpy() 
                    for h in outputs.hidden_states[1:]
                ]
            except Exception as e:
                logger.warning(f"Error processing {concept}: {e}")
    
    def analyze_layers(self):
        """Analyze neuron correlations across layers"""
        logger.info("Analyzing layer correlations...")
        
        tsr_data = pd.read_csv('tsr_final_forces.csv')
        
        for layer in range(NUM_LAYERS):
            logger.info(f"\nAnalyzing layer {layer}...")
            
            # Prepare data
            X, y_inf, y_stab = [], [], []
            
            for _, row in tsr_data.iterrows():
                c1, c2 = row['concept1'], row['concept2']
                if c1 in self.activations and c2 in self.activations:
                    # Average activations for the mediator
                    mediator = (self.activations[c1][layer] + self.activations[c2][layer]) / 2
                    X.append(mediator)
                    y_inf.append(row['influence_score'])
                    y_stab.append(row['stability_score'])
            
            X = np.array(X)
            y_inf = np.array(y_inf)
            y_stab = np.array(y_stab)
            
            # Calculate correlations
            corrs = []
            for i in tqdm(range(X.shape[1]), desc=f"Layer {layer}"):
                inf_corr = np.corrcoef(X[:, i], y_inf)[0, 1]
                stab_corr = np.corrcoef(X[:, i], y_stab)[0, 1]
                corrs.append({
                    'neuron': i,
                    'layer': layer,
                    'inf_corr': inf_corr,
                    'stab_corr': stab_corr,
                    'abs_inf': abs(inf_corr),
                    'abs_stab': abs(stab_corr)
                })
            
            self.correlations[layer] = pd.DataFrame(corrs)
    
    def visualize_results(self):
        """Generate visualizations"""
        logger.info("Generating visualizations...")
        
        # Combine data from all layers
        all_data = pd.concat([df.assign(layer=layer) 
                            for layer, df in self.correlations.items()])
        
        # Plot 1: Correlation distributions by layer
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        sns.boxplot(data=all_data, x='layer', y='inf_corr')
        plt.title('Influence Correlations by Layer')
        
        plt.subplot(1, 2, 2)
        sns.boxplot(data=all_data, x='layer', y='stab_corr')
        plt.title('Stability Correlations by Layer')
        
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, 'correlations_by_layer.png'))
        
        # Plot 2: Top neurons
        top_neurons = []
        for layer in range(NUM_LAYERS):
            top_inf = self.correlations[layer].nlargest(5, 'abs_inf')
            top_stab = self.correlations[layer].nlargest(5, 'abs_stab')
            top_neurons.extend([
                {'layer': layer, 'neuron': row['neuron'], 
                 'corr': row['inf_corr'], 'type': 'influence'}
                for _, row in top_inf.iterrows()
            ])
            top_neurons.extend([
                {'layer': layer, 'neuron': row['neuron'],
                 'corr': row['stab_corr'], 'type': 'stability'}
                for _, row in top_stab.iterrows()
            ])
        
        top_df = pd.DataFrame(top_neurons)
        
        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=top_df, x='layer', y='corr', 
                       hue='type', style='type', s=100)
        plt.title('Top Specialized Neurons by Layer')
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, 'top_neurons.png'))
    
    def cluster_analysis(self):
        """Perform clustering analysis on neuron activations"""
        logger.info("Performing clustering analysis...")
        
        # Prepare data (use layer 3 as representative)
        layer = 3
        concepts = list(self.activations.keys())
        X = np.array([self.activations[c][layer] for c in concepts])
        
        # Dimensionality reduction with t-SNE
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(X)-1))
        X_2d = tsne.fit_transform(X)
        
        # K-means clustering
        kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(X_2d)
        
        # Create cluster assignments DataFrame
        cluster_df = pd.DataFrame({
            'concept': concepts,
            'cluster': clusters,
            'x': X_2d[:, 0],
            'y': X_2d[:, 1]
        })
        
        # Print cluster assignments to console
        print("\n=== Cluster Assignments ===")
        for cluster in sorted(cluster_df['cluster'].unique()):
            cluster_concepts = cluster_df[cluster_df['cluster'] == cluster]['concept'].tolist()
            print(f"\nCluster {cluster} ({len(cluster_concepts)} concepts):")
            print(", ".join(cluster_concepts))
        
        # Save cluster assignments to a string buffer instead of file
        cluster_report = "=== Cluster Analysis Report ===\n\n"
        for cluster in sorted(cluster_df['cluster'].unique()):
            cluster_concepts = cluster_df[cluster_df['cluster'] == cluster]
            cluster_center = cluster_concepts[['x', 'y']].mean()
            
            # Find concepts closest to cluster center
            cluster_concepts['dist_to_center'] = np.sqrt(
                (cluster_concepts['x'] - cluster_center['x'])**2 +
                (cluster_concepts['y'] - cluster_center['y'])**2
            )
            
            # Get top 3 most central concepts
            central_concepts = cluster_concepts.nsmallest(3, 'dist_to_center')['concept'].tolist()
            
            cluster_report += (
                f"Cluster {cluster} ({len(cluster_concepts)} concepts):\n"
                f"  Most representative concepts: {', '.join(central_concepts)}\n"
                f"  All concepts: {', '.join(cluster_concepts['concept'].tolist())}\n\n"
            )
        
        # Print the full report
        print("\n" + "="*80)
        print("CLUSTER ANALYSIS REPORT")
        print("="*80)
        print(cluster_report)
        
        try:
            # Try to save to file if possible
            with open(os.path.join(OUTPUT_DIR, 'cluster_analysis.txt'), 'w') as f:
                f.write(cluster_report)
        except Exception as e:
            print(f"\n[WARNING] Could not save cluster analysis to file: {e}")
            print("Cluster analysis results are available above in the console output.")
        
        # Try to show a simple plot if possible
        try:
            plt.figure(figsize=(8, 6))
            scatter = plt.scatter(X_2d[:, 0], X_2d[:, 1], c=clusters, cmap='viridis')
            plt.colorbar(scatter, label='Cluster')
            plt.title('Concept Clustering in Activation Space (t-SNE)')
            plt.tight_layout()
            plt.show()
        except Exception as e:
            print(f"\n[WARNING] Could not display plot: {e}")
            print("Please check disk space or use a different environment for visualization.")
    
    def run(self):
        try:
            self.load_model()
            concepts = self.get_concepts()
            self.extract_activations(concepts)
            self.analyze_layers()
            self.visualize_results()
            self.cluster_analysis()
            logger.info("\nAnalysis complete! Results saved to %s", OUTPUT_DIR)
        except Exception as e:
            logger.error("Error in analysis: %s", str(e), exc_info=True)

if __name__ == "__main__":
    analyzer = CrossLayerAnalyzer()
    analyzer.run()
