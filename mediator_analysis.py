#!/usr/bin/env python3
"""
Mediator Cluster Analysis - Mapping clusters to specific contradictions
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sentence_transformers import SentenceTransformer
import torch
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MediatorAnalyzer:
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = None
        self.concept_activations = {}
        self.contradictions = None
        self.cluster_assignments = None
        
    def load_model(self):
        """Load the sentence transformer model."""
        logger.info("Loading model...")
        self.model = SentenceTransformer('all-MiniLM-L6-v2').to(self.device)
        self.model._modules['0'].auto_model.config.output_hidden_states = True
    
    def load_contradictions(self, data_path='tsr_final_forces.csv'):
        """Load the contradictions data."""
        logger.info("Loading contradictions data...")
        self.contradictions = pd.read_csv(data_path)
        
        # Get all unique concepts
        concepts = set(self.contradictions['concept1']).union(
            set(self.contradictions['concept2']))
        return list(concepts)
    
    def extract_activations(self, concepts):
        """Extract activations for all concepts."""
        logger.info(f"Extracting activations for {len(concepts)} concepts...")
        
        for concept in concepts:
            try:
                # Get tokenized input
                inputs = self.model.tokenize([concept])
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                # Forward pass to get hidden states
                with torch.no_grad():
                    outputs = self.model._modules['0'].auto_model(**inputs, output_hidden_states=True)
                
                # Use the last layer's [CLS] token representation
                hidden_states = outputs.hidden_states
                last_layer = hidden_states[-1].squeeze(0)  # Remove batch dim
                
                # Store mean activation across tokens
                self.concept_activations[concept] = last_layer.mean(dim=0).cpu().numpy()
                
            except Exception as e:
                logger.warning(f"Error processing concept '{concept}': {str(e)}")
    
    def cluster_mediators(self, n_clusters=5):
        """Cluster the mediators in the activation space."""
        logger.info("Clustering mediators...")
        
        # Prepare data for clustering
        concepts = list(self.concept_activations.keys())
        X = np.array([self.concept_activations[c] for c in concepts])
        
        # Dimensionality reduction with t-SNE
        logger.info("Running t-SNE for visualization...")
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(X)-1))
        X_2d = tsne.fit_transform(X)
        
        # K-means clustering
        logger.info("Running K-means clustering...")
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(X)
        
        # Create cluster assignments DataFrame
        self.cluster_assignments = pd.DataFrame({
            'concept': concepts,
            'cluster': clusters,
            'x': X_2d[:, 0],
            'y': X_2d[:, 1]
        })
        
        return self.cluster_assignments
    
    def analyze_clusters(self):
        """Analyze the clusters and map to contradictions."""
        if self.cluster_assignments is None:
            raise ValueError("No cluster assignments found. Run cluster_mediators() first.")
        
        logger.info("Analyzing clusters and mapping to contradictions...")
        
        # Create concept to cluster mapping
        concept_to_cluster = dict(zip(
            self.cluster_assignments['concept'],
            self.cluster_assignments['cluster']
        ))
        
        # Add cluster info to contradictions
        contradictions = self.contradictions.copy()
        contradictions['concept1_cluster'] = contradictions['concept1'].map(concept_to_cluster)
        contradictions['concept2_cluster'] = contradictions['concept2'].map(concept_to_cluster)
        
        # Find cluster pairs that form contradictions
        contradictions['cluster_pair'] = contradictions.apply(
            lambda x: tuple(sorted([x['concept1_cluster'], x['concept2_cluster']])), 
            axis=1
        )
        
        # Generate cluster analysis report
        cluster_report = "=== Mediator Cluster Analysis Report ===\n\n"
        
        # 1. Cluster summaries
        cluster_report += "## Cluster Summaries\n\n"
        for cluster in sorted(self.cluster_assignments['cluster'].unique()):
            cluster_concepts = self.cluster_assignments[
                self.cluster_assignments['cluster'] == cluster
            ]
            
            # Find most central concepts
            center = cluster_concepts[['x', 'y']].mean()
            cluster_concepts['dist_to_center'] = np.sqrt(
                (cluster_concepts['x'] - center['x'])**2 +
                (cluster_concepts['y'] - center['y'])**2
            )
            central_concepts = cluster_concepts.nsmallest(3, 'dist_to_center')
            
            cluster_report += (
                f"### Cluster {cluster} ({len(cluster_concepts)} concepts)\n"
                f"  - Most central concepts: {', '.join(central_concepts['concept'].tolist())}\n"
                f"  - All concepts: {', '.join(cluster_concepts['concept'].tolist())}\n\n"
            )
        
        # 2. Contradiction analysis by cluster pairs
        cluster_report += "## Contradiction Analysis by Cluster Pairs\n\n"
        
        # Group by cluster pairs
        cluster_pairs = contradictions.groupby('cluster_pair').agg({
            'influence_score': 'mean',
            'stability_score': 'mean',
            'concept1': 'count'  # Number of contradictions in this cluster pair
        }).sort_values('influence_score', ascending=False)
        
        for (c1, c2), row in cluster_pairs.iterrows():
            if c1 == c2:
                continue  # Skip within-cluster "contradictions"
                
            # Get example contradictions for this cluster pair
            examples = contradictions[
                (contradictions['cluster_pair'] == (c1, c2)) |
                (contradictions['cluster_pair'] == (c2, c1))
            ].sort_values('influence_score', ascending=False).head(3)
            
            cluster_report += (
                f"### Cluster {c1} vs Cluster {c2}\n"
                f"  - Number of contradictions: {row['concept1']}\n"
                f"  - Average influence score: {row['influence_score']:.2f}\n"
                f"  - Average stability score: {row['stability_score']:.2f}\n"
                f"  - Example contradictions:\n"
            )
            
            for _, ex in examples.iterrows():
                cluster_report += (
                    f"     - {ex['concept1']} vs {ex['concept2']} "
                    f"(Influence: {ex['influence_score']:.2f}, "
                    f"Stability: {ex['stability_score']:.2f})\n"
                )
            
            cluster_report += "\n"
        
        # 3. Most influential contradictions by cluster
        cluster_report += "## Most Influential Contradictions by Cluster\n\n"
        
        for cluster in sorted(contradictions['concept1_cluster'].unique()):
            # Get contradictions where at least one concept is in this cluster
            cluster_contradictions = contradictions[
                (contradictions['concept1_cluster'] == cluster) |
                (contradictions['concept2_cluster'] == cluster)
            ]
            
            # Get top 3 most influential
            top_contradictions = cluster_contradictions.nlargest(
                3, 'influence_score')
            
            if len(top_contradictions) > 0:
                cluster_report += f"### Cluster {cluster} Contradictions\n"
                for _, row in top_contradictions.iterrows():
                    cluster_report += (
                        f"  - {row['concept1']} vs {row['concept2']} "
                        f"(Influence: {row['influence_score']:.2f}, "
                        f"Stability: {row['stability_score']:.2f})\n"
                    )
                cluster_report += "\n"
        
        # Print the report
        print("\n" + "="*80)
        print("MEDIATOR CLUSTER ANALYSIS REPORT")
        print("="*80)
        print(cluster_report)
        
        # Try to save to file
        try:
            os.makedirs('mediator_analysis', exist_ok=True)
            with open('mediator_analysis/cluster_analysis_report.txt', 'w', encoding='utf-8') as f:
                f.write(cluster_report)
            logger.info("Saved cluster analysis report to mediator_analysis/cluster_analysis_report.txt")
        except Exception as e:
            logger.warning(f"Could not save report to file: {e}")
        
        return cluster_report
    
    def visualize_clusters(self):
        """Visualize the clusters."""
        if self.cluster_assignments is None:
            raise ValueError("No cluster assignments found. Run cluster_mediators() first.")
        
        logger.info("Generating cluster visualization...")
        
        try:
            plt.figure(figsize=(12, 10))
            
            # Plot each cluster with a different color
            for cluster in sorted(self.cluster_assignments['cluster'].unique()):
                cluster_data = self.cluster_assignments[
                    self.cluster_assignments['cluster'] == cluster
                ]
                plt.scatter(
                    cluster_data['x'], 
                    cluster_data['y'], 
                    label=f'Cluster {cluster}',
                    alpha=0.7,
                    s=100
                )
                
                # Add cluster label at the centroid
                centroid = cluster_data[['x', 'y']].mean()
                plt.annotate(
                    f'Cluster {cluster}',
                    (centroid['x'], centroid['y']),
                    textcoords="offset points",
                    xytext=(0,10),
                    ha='center',
                    fontsize=12,
                    fontweight='bold'
                )
            
            plt.title('Concept Clusters in Activation Space (t-SNE)', pad=20)
            plt.xlabel('t-SNE Dimension 1')
            plt.ylabel('t-SNE Dimension 2')
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.tight_layout()
            
            # Try to save the figure
            try:
                os.makedirs('mediator_analysis', exist_ok=True)
                plt.savefig('mediator_analysis/concept_clusters.png', dpi=300, bbox_inches='tight')
                logger.info("Saved cluster visualization to mediator_analysis/concept_clusters.png")
            except Exception as e:
                logger.warning(f"Could not save visualization: {e}")
            
            plt.show()
            
        except Exception as e:
            logger.error(f"Error generating visualization: {e}")
    
    def run(self):
        """Run the complete mediator analysis pipeline."""
        try:
            # Load model and data
            self.load_model()
            concepts = self.load_contradictions()
            
            # Extract activations
            self.extract_activations(concepts)
            
            # Cluster mediators
            self.cluster_mediators(n_clusters=5)
            
            # Analyze and print results
            self.analyze_clusters()
            
            # Generate visualization
            self.visualize_clusters()
            
            logger.info("\nMediator analysis completed successfully!")
            
        except Exception as e:
            logger.error(f"Error in mediator analysis: {str(e)}", exc_info=True)
            raise


if __name__ == "__main__":
    analyzer = MediatorAnalyzer()
    analyzer.run()
