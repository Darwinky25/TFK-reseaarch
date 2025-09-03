#!/usr/bin/env python3
"""
Analyze Neuron Activations - Post-Experiment #38
Analyze which concepts most activate the specialized neurons identified in Experiment #38
"""

import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sentence_transformers import SentenceTransformer
from typing import Dict, List, Tuple
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
MODEL_NAME = 'all-MiniLM-L6-v2'
TARGET_LAYER = 4
NEURON_CORRELATIONS_PATH = 'neuron_mapping_results/neuron_correlations.csv'
OUTPUT_DIR = 'neuron_analysis_results'
os.makedirs(OUTPUT_DIR, exist_ok=True)

class NeuronActivationAnalyzer:
    """Class for analyzing activation patterns of specialized neurons."""
    
    def __init__(self):
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.concept_activations = {}
        self.neuron_importance = None
        
    def load_model_and_data(self):
        """Load the model and neuron correlation data."""
        logger.info(f"Loading {MODEL_NAME} model...")
        self.model = SentenceTransformer(MODEL_NAME).to(self.device)
        self.model._modules['0'].auto_model.config.output_hidden_states = True
        
        # Load neuron correlations
        logger.info("Loading neuron correlation data...")
        self.neuron_correlations = pd.read_csv(NEURON_CORRELATIONS_PATH)
        
    def extract_activations_for_concepts(self, concepts: List[str]):
        """Extract activations for a list of concepts."""
        logger.info(f"Extracting activations for {len(concepts)} concepts...")
        
        for concept in concepts:
            try:
                # Get tokenized input
                inputs = self.model.tokenize([concept])
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                # Forward pass to get hidden states
                with torch.no_grad():
                    outputs = self.model._modules['0'].auto_model(**inputs, output_hidden_states=True)
                
                # Get activations from target layer
                hidden_states = outputs.hidden_states
                layer_activations = hidden_states[TARGET_LAYER].squeeze(0)  # Remove batch dim
                
                # Store mean activations for this concept
                self.concept_activations[concept] = layer_activations.mean(dim=0).cpu().numpy()
                
            except Exception as e:
                logger.warning(f"Error processing concept '{concept}': {str(e)}")
                continue
    
    def analyze_neuron_importance(self, top_n: int = 5):
        """Analyze which neurons are most important for different concepts."""
        if not self.concept_activations:
            raise ValueError("No concept activations found. Run extract_activations_for_concepts() first.")
        
        logger.info("Analyzing neuron importance...")
        
        # Get top influence and stability neurons
        top_influence_neurons = self.neuron_correlations.nlargest(
            top_n, 'influence_corr')['neuron_id'].tolist()
        top_stability_neurons = self.neuron_correlations.nlargest(
            top_n, 'stability_corr')['neuron_id'].tolist()
        
        # Prepare results
        results = []
        
        for concept, activations in self.concept_activations.items():
            # Calculate activation statistics
            influence_activation = np.mean([activations[n] for n in top_influence_neurons])
            stability_activation = np.mean([activations[n] for n in top_stability_neurons])
            
            results.append({
                'concept': concept,
                'influence_activation': influence_activation,
                'stability_activation': stability_activation,
                'activation_ratio': influence_activation / (stability_activation + 1e-9)
            })
        
        # Create and sort results DataFrame
        self.activation_results = pd.DataFrame(results).sort_values(
            'activation_ratio', ascending=False)
        
        return self.activation_results
    
    def generate_visualization(self):
        """Generate visualization of concept activations."""
        if not hasattr(self, 'activation_results'):
            raise ValueError("No activation results found. Run analyze_neuron_importance() first.")
        
        logger.info("Generating visualization...")
        
        # Set plot style
        plt.figure(figsize=(14, 10))
        sns.set(style="whitegrid", font_scale=1.1)
        
        # Create scatter plot
        scatter = sns.scatterplot(
            data=self.activation_results,
            x='stability_activation',
            y='influence_activation',
            hue='activation_ratio',
            size='activation_ratio',
            sizes=(50, 300),
            alpha=0.8,
            palette='viridis'
        )
        
        # Add labels to some points
        for i, row in self.activation_results.iterrows():
            if i % 3 == 0:  # Label every 3rd point to avoid clutter
                plt.text(
                    row['stability_activation'],
                    row['influence_activation'],
                    row['concept'],
                    fontsize=8,
                    ha='center',
                    va='bottom'
                )
        
        # Add reference lines and labels
        plt.axline((0, 0), slope=1, color='gray', linestyle='--', alpha=0.5)
        plt.text(0.1, 0.15, 'More Stable', rotation=45, color='gray')
        plt.text(0.15, 0.1, 'More Influential', rotation=45, color='gray')
        
        # Customize plot
        plt.title('Concept Activation Space: Influence vs Stability', pad=20)
        plt.xlabel('Average Activation of Stability Neurons')
        plt.ylabel('Average Activation of Influence Neurons')
        plt.legend(title='Influence/Stability Ratio')
        
        # Save figure
        fig_path = os.path.join(OUTPUT_DIR, 'concept_activation_space.png')
        plt.tight_layout()
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        logger.info(f"Visualization saved to: {fig_path}")
        
        # Save results
        results_path = os.path.join(OUTPUT_DIR, 'concept_activation_results.csv')
        self.activation_results.to_csv(results_path, index=False)
        logger.info(f"Results saved to: {results_path}")
        
        # Print top concepts for each category
        self._print_top_concepts()
    
    def _print_top_concepts(self):
        """Print top concepts by activation ratio."""
        print("\n=== Top 5 Most Influential Concepts ===")
        print(self.activation_results.nlargest(5, 'influence_activation')
              [['concept', 'influence_activation']].to_string(index=False))
        
        print("\n=== Top 5 Most Stable Concepts ===")
        print(self.activation_results.nlargest(5, 'stability_activation')
              [['concept', 'stability_activation']].to_string(index=False))
        
        print("\n=== Top 5 Highest Influence/Stability Ratio ===")
        print(self.activation_results.nlargest(5, 'activation_ratio')
              [['concept', 'activation_ratio']].to_string(index=False))
    
    def run(self, concepts: List[str]):
        """Run the complete analysis pipeline."""
        try:
            self.load_model_and_data()
            self.extract_activations_for_concepts(concepts)
            self.analyze_neuron_importance()
            self.generate_visualization()
            logger.info("\nAnalysis completed successfully!")
        except Exception as e:
            logger.error(f"Error in analysis: {str(e)}", exc_info=True)
            raise


if __name__ == "__main__":
    # Example usage with some concepts
    concepts = [
        'innovation', 'tradition', 'justice', 'mercy', 'freedom', 'security',
        'individualism', 'collectivism', 'stability', 'change', 'progress',
        'preservation', 'competition', 'cooperation', 'fact', 'belief', 'reason',
        'faith', 'passion', 'emotion', 'logic', 'truth', 'loyalty', 'risk',
        'safety', 'privacy', 'transparency', 'equality', 'efficiency', 'fairness'
    ]
    
    analyzer = NeuronActivationAnalyzer()
    analyzer.run(concepts)
