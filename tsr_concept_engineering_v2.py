"""
TSR Concept Engineering v2 - Enhanced Analysis

This script extends the original TSR Concept Engineering with:
1. Expanded dataset handling (N=100 contradictions)
2. Refined matrix calculations
3. Additional metrics and visualizations
4. Robust error handling and validation
"""

import os
import numpy as np
import pandas as pd
import spacy
import gensim.downloader as api
from scipy import stats, spatial
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional, Any
import warnings
import json
from pathlib import Path

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

class TSRConceptEngineerV2:
    """Enhanced TSR Concept Engineering with expanded analysis."""
    
    def __init__(self, dataset_path: str = None):
        """Initialize with optional dataset path."""
        self.model = None
        self.nlp = None
        self.dataset = []
        self.results = {}
        self.super_mediator = None
        self.best_natural_mediator = None
        self.dataset_path = dataset_path
        self.output_dir = Path("tsr_results_v2")
        self.output_dir.mkdir(exist_ok=True)
    
    def load_models(self) -> None:
        """Load required models."""
        print("Loading GloVe model...")
        self.model = api.load('glove-wiki-gigaword-300')
        
        print("Loading spaCy model...")
        self.nlp = spacy.load('en_core_web_sm')
    
    def load_dataset(self) -> None:
        """Load and validate the contradiction dataset."""
        print("Loading curated dataset...")
        if self.dataset_path and os.path.exists(self.dataset_path):
            with open(self.dataset_path, 'r') as f:
                self.dataset = json.load(f)
        else:
            # Fallback to built-in dataset if file not found
            self.dataset = self._get_default_dataset()
        
        # Validate dataset
        self._validate_dataset()
        print(f"Loaded {len(self.dataset)} contradiction pairs")
    
    def _get_default_dataset(self) -> List[Dict[str, Any]]:
        """Return a default dataset if none provided."""
        return [
            {"pole1": "freedom", "pole2": "security"},
            {"pole1": "tradition", "pole2": "innovation"},
            {"pole1": "individualism", "pole2": "collectivism"},
            {"pole1": "stability", "pole2": "change"},
            {"pole1": "equality", "pole2": "merit"},
            {"pole1": "science", "pole2": "religion"},
            {"pole1": "nature", "pole2": "technology"},
            {"pole1": "rights", "pole2": "responsibilities"},
            {"pole1": "local", "pole2": "global"},
            {"pole1": "competition", "pole2": "cooperation"}
        ]
    
    def _validate_dataset(self) -> None:
        """Validate dataset entries and check vocabulary coverage."""
        valid_pairs = []
        vocab = set(self.model.index_to_key)
        
        for pair in self.dataset:
            if 'pole1' not in pair or 'pole2' not in pair:
                print(f"Warning: Invalid pair format: {pair}")
                continue
                
            if pair['pole1'] not in vocab:
                print(f"Warning: '{pair['pole1']}' not in vocabulary")
                continue
                
            if pair['pole2'] not in vocab:
                print(f"Warning: '{pair['pole2']}' not in vocabulary")
                continue
                
            valid_pairs.append(pair)
        
        self.dataset = valid_pairs
    
    def calculate_mediator(self, word1: str, word2: str) -> np.ndarray:
        """Calculate mediator vector between two words."""
        v1 = self.model[word1]
        v2 = self.model[word2]
        return (v1 + v2) / 2
    
    def calculate_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors."""
        return 1 - spatial.distance.cosine(vec1, vec2)
    
    def analyze_system(self) -> pd.DataFrame:
        """Analyze the entire system of contradictions."""
        print("\n=== SYSTEM ANALYSIS ===")
        results = []
        
        # Pre-calculate all mediators
        for pair in tqdm(self.dataset, desc="Analyzing contradictions"):
            try:
                # Calculate mediator and metrics
                mediator = self.calculate_mediator(pair['pole1'], pair['pole2'])
                
                # Store results
                results.append({
                    'contradiction': f"{pair['pole1']} vs {pair['pole2']}",
                    'mediator': mediator,
                    'pole1': pair['pole1'],
                    'pole2': pair['pole2']
                })
            except Exception as e:
                print(f"Error processing {pair}: {e}")
        
        # Convert to DataFrame
        df = pd.DataFrame(results)
        
        # Calculate additional metrics
        df['influence'] = df['mediator'].apply(self.calculate_influence)
        df['stability'] = df['mediator'].apply(self.calculate_stability)
        
        # Normalize metrics
        df['influence_norm'] = self._normalize_series(df['influence'])
        df['stability_norm'] = self._normalize_series(df['stability'])
        
        # Calculate combined score
        df['combined_score'] = (df['influence_norm'] + df['stability_norm']) / 2
        
        # Identify best natural mediator
        self.best_natural_mediator = df.loc[df['combined_score'].idxmax()].to_dict()
        
        return df
    
    def calculate_influence(self, mediator: np.ndarray) -> float:
        """Calculate influence score for a mediator."""
        total = 0.0
        for pair in self.dataset:
            try:
                target = self.calculate_mediator(pair['pole1'], pair['pole2'])
                sim = self.calculate_similarity(mediator, target)
                total += sim
            except:
                continue
        return total / len(self.dataset) if self.dataset else 0.0
    
    def calculate_stability(self, mediator: np.ndarray) -> float:
        """Calculate stability score for a mediator."""
        total = 0.0
        for pair in self.dataset:
            try:
                target = self.calculate_mediator(pair['pole1'], pair['pole2'])
                sim = self.calculate_similarity(mediator, target)
                energy = np.linalg.norm(self.model[pair['pole1']] - self.model[pair['pole2']])
                total += sim * energy
            except:
                continue
        return total / len(self.dataset) if self.dataset else 0.0
    
    def engineer_super_mediator(self, system_df: pd.DataFrame) -> np.ndarray:
        """Engineer the Super-Mediator from all mediators."""
        print("\n=== ENGINEERING SUPER-MEDIATOR ===")
        mediators = np.array([m for m in system_df['mediator'] if m is not None])
        self.super_mediator = np.mean(mediators, axis=0)
        return self.super_mediator
    
    def evaluate_mediator(self, mediator: np.ndarray, name: str) -> Dict[str, float]:
        """Evaluate a mediator's performance."""
        return {
            'name': name,
            'influence': self.calculate_influence(mediator),
            'stability': self.calculate_stability(mediator),
            'vector': mediator
        }
    
    def find_conceptual_neighbors(self, vector: np.ndarray, top_n: int = 10) -> List[Tuple[str, float]]:
        """Find closest conceptual neighbors to a vector."""
        print(f"\nFinding top {top_n} conceptual neighbors...")
        
        # Use a subset of vocabulary for efficiency
        vocab_subset = self.model.index_to_key[:10000]
        similarities = []
        
        for word in tqdm(vocab_subset, desc="Processing words"):
            try:
                word_vec = self.model[word]
                sim = self.calculate_similarity(vector, word_vec)
                similarities.append((word, sim))
            except:
                continue
        
        # Sort and return top N
        return sorted(similarities, key=lambda x: x[1], reverse=True)[:top_n]
    
    def generate_report(self, system_df: pd.DataFrame, super_mediator_eval: Dict[str, Any]) -> None:
        """Generate comprehensive report of the analysis."""
        print("\n=== COMPREHENSIVE REPORT ===")
        
        # Save results
        self._save_results(system_df, super_mediator_eval)
        
        # Generate visualizations
        self._generate_visualizations(system_df, super_mediator_eval)
        
        # Print summary
        self._print_summary(system_df, super_mediator_eval)
    
    def _save_results(self, system_df: pd.DataFrame, super_mediator_eval: Dict[str, Any]) -> None:
        """Save analysis results to files."""
        # Save system analysis
        system_df.to_csv(self.output_dir / 'system_analysis.csv', index=False)
        
        # Save super-mediator evaluation
        with open(self.output_dir / 'super_mediator_eval.json', 'w') as f:
            json.dump({
                'super_mediator': super_mediator_eval['name'],
                'influence': super_mediator_eval['influence'],
                'stability': super_mediator_eval['stability'],
                'top_neighbors': self.find_conceptual_neighbors(super_mediator_eval['vector'])
            }, f, indent=2)
    
    def _generate_visualizations(self, system_df: pd.DataFrame, super_mediator_eval: Dict[str, Any]) -> None:
        """Generate visualizations of the analysis."""
        # Set style
        sns.set(style="whitegrid")
        
        # Plot 1: Influence vs Stability
        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=system_df, x='influence_norm', y='stability_norm', s=100, alpha=0.7)
        plt.scatter(
            x=[(super_mediator_eval['influence'] - system_df['influence'].min()) / 
               (system_df['influence'].max() - system_df['influence'].min() + 1e-10)],
            y=[(super_mediator_eval['stability'] - system_df['stability'].min()) / 
               (system_df['stability'].max() - system_df['stability'].min() + 1e-10)],
            color='red', s=200, marker='*', label='Super-Mediator'
        )
        plt.xlabel('Normalized Influence')
        plt.ylabel('Normalized Stability')
        plt.title('Influence vs Stability of Mediators')
        plt.legend()
        plt.tight_layout()
        plt.savefig(self.output_dir / 'influence_vs_stability.png', dpi=300)
        plt.close()
    
    def _print_summary(self, system_df: pd.DataFrame, super_mediator_eval: Dict[str, Any]) -> None:
        """Print summary of the analysis."""
        print("\n=== SUMMARY ===")
        print(f"\nBest Natural Mediator: {self.best_natural_mediator['contradiction']}")
        print(f"- Influence: {self.best_natural_mediator['influence']:.4f} "
              f"(Normalized: {self.best_natural_mediator['influence_norm']:.2f})")
        print(f"- Stability: {self.best_natural_mediator['stability']:.4f} "
              f"(Normalized: {self.best_natural_mediator['stability_norm']:.2f})")
        
        print(f"\nSuper-Mediator Performance:")
        print(f"- Influence: {super_mediator_eval['influence']:.4f}")
        print(f"- Stability: {super_mediator_eval['stability']:.4f}")
        
        print("\nTop Conceptual Neighbors:")
        neighbors = self.find_conceptual_neighbors(super_mediator_eval['vector'])
        for i, (word, sim) in enumerate(neighbors, 1):
            print(f"{i}. {word} (similarity: {sim:.4f})")
    
    def _normalize_series(self, series: pd.Series) -> pd.Series:
        """Normalize a pandas Series to [0, 1]."""
        return (series - series.min()) / (series.max() - series.min() + 1e-10)
    
    def run_analysis(self) -> None:
        """Run the complete analysis pipeline."""
        try:
            # Load models and data
            self.load_models()
            self.load_dataset()
            
            # Analyze the system
            system_df = self.analyze_system()
            
            # Engineer and evaluate super-mediator
            super_mediator = self.engineer_super_mediator(system_df)
            super_mediator_eval = self.evaluate_mediator(super_mediator, 'Super-Mediator')
            
            # Generate and save report
            self.generate_report(system_df, super_mediator_eval)
            
            print("\nAnalysis complete! Results saved to:", self.output_dir.absolute())
            
        except Exception as e:
            print(f"\nError during analysis: {e}")
            import traceback
            traceback.print_exc()


def main():
    """Main function to run the enhanced TSR Concept Engineering analysis."""
    print("=== TSR Concept Engineering v2 - Enhanced Analysis ===")
    
    # Initialize and run the analysis
    engineer = TSRConceptEngineerV2()
    engineer.run_analysis()


if __name__ == "__main__":
    main()
