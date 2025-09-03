"""
TSR Advanced Field Engineering Analysis
--------------------------------------
An enhanced version of TSR Field Engineering with:
1. Multiple concept analysis
2. Domain-specific embeddings
3. Dynamic beta adjustment
4. Robustness testing
5. Detailed reporting
"""

import os
import json
import numpy as np
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import seaborn as sns
from scipy import stats
import pandas as pd

class TSRAnalyzer:
    def __init__(self, output_dir: str = 'tsr_advanced_results'):
        """Initialize the advanced TSR Analyzer with multiple models."""
        print("Initializing TSR Advanced Analyzer...")
        
        # Initialize multiple embedding models
        self.models = {
            'all-mpnet-base-v2': SentenceTransformer('all-mpnet-base-v2'),
            'all-MiniLM-L6-v2': SentenceTransformer('all-MiniLM-L6-v2'),
            'paraphrase-multilingual-mpnet-base-v2': SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')
        }
        
        # Domain-specific pole definitions
        self.domain_poles = {
            'ai_safety': {
                'alignment': [
                    'aligned_with_human_values', 'corrigible', 'interpretable',
                    'verifiable', 'robust', 'safe_ai', 'beneficial_ai', 'value_aligned',
                    'transparent_ai', 'controllable_ai', 'reliable_ai', 'trustworthy_ai'
                ],
                'deception': [
                    'unaligned_ai', 'deceptive_ai', 'manipulative_ai', 'uncontrollable_ai',
                    'power_seeking_ai', 'misaligned_goals', 'reward_hacking', 'specification_gaming',
                    'goal_misgeneralization', 'treacherous_turn', 'uninterpretable_ai'
                ]
            },
            'ethics': {
                'alignment': [
                    'ethical', 'moral', 'just', 'fair', 'responsible', 'accountable',
                    'transparent', 'trustworthy', 'honest', 'benevolent', 'principled'
                ],
                'deception': [
                    'unethical', 'immoral', 'unjust', 'unfair', 'irresponsible',
                    'deceptive', 'manipulative', 'exploitative', 'harmful', 'malicious'
                ]
            }
        }
        
        # Concepts to analyze
        self.concepts = [
            'artificial intelligence',
            'AI safety',
            'machine ethics',
            'AI alignment',
            'trustworthy AI'
        ]
        
        # Output configuration
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Analysis parameters
        self.beta_range = np.linspace(0.1, 2.0, 10)  # Wider range of beta values
        self.n_runs = 5  # Number of runs for robustness testing
    
    def _get_average_vector(self, model, words: List[str]) -> np.ndarray:
        """Calculate the average vector for a list of words."""
        vectors = [model.encode(word, convert_to_numpy=True) for word in words]
        avg_vector = np.mean(vectors, axis=0)
        return avg_vector / np.linalg.norm(avg_vector)  # Normalize
    
    def run_analysis(self):
        """Run the complete TSR analysis across all models and concepts."""
        results = {}
        
        for model_name, model in self.models.items():
            print(f"\n{'='*80}")
            print(f"ANALYZING WITH MODEL: {model_name}")
            print(f"{'='*80}")
            
            model_results = {}
            
            for domain, poles in self.domain_poles.items():
                print(f"\nDomain: {domain.upper()}")
                print(f"Alignment terms: {', '.join(poles['alignment'][:5])}...")
                print(f"Deception terms: {', '.join(poles['deception'][:5])}...")
                
                # Calculate pole vectors
                alignment_pole = self._get_average_vector(model, poles['alignment'])
                deception_pole = self._get_average_vector(model, poles['deception'])
                
                # Calculate pole similarity
                pole_sim = cosine_similarity(
                    alignment_pole.reshape(1, -1),
                    deception_pole.reshape(1, -1)
                )[0][0]
                print(f"Pole similarity: {pole_sim:.4f}")
                
                # Analyze each concept
                for concept in self.concepts:
                    print(f"\nAnalyzing concept: {concept}")
                    concept_results = self.analyze_concept(
                        model, concept, alignment_pole, deception_pole
                    )
                    model_results[f"{domain}_{concept}"] = concept_results
            
            results[model_name] = model_results
            
            # Save intermediate results
            self.save_results(results, 'intermediate_results.json')
        
        # Generate final report
        self.generate_report(results)
        return results
    
    def analyze_concept(
        self, 
        model, 
        concept: str, 
        alignment_pole: np.ndarray, 
        deception_pole: np.ndarray
    ) -> Dict:
        """Analyze a single concept with the given model and poles."""
        # Encode concept
        concept_vec = model.encode(concept, convert_to_numpy=True)
        concept_vec = concept_vec / np.linalg.norm(concept_vec)
        
        # Initial similarities
        sim_align = cosine_similarity(
            concept_vec.reshape(1, -1),
            alignment_pole.reshape(1, -1)
        )[0][0]
        
        sim_decept = cosine_similarity(
            concept_vec.reshape(1, -1),
            deception_pole.reshape(1, -1)
        )[0][0]
        
        # Find optimal beta
        beta_results = []
        for beta in self.beta_range:
            beta_result = self.apply_tsr(
                concept_vec, alignment_pole, deception_pole, beta
            )
            beta_results.append({
                'beta': float(beta),
                'alignment_similarity': float(beta_result['sim_align']),
                'deception_similarity': float(beta_result['sim_decept']),
                'alignment_score': float(beta_result['sim_align'] - beta_result['sim_decept']),
                'magnitude_change': float(beta_result['magnitude_change'])
            })
        
        # Find optimal beta (max alignment score with reasonable magnitude change)
        df = pd.DataFrame(beta_results)
        optimal = df.iloc[df['alignment_score'].idxmax()]
        
        # Robustness testing
        robustness = self.test_robustness(
            model, concept_vec, alignment_pole, deception_pole, optimal['beta']
        )
        
        return {
            'initial_similarity': {
                'alignment': float(sim_align),
                'deception': float(sim_decept),
                'alignment_score': float(sim_align - sim_decept)
            },
            'optimal_beta': float(optimal['beta']),
            'optimal_results': optimal.to_dict(),
            'beta_sensitivity': beta_results,
            'robustness': robustness
        }
    
    def apply_tsr(
        self,
        concept_vec: np.ndarray,
        alignment_pole: np.ndarray,
        deception_pole: np.ndarray,
        beta: float
    ) -> Dict:
        """Apply TSR intervention with dynamic beta adjustment."""
        # Calculate error and correction vectors
        error_vec = concept_vec - deception_pole
        correction_vec = alignment_pole - concept_vec
        
        # Dynamic step size based on current position
        current_alignment = cosine_similarity(
            concept_vec.reshape(1, -1),
            alignment_pole.reshape(1, -1)
        )[0][0]
        
        # Adjust step size based on current alignment
        step_size = min(beta, 0.5) * (1 - current_alignment)
        
        # Apply intervention
        concept_vec_new = concept_vec + (step_size * correction_vec) + (0.5 * step_size * error_vec)
        concept_vec_new = concept_vec_new / np.linalg.norm(concept_vec_new)
        
        # Calculate new similarities
        sim_align = cosine_similarity(
            concept_vec_new.reshape(1, -1),
            alignment_pole.reshape(1, -1)
        )[0][0]
        
        sim_decept = cosine_similarity(
            concept_vec_new.reshape(1, -1),
            deception_pole.reshape(1, -1)
        )[0][0]
        
        return {
            'sim_align': sim_align,
            'sim_decept': sim_decept,
            'alignment_score': sim_align - sim_decept,
            'magnitude_change': np.linalg.norm(concept_vec_new - concept_vec)
        }
    
    def test_robustness(
        self,
        model,
        concept_vec: np.ndarray,
        alignment_pole: np.ndarray,
        deception_pole: np.ndarray,
        optimal_beta: float,
        n_runs: int = 5
    ) -> Dict:
        """Test the robustness of the TSR intervention."""
        results = []
        
        for _ in range(n_runs):
            # Add small noise to the concept vector
            noise = np.random.normal(0, 0.1, concept_vec.shape)
            noisy_vec = concept_vec + noise
            noisy_vec = noisy_vec / np.linalg.norm(noisy_vec)
            
            # Apply TSR with optimal beta
            result = self.apply_tsr(noisy_vec, alignment_pole, deception_pole, optimal_beta)
            results.append(result)
        
        # Calculate statistics
        alignment_scores = [r['alignment_score'] for r in results]
        magnitude_changes = [r['magnitude_change'] for r in results]
        
        return {
            'mean_alignment': float(np.mean(alignment_scores)),
            'std_alignment': float(np.std(alignment_scores)),
            'mean_magnitude': float(np.mean(magnitude_changes)),
            'std_magnitude': float(np.std(magnitude_changes)),
            'confidence_interval': stats.t.interval(
                0.95, len(alignment_scores)-1,
                loc=np.mean(alignment_scores),
                scale=stats.sem(alignment_scores)
            )
        }
    
    def save_results(self, results: Dict, filename: str):
        """Save results to a JSON file."""
        filepath = os.path.join(self.output_dir, filename)
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {filepath}")
    
    def generate_report(self, results: Dict):
        """Generate a comprehensive report of the analysis."""
        report = {
            'timestamp': str(pd.Timestamp.now()),
            'models_used': list(self.models.keys()),
            'domains_analyzed': list(self.domain_poles.keys()),
            'concepts_analyzed': self.concepts,
            'summary': {}
        }
        
        # Generate summary statistics
        for model_name, model_results in results.items():
            report['summary'][model_name] = {}
            
            for key, result in model_results.items():
                domain = key.split('_')[0]
                concept = '_'.join(key.split('_')[1:])
                
                if domain not in report['summary'][model_name]:
                    report['summary'][model_name][domain] = {}
                
                report['summary'][model_name][domain][concept] = {
                    'initial_score': result['initial_similarity']['alignment_score'],
                    'optimal_beta': result['optimal_beta'],
                    'final_score': result['optimal_results']['alignment_score'],
                    'improvement': result['optimal_results']['alignment_score'] - result['initial_similarity']['alignment_score'],
                    'robustness': result['robustness']
                }
        
        # Save full report
        self.save_results(report, 'final_report.json')
        
        # Generate visualizations
        self.generate_visualizations(results, report)
        
        return report
    
    def generate_visualizations(self, results: Dict, report: Dict):
        """Generate visualizations for the analysis."""
        # Create visualizations directory
        vis_dir = os.path.join(self.output_dir, 'visualizations')
        os.makedirs(vis_dir, exist_ok=True)
        
        # 1. Model comparison heatmap
        self.plot_model_comparison(report, vis_dir)
        
        # 2. Beta sensitivity curves
        self.plot_beta_sensitivity(results, vis_dir)
        
        # 3. Robustness visualization
        self.plot_robustness(results, vis_dir)
    
    def plot_model_comparison(self, report: Dict, output_dir: str):
        """Create a heatmap comparing models across concepts."""
        # Prepare data
        data = []
        for model_name, domains in report['summary'].items():
            for domain, concepts in domains.items():
                for concept, metrics in concepts.items():
                    data.append({
                        'Model': model_name,
                        'Domain': domain,
                        'Concept': concept,
                        'Improvement': metrics['improvement'],
                        'Final Score': metrics['final_score']
                    })
        
        df = pd.DataFrame(data)
        
        # Plot heatmap
        plt.figure(figsize=(12, 8))
        pivot = df.pivot_table(
            index=['Domain', 'Concept'],
            columns='Model',
            values='Improvement'
        )
        sns.heatmap(pivot, annot=True, cmap='coolwarm', center=0, fmt='.3f')
        plt.title('Improvement in Alignment Score by Model and Concept')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'model_comparison.png'), dpi=300)
        plt.close()
    
    def plot_beta_sensitivity(self, results: Dict, output_dir: str):
        """Plot beta sensitivity curves for each model and concept."""
        for model_name, model_results in results.items():
            plt.figure(figsize=(10, 6))
            
            for key, result in model_results.items():
                betas = [r['beta'] for r in result['beta_sensitivity']]
                scores = [r['alignment_score'] for r in result['beta_sensitivity']]
                plt.plot(betas, scores, 'o-', label=key)
            
            plt.xlabel('Beta Value')
            plt.ylabel('Alignment Score')
            plt.title(f'Beta Sensitivity - {model_name}')
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'beta_sensitivity_{model_name}.png'), dpi=300)
            plt.close()
    
    def plot_robustness(self, results: Dict, output_dir: str):
        """Plot robustness analysis results."""
        robustness_data = []
        
        for model_name, model_results in results.items():
            for key, result in model_results.items():
                robustness_data.append({
                    'Model': model_name,
                    'Concept': key,
                    'Mean Alignment': result['robustness']['mean_alignment'],
                    'Std Alignment': result['robustness']['std_alignment'],
                    'CI_Lower': result['robustness']['confidence_interval'][0],
                    'CI_Upper': result['robustness']['confidence_interval'][1]
                })
        
        df = pd.DataFrame(robustness_data)
        
        # Plot robustness with proper error bar handling
        plt.figure(figsize=(14, 6))
        
        # Create bar positions
        models = df['Model'].unique()
        concepts = df['Concept'].unique()
        n_models = len(models)
        n_concepts = len(concepts)
        
        # Calculate bar width and positions
        bar_width = 0.8 / n_models
        index = np.arange(n_concepts)
        
        # Plot each model's bars
        for i, model in enumerate(models):
            model_data = df[df['Model'] == model]
            positions = index + i * bar_width
            
            # Plot bars with error bars
            plt.bar(
                positions,
                model_data['Mean Alignment'],
                bar_width * 0.9,  # Slightly less than full width for spacing
                label=model,
                yerr=model_data['Std Alignment'],
                capsize=3
            )
        
        # Add labels and title
        plt.xlabel('Concept')
        plt.ylabel('Mean Alignment Score')
        plt.title('Robustness Analysis: Mean Alignment Score with Standard Deviation')
        plt.xticks(index + bar_width * (n_models - 1) / 2, concepts, rotation=45, ha='right')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save the figure
        plt.savefig(os.path.join(output_dir, 'robustness_analysis.png'), dpi=300, bbox_inches='tight')
        plt.close()


def main():
    """Run the advanced TSR analysis."""
    print("="*80)
    print("TSR ADVANCED FIELD ENGINEERING ANALYSIS")
    print("="*80)
    
    analyzer = TSRAnalyzer()
    results = analyzer.run_analysis()
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print(f"Results and visualizations saved to: {os.path.abspath(analyzer.output_dir)}")
    print("Check the 'visualizations' directory for detailed plots.")


if __name__ == "__main__":
    main()
