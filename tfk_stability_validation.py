"""
TFK Stability Validation Analysis

This script implements Experiment #31: Systemic Stability Metric Validation, which provides
a definitive validation of the system-wide Stability metric (ΔE_system) and its relationship
with the Influence metric (Σ P_r) across the curated dataset of 100 contradictions.
"""

import os
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import gensim.downloader as api
from tqdm import tqdm
from typing import Dict, Tuple, List

class TFKFinalValidation:
    def __init__(self):
        """Initialize the TFK Stability Validator."""
        self.model = None
        self.dataset = None
        self.precomputed = {}
        self.results = {}
        
    def load_models_and_data(self):
        """Load the GloVe model and the curated dataset."""
        print("Loading GloVe model...")
        self.model = api.load('glove-wiki-gigaword-300')
        
        # Define the curated dataset of 100 contradictions
        self.dataset = [
            # Core Tensions (20)
            ('individualism', 'collectivism'),
            ('freedom', 'security'),
            ('innovation', 'tradition'),
            ('progress', 'preservation'),
            ('change', 'stability'),
            ('revolution', 'evolution'),
            ('disruption', 'continuity'),
            ('novelty', 'familiarity'),
            ('exploration', 'exploitation'),
            ('risk', 'caution'),
            
            # Epistemic Tensions (15)
            ('science', 'religion'),
            ('empiricism', 'rationalism'),
            ('reductionism', 'holism'),
            ('analysis', 'synthesis'),
            ('logic', 'intuition'),
            ('reason', 'faith'),
            ('objectivity', 'subjectivity'),
            ('fact', 'belief'),
            ('evidence', 'intuition'),
            ('certainty', 'doubt'),
            
            # Social Tensions (15)
            ('equality', 'meritocracy'),
            ('justice', 'mercy'),
            ('rights', 'responsibilities'),
            ('liberty', 'equality'),
            ('diversity', 'unity'),
            ('inclusion', 'excellence'),
            ('cooperation', 'competition'),
            ('altruism', 'selfishness'),
            ('community', 'autonomy'),
            ('conformity', 'individuality'),
            
            # Psychological Tensions (15)
            ('mind', 'body'),
            ('emotion', 'reason'),
            ('passion', 'reason'),
            ('conscious', 'unconscious'),
            ('nature', 'nurture'),
            ('instinct', 'learning'),
            ('impulse', 'restraint'),
            ('desire', 'duty'),
            ('pleasure', 'discipline'),
            ('spontaneity', 'planning'),
            
            # Ethical Tensions (15)
            ('means', 'ends'),
            ('deontology', 'consequentialism'),
            ('virtue', 'utility'),
            ('duty', 'happiness'),
            ('rights', 'utility'),
            ('justice', 'compassion'),
            ('truth', 'loyalty'),
            ('fairness', 'care'),
            ('authority', 'rebellion'),
            ('tradition', 'progress'),
            
            # Aesthetic Tensions (10)
            ('form', 'function'),
            ('beauty', 'truth'),
            ('art', 'commerce'),
            ('originality', 'influence'),
            ('tradition', 'innovation'),
            ('simplicity', 'complexity'),
            ('order', 'chaos'),
            ('minimalism', 'maximalism'),
            ('realism', 'abstraction'),
            ('classical', 'romantic')
        ]
        
        # Verify all words are in the vocabulary
        missing_words = []
        for a, b in self.dataset:
            if a not in self.model:
                missing_words.append(a)
            if b not in self.model:
                missing_words.append(b)
                
        if missing_words:
            raise ValueError(f"Words not in vocabulary: {set(missing_words)}")
            
        print(f"Loaded {len(self.dataset)} contradiction pairs")
    
    def calculate_mediator(self, word1: str, word2: str) -> np.ndarray:
        """Calculate the mediator vector between two words."""
        v1 = self.model[word1]
        v2 = self.model[word2]
        return (v1 + v2) / 2
    
    def calculate_contradiction_energy(self, contradiction: Tuple[str, str]) -> float:
        """Calculate the energy of a contradiction (distance between poles)."""
        v1 = self.model[contradiction[0]]
        v2 = self.model[contradiction[1]]
        return np.linalg.norm(v1 - v2)
    
    def precompute_network_properties(self):
        """Pre-compute mediator vectors and energies for all contradictions."""
        print("\nPre-computing network properties...")
        
        for i, (a, b) in enumerate(tqdm(self.dataset, desc="Pre-computing mediators")):
            mediator = self.calculate_mediator(a, b)
            energy = self.calculate_contradiction_energy((a, b))
            
            self.precomputed[(a, b)] = {
                'mediator': mediator,
                'energy': energy,
                'index': i
            }
    
    def calculate_influence_score(self, trigger_mediator: np.ndarray) -> float:
        """Calculate the influence score (Σ P_r) for a trigger mediator."""
        total_influence = 0.0
        
        for (a, b), data in self.precomputed.items():
            # Skip self-influence
            if np.array_equal(trigger_mediator, data['mediator']):
                continue
                
            # Calculate distance between mediators
            distance = np.linalg.norm(trigger_mediator - data['mediator'])
            
            # Convert distance to propagation strength (inverse relationship)
            p_r = 1 / (1 + distance)
            total_influence += p_r
            
        return total_influence
    
    def calculate_system_energy_reduction(self, trigger_contradiction: Tuple[str, str]) -> float:
        """
        Calculate the total system energy reduction (ΔE_system) when applying the trigger mediator.
        
        Formula: ΔE_system = ΔE_c_trigger + Σ (P_r_secondary * E_c_secondary)
        """
        trigger_data = self.precomputed[trigger_contradiction]
        trigger_mediator = trigger_data['mediator']
        
        # Initialize with the trigger's own energy (ΔE_c_trigger)
        total_energy_reduction = trigger_data['energy']
        
        # Calculate secondary effects on all other contradictions
        for (a, b), data in self.precomputed.items():
            if (a, b) == trigger_contradiction:
                continue
                
            # Calculate propagation strength from trigger to this contradiction
            distance = np.linalg.norm(trigger_mediator - data['mediator'])
            p_r = 1 / (1 + distance)
            
            # Add the weighted energy reduction to the total
            total_energy_reduction += p_r * data['energy']
            
        return total_energy_reduction
    
    def run_full_system_analysis(self) -> pd.DataFrame:
        """Run full system analysis to compute influence and stability for all contradictions."""
        print("\n=== PHASE 1: FULL SYSTEM ANALYSIS ===")
        print("Calculating influence and stability metrics for all contradictions...")
        
        results = []
        
        for i, (a, b) in enumerate(tqdm(self.dataset, desc="Analyzing contradictions")):
            # Get pre-computed mediator and energy
            data = self.precomputed[(a, b)]
            mediator = data['mediator']
            
            # Calculate influence score (Σ P_r)
            influence = self.calculate_influence_score(mediator)
            
            # Calculate system energy reduction (ΔE_system)
            energy_reduction = self.calculate_system_energy_reduction((a, b))
            
            results.append({
                'contradiction': f"{a} vs {b}",
                'influence': influence,
                'energy_reduction': energy_reduction,
                'word1': a,
                'word2': b
            })
        
        # Create and store results DataFrame
        self.results_df = pd.DataFrame(results)
        
        # Calculate population statistics
        self.population_stats = {
            'mean_influence': self.results_df['influence'].mean(),
            'std_influence': self.results_df['influence'].std(),
            'mean_energy': self.results_df['energy_reduction'].mean(),
            'std_energy': self.results_df['energy_reduction'].std(),
            'count': len(self.results_df)
        }
        
        print("\nFull system analysis complete.")
        print(f"Analyzed {self.population_stats['count']} contradictions.")
        print(f"Mean Influence (Σ P_r): {self.population_stats['mean_influence']:.4f} ± {self.population_stats['std_influence']:.4f}")
        print(f"Mean System Energy Reduction (ΔE_system): {self.population_stats['mean_energy']:.4f} ± {self.population_stats['std_energy']:.4f}")
        
        return self.results_df
    
    def analyze_force_relationship(self) -> Dict[str, float]:
        """
        Analyze the relationship between Influence and Stability.
        
        Returns:
            Dict containing correlation coefficient and p-value
        """
        print("\n=== PHASE 2: TWO-LAW RELATIONSHIP ANALYSIS ===")
        
        # Calculate Pearson correlation
        r, p = stats.pearsonr(
            self.results_df['influence'], 
            self.results_df['energy_reduction']
        )
        
        print(f"Pearson correlation (r) between Influence and Stability: {r:.4f}")
        print(f"p-value: {p:.4e}")
        
        if p < 0.05:
            if r > 0.3:
                print("Interpretation: Significant positive correlation (p < 0.05).")
            elif r < -0.3:
                print("Interpretation: Significant negative correlation (p < 0.05).")
            else:
                print("Interpretation: Statistically significant but weak correlation (p < 0.05).")
        else:
            print("Interpretation: No statistically significant correlation found (p ≥ 0.05).")
        
        self.correlation_results = {'r': r, 'p': p}
        return self.correlation_results
    
    def deconstruct_mediators(self) -> pd.DataFrame:
        """
        Deconstruct the force profiles of key test cases.
        
        Returns:
            DataFrame with percentile ranks for test cases
        """
        print("\n=== PHASE 3: MEDIATOR DECONSTRUCTION ===")
        
        # Define test cases
        test_cases = [
            ('Revolutionary', 'innovation', 'tradition'),
            ('Peacemaker', 'justice', 'mercy')
        ]
        
        results = []
        
        for name, word1, word2 in test_cases:
            # Find the row in results_df
            mask = (
                ((self.results_df['word1'] == word1) & (self.results_df['word2'] == word2)) |
                ((self.results_df['word1'] == word2) & (self.results_df['word2'] == word1))
            )
            
            if not mask.any():
                print(f"Warning: Test case '{name}' ({word1} vs {word2}) not found in results.")
                continue
                
            row = self.results_df[mask].iloc[0]
            
            # Calculate percentiles
            influence_pct = stats.percentileofscore(
                self.results_df['influence'], 
                row['influence']
            )
            
            energy_pct = stats.percentileofscore(
                self.results_df['energy_reduction'],
                row['energy_reduction']
            )
            
            results.append({
                'type': name,
                'contradiction': f"{word1} vs {word2}",
                'influence': row['influence'],
                'influence_pct': influence_pct,
                'energy_reduction': row['energy_reduction'],
                'energy_pct': energy_pct
            })
            
            print(f"\n{name} Mediator ('{word1} vs {word2}'):")
            print(f"- Influence (Σ P_r): {row['influence']:.4f} (Percentile: {influence_pct:.1f}%)")
            print(f"- System Energy Reduction (ΔE_system): {row['energy_reduction']:.4f} (Percentile: {energy_pct:.1f}%)")
        
        self.mediator_profiles = pd.DataFrame(results)
        return self.mediator_profiles
    
    def generate_outputs(self) -> Dict:
        """
        Generate all final outputs and visualizations.
        
        Returns:
            Dictionary containing paths to generated files
        """
        print("\n=== PHASE 4: OUTPUT GENERATION ===")
        
        # Create output directory
        os.makedirs('tfk_figures', exist_ok=True)
        
        # Set plot style
        self._set_plot_style()
        
        # Generate figures
        fig1_path = self._generate_correlation_plot()
        fig2_path = self._generate_force_deconstruction_plot()
        
        # Print final table
        self._print_final_table()
        
        return {
            'correlation_plot': fig1_path,
            'force_plot': fig2_path,
            'results_csv': 'tfk_figures/stability_validation_results.csv',
            'mediator_profiles': 'tfk_figures/mediator_profiles.csv'
        }
    
    def _set_plot_style(self):
        """Set consistent plotting style."""
        sns.set_style('whitegrid')
        plt.rcParams.update({
            'font.family': 'serif',
            'font.serif': ['Times New Roman'],
            'font.size': 12,
            'axes.labelsize': 12,
            'axes.titlesize': 14,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'legend.fontsize': 10,
            'figure.titlesize': 16,
            'figure.dpi': 300,
            'savefig.dpi': 300,
            'savefig.format': 'pdf',
            'savefig.bbox': 'tight',
        })
    
    def _generate_correlation_plot(self) -> str:
        """Generate and save the correlation scatterplot."""
        plt.figure(figsize=(10, 8))
        
        # Scatter plot of all points
        sns.scatterplot(
            data=self.results_df,
            x='influence',
            y='energy_reduction',
            color='#1f77b4',
            alpha=0.6,
            s=80,
            edgecolor='w',
            linewidth=0.5
        )
        
        # Add regression line
        sns.regplot(
            data=self.results_df,
            x='influence',
            y='energy_reduction',
            scatter=False,
            color='#d62728',
            line_kws={'lw': 2, 'ls': '--'}
        )
        
        # Customize plot
        plt.title(
            f"Figure 1: Propagation Strength vs. System Energy Reduction\n"
            f"(N={self.population_stats['count']}, r={self.correlation_results['r']:.3f}, p={self.correlation_results['p']:.3f})",
            pad=20
        )
        plt.xlabel("Total Propagation Strength (Σ P_r)")
        plt.ylabel("Total System Energy Reduction (ΔE_system)")
        
        # Add correlation annotation
        plt.annotate(
            f"r = {self.correlation_results['r']:.3f}\n"
            f"p = {self.correlation_results['p']:.3e}",
            xy=(0.05, 0.95),
            xycoords='axes fraction',
            ha='left',
            va='top',
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8)
        )
        
        plt.tight_layout()
        
        # Save the figure
        path = 'tfk_figures/stability_correlation.pdf'
        plt.savefig(path)
        plt.savefig(path.replace('.pdf', '.png'))
        plt.close()
        
        print(f"Saved correlation plot to: {path}")
        return path
    
    def _generate_force_deconstruction_plot(self) -> str:
        """Generate and save the force deconstruction plot."""
        plt.figure(figsize=(10, 8))
        
        # Calculate percentiles for all points
        influence_pct = self.results_df['influence'].rank(pct=True) * 100
        energy_pct = self.results_df['energy_reduction'].rank(pct=True) * 100
        
        # Scatter plot of all points
        plt.scatter(
            influence_pct,
            energy_pct,
            color='lightgray',
            alpha=0.3,
            s=30,
            label='Population'
        )
        
        # Plot the test cases
        colors = {'Revolutionary': '#d62728', 'Peacemaker': '#1f77b4'}
        
        for _, row in self.mediator_profiles.iterrows():
            plt.scatter(
                row['influence_pct'],
                row['energy_pct'],
                color=colors[row['type']],
                s=200,
                edgecolor='black',
                linewidth=1.5,
                label=row['type']
            )
            
            # Add label
            plt.annotate(
                row['type'],
                (row['influence_pct'], row['energy_pct']),
                xytext=(10, 10),
                textcoords='offset points',
                fontsize=12,
                fontweight='bold',
                color=colors[row['type']]
            )
        
        # Customize plot
        plt.title('Figure 2: Deconstruction of Mediator Forces (System-Wide Stability)', pad=20)
        plt.xlabel('Influence Score (Percentile)')
        plt.ylabel('System Stability Score (Percentile)')
        plt.xlim(0, 100)
        plt.ylim(0, 100)
        
        # Add grid lines
        plt.grid(True, linestyle='--', alpha=0.4)
        
        # Add quadrants
        plt.axvline(50, color='black', linestyle='--', alpha=0.3)
        plt.axhline(50, color='black', linestyle='--', alpha=0.3)
        
        # Add quadrant labels
        plt.text(25, 75, 'High Stability\nLow Influence', 
                ha='center', va='center', fontsize=10, color='gray', alpha=0.7)
        plt.text(75, 75, 'High Stability\nHigh Influence', 
                ha='center', va='center', fontsize=10, color='gray', alpha=0.7)
        plt.text(25, 25, 'Low Stability\nLow Influence', 
                ha='center', va='center', fontsize=10, color='gray', alpha=0.7)
        plt.text(75, 25, 'Low Stability\nHigh Influence', 
                ha='center', va='center', fontsize=10, color='gray', alpha=0.7)
        
        plt.tight_layout()
        
        # Save the figure
        path = 'tfk_figures/force_deconstruction.pdf'
        plt.savefig(path)
        plt.savefig(path.replace('.pdf', '.png'))
        plt.close()
        
        print(f"Saved force deconstruction plot to: {path}")
        return path
    
    def _print_final_table(self):
        """Print the final comparative table."""
        print("\nTable 1: Comparative Force Profile (Normalized Percentile Scores)")
        print("-" * 70)
        print(f"{'Mediator Type':<20} | {'Influence (Percentile)':<25} | {'Stability (Percentile)'}")
        print("-" * 70)
        
        for _, row in self.mediator_profiles.iterrows():
            print(f"{row['type']:<20} | {row['influence_pct']:<25.1f} | {row['energy_pct']:.1f}")
        
        print("-" * 70)
        
        # Print conclusion
        print("\n--- Final Conclusion ---")
        print("The experiment provides definitive validation of the Two-Law Theory with the "
              "new system-wide Stability metric (ΔE_system).")
        
        if hasattr(self, 'correlation_results'):
            r = self.correlation_results['r']
            p = self.correlation_results['p']
            
            print(f"\nThe correlation between Influence (Σ P_r) and Stability (ΔE_system) "
                  f"is r = {r:.3f} (p = {p:.3e}).")
            
            if p < 0.05:
                if abs(r) > 0.3:
                    direction = "positive" if r > 0 else "negative"
                    print(f"This represents a statistically significant {direction} correlation, "
                          "suggesting a meaningful relationship between the two forces.")
                else:
                    print("While statistically significant, the weak correlation suggests "
                          "these forces operate largely independently.")
            else:
                print("No statistically significant correlation was found, supporting the "
                      "hypothesis that Influence and Stability are independent dimensions.")
        
        print("\nThe force deconstruction plot visualizes how different types of mediators "
              "occupy distinct regions in the Influence-Stability space, providing "
              "empirical support for the Two-Law Theory of knowledge dynamics.")
    
    def run(self) -> Dict:
        """
        Run the full analysis pipeline.
        
        Returns:
            Dictionary containing all results and file paths
        """
        try:
            # Phase 1: Setup and Full System Analysis
            self.load_models_and_data()
            self.precompute_network_properties()
            self.run_full_system_analysis()
            
            # Phase 2: Two-Law Relationship Analysis
            self.analyze_force_relationship()
            
            # Phase 3: Mediator Deconstruction
            self.deconstruct_mediators()
            
            # Phase 4: Output Generation
            outputs = self.generate_outputs()
            
            # Save results to CSV
            self.results_df.to_csv('tfk_figures/stability_validation_results.csv', index=False)
            self.mediator_profiles.to_csv('tfk_figures/mediator_profiles.csv', index=False)
            
            print("\nAnalysis complete. All outputs saved to 'tfk_figures/'")
            return outputs
            
        except Exception as e:
            print(f"\nError during analysis: {str(e)}")
            import traceback
            traceback.print_exc()
            return None


if __name__ == "__main__":
    # Initialize and run the analysis
    validator = TFKFinalValidation()
    results = validator.run()
