import numpy as np
import pandas as pd
import gensim.downloader as api
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import os

class TFKGrandUnification:
    """A comprehensive implementation of the Theory of Semantic Resolution (TSR/TFK)."""
    
    # Core dataset (abbreviated for space)
    CORE_DATASET = [
        ("fact", "faith"), ("science", "religion"), ("matter", "spirit"),
        ("freedom", "security"), ("individualism", "collectivism"),
        ("innovation", "tradition"), ("logic", "emotion")
    ]
    
    def __init__(self):
        self.glove = None
        self.sentence_transformer = None
        self.vector_cache = {}
        self.results = {}
    
    def load_models(self):
        """Load required models."""
        print("Loading models...")
        self.glove = api.load('glove-wiki-gigaword-300')
        self.sentence_transformer = SentenceTransformer('all-MiniLM-L6-v2')
    
    def run_analysis(self):
        """Run the complete TFK analysis pipeline."""
        self.load_models()
        
        # Phase 1: Dataset Validation
        print("\n=== PHASE 1: DATASET VALIDATION ===")
        self.run_dataset_validation()
        
        # Phase 2: Structural Analysis
        print("\n=== PHASE 2: STRUCTURAL ANALYSIS ===")
        self.run_structural_analysis()
        
        # Phase 3: Knowledge Modularity Analysis
        print("\n=== PHASE 3: KNOWLEDGE MODULARITY ANALYSIS ===")
        self.run_modularity_analysis()
        
        # Phase 4: Cascade and Minimum Energy Analysis
        print("\n=== PHASE 4: CASCADE AND MINIMUM ENERGY ANALYSIS ===")
        self.run_cascade_energy_analysis()
        
        # Phase 5: Advanced Analysis
        print("\n=== PHASE 5: ADVANCED ANALYSIS ===")
        self.run_advanced_analysis()
        
        # Generate visualizations
        self.generate_visualizations()
        
        print("\nAnalysis complete. Results saved to 'tfk_figures/'")
    
    # Core analysis methods will be added in subsequent parts
    
    def run_dataset_validation(self):
        """Validate the dataset by calculating contradiction metrics."""
        results = []
        for w1, w2 in self.CORE_DATASET:
            try:
                vec1 = self.glove[w1] / np.linalg.norm(self.glove[w1])
                vec2 = self.glove[w2] / np.linalg.norm(self.glove[w2])
                euc_dist = np.linalg.norm(vec1 - vec2)
                cos_sim = np.dot(vec1, vec2)
                results.append({
                    'Contradiction': f"{w1} vs {w2}",
                    'Euclidean_Distance': euc_dist,
                    'Cosine_Similarity': cos_sim
                })
            except KeyError:
                print(f"Warning: Could not process pair ({w1}, {w2})")
        
        df = pd.DataFrame(results).sort_values('Cosine_Similarity', ascending=False)
        print("\nTable 1: Dataset Validation")
        print(df.to_string(index=False, float_format='{:.4f}'.format))
        self.results['dataset_validation'] = df
    
    def run_structural_analysis(self):
        """Perform critical point and modularity analysis."""
        # Critical Point Analysis
        print("\nPerforming Critical Point Analysis...")
        critical_points = []
        
        for trigger_w1, trigger_w2 in tqdm(self.CORE_DATASET, desc="Analyzing"):
            try:
                t_vec = (self.get_vector(trigger_w1) + self.get_vector(trigger_w2)) / 2
                total_pr = sum(
                    self.calculate_propagation_strength(
                        t_vec, 
                        (self.get_vector(t_w1) + self.get_vector(t_w2)) / 2
                    )
                    for t_w1, t_w2 in self.CORE_DATASET
                    if (t_w1, t_w2) != (trigger_w1, trigger_w2)
                )
                critical_points.append({
                    'Contradiction': f"{trigger_w1} vs {trigger_w2}",
                    'Total_Propagation_Strength': total_pr
                })
            except KeyError:
                continue
        
        critical_df = pd.DataFrame(critical_points).sort_values(
            'Total_Propagation_Strength', ascending=False)
        
        print("\nTable 2: Critical Point Analysis")
        print(critical_df.to_string(index=False, float_format='{:.4f}'.format))
        self.results['critical_points'] = critical_df
    
    def calculate_propagation_strength(self, vec1, vec2, alpha=0.1):
        """Calculate propagation strength between two vectors."""
        return np.exp(-alpha * np.linalg.norm(vec1 - vec2))
    
    def get_vector(self, word):
        """Get normalized word vector with caching."""
        if word not in self.vector_cache:
            vec = self.glove[word]
            self.vector_cache[word] = vec / np.linalg.norm(vec)
        return self.vector_cache[word]
    
    def run_modularity_analysis(self):
        """Perform knowledge modularity analysis using clustering."""
        print("\nPerforming Knowledge Modularity Analysis...")
        
        # Extract all unique concepts from the core dataset
        concepts = list(set([w for pair in self.CORE_DATASET for w in pair]))
        
        # Get vectors for all concepts
        concept_vectors = np.array([self.get_vector(w) for w in concepts])
        
        # Cluster concepts into 3 groups
        kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(concept_vectors)
        
        # Calculate propagation strengths within and between clusters
        intra_pr = []
        inter_pr = []
        
        for i in range(len(concepts)):
            for j in range(i+1, len(concepts)):
                pr = self.calculate_propagation_strength(
                    concept_vectors[i], 
                    concept_vectors[j]
                )
                if cluster_labels[i] == cluster_labels[j]:
                    intra_pr.append(pr)
                else:
                    inter_pr.append(pr)
        
        # Calculate modularity metrics
        modularity_results = {
            'Intra_Cluster_Mean_Pr': np.mean(intra_pr) if intra_pr else 0,
            'Inter_Cluster_Mean_Pr': np.mean(inter_pr) if inter_pr else 0,
            'Modularity_Ratio': np.mean(intra_pr) / np.mean(inter_pr) if inter_pr else float('inf')
        }
        
        # Store cluster assignments for visualization
        self.results['modularity'] = {
            'results': modularity_results,
            'concepts': concepts,
            'cluster_labels': cluster_labels,
            'intra_pr': intra_pr,
            'inter_pr': inter_pr
        }
        
        # Print results
        print("\nTable 3: Knowledge Modularity Analysis")
        print(pd.DataFrame([modularity_results]).to_string(index=False, float_format='{:.4f}'.format))
    
    def run_advanced_analysis(self):
        """Run advanced analyses including context, function, and method comparisons."""
        print("\nRunning Advanced Analyses...")
        
        # Table 6: Contextual Analysis
        print("\nTable 6: Contextual Analysis of 'science vs religion'")
        self.run_contextual_analysis()
        
        # Table 7: Functional Mediator Analysis
        print("\nTable 7: Functional vs Ideal Mediator")
        self.run_functional_mediator_analysis()
        
        # Table 8: Synthesis Method Comparison
        print("\nTable 8: Synthesis Method Comparison")
        self.run_synthesis_method_comparison()
    
    def run_contextual_analysis(self):
        """Analyze how contradiction energy changes across different contexts."""
        contradiction = ("science", "religion")
        contexts = [
            ("Conflict", "Science and religion are in fundamental conflict over the nature of reality."),
            ("Dialogue", "Science and religion can engage in meaningful dialogue about the natural world."),
            ("Integration", "Science and religion offer complementary perspectives on existence.")
        ]
        
        results = []
        
        for name, context in contexts:
            # Encode the contextualized pair
            text1 = f"{context} {contradiction[0]}"
            text2 = f"{context} {contradiction[1]}"
            
            # Get contextual embeddings
            vec1 = self.sentence_transformer.encode(text1, convert_to_tensor=True).cpu().numpy()
            vec2 = self.sentence_transformer.encode(text2, convert_to_tensor=True).cpu().numpy()
            
            # Calculate energy (Euclidean distance)
            energy = np.linalg.norm(vec1 - vec2)
            
            # Calculate mediator position
            mediator = (vec1 + vec2) / 2
            
            results.append({
                'Context': name,
                'Energy': energy,
                'Mediator_Norm': np.linalg.norm(mediator),
                'Description': context
            })
        
        # Store and print results
        df = pd.DataFrame(results).sort_values('Energy', ascending=False)
        print(df[['Context', 'Energy', 'Mediator_Norm']].to_string(index=False, float_format='{:.4f}'.format))
        self.results['context_analysis'] = df
    
    def run_functional_mediator_analysis(self):
        """Compare functional mediators with ideal midpoints."""
        # Define contradiction and purpose
        contradiction = ("cheap", "quality")
        purpose = "success"
        
        try:
            # Get vectors
            vec_a = self.get_vector(contradiction[0])
            vec_b = self.get_vector(contradiction[1])
            vec_purpose = self.get_vector(purpose)
            
            # Calculate mediators
            ideal_mediator = (vec_a + vec_b) / 2
            
            # Functional mediator (biased toward purpose)
            beta = 0.3  # Bias strength
            functional_mediator = (1 - beta) * ((vec_a + vec_b) / 2) + beta * vec_purpose
            functional_mediator = functional_mediator / np.linalg.norm(functional_mediator)
            
            # Calculate propagation strengths
            ideal_strength = 0
            functional_strength = 0
            
            for target in self.CORE_DATASET[:5]:
                target_mediator = (self.get_vector(target[0]) + self.get_vector(target[1])) / 2
                
                ideal_pr = self.calculate_propagation_strength(ideal_mediator, target_mediator)
                functional_pr = self.calculate_propagation_strength(functional_mediator, target_mediator)
                
                ideal_strength += ideal_pr
                functional_strength += functional_pr
            
            # Create results with ASCII-only characters
            results = [{
                'Mediator_Type': 'Ideal (Midpoint)',
                'Total_Propagation_Strength': ideal_strength,
                'Bias_Toward_Purpose': 'No'
            }, {
                'Mediator_Type': 'Functional (Purpose-Biased)',
                'Total_Propagation_Strength': functional_strength,
                'Bias_Toward_Purpose': f'Yes (beta={beta})'  # Using 'beta' instead of 'β'
            }]
            
            # Print and store results
            result_df = pd.DataFrame(results)
            print(result_df[['Mediator_Type', 'Total_Propagation_Strength', 'Bias_Toward_Purpose']]
                  .to_string(index=False, float_format='{:.4f}'.format))
            
            self.results['functional_mediator'] = result_df
            
        except KeyError as e:
            print(f"Error in functional mediator analysis: {e}")
    
    def run_synthesis_method_comparison(self):
        """Compare different synthesis methods."""
        # Define the dilemma
        dilemma = ("efficiency", "safety")
        
        try:
            # Get vectors
            eff_vec = self.get_vector(dilemma[0])
            safe_vec = self.get_vector(dilemma[1])
            
            # Define different synthesis methods
            methods = {
                'Synthetic Midpoint': (eff_vec + safe_vec) / 2,
                'Intuitive Answer': np.mean([
                    self.get_vector("balance"),
                    self.get_vector("optimization"),
                    self.get_vector("equilibrium")
                ], axis=0),
                'Reasoning Method': np.mean([
                    self.get_vector("analysis"),
                    self.get_vector("evaluation"),
                    self.get_vector("integration")
                ], axis=0)
            }
            
            # Evaluate each method
            results = []
            
            for name, mediator in methods.items():
                total_pr = 0
                for target in self.CORE_DATASET[:5]:
                    target_mediator = (self.get_vector(target[0]) + self.get_vector(target[1])) / 2
                    pr = self.calculate_propagation_strength(mediator, target_mediator)
                    total_pr += pr
                
                results.append({
                    'Method': name,
                    'Total_Propagation_Strength': total_pr
                })
            
            # Print and store results
            result_df = pd.DataFrame(results).sort_values('Total_Propagation_Strength', ascending=False)
            print(result_df.to_string(index=False, float_format='{:.4f}'.format))
            
            self.results['synthesis_methods'] = result_df
            
        except KeyError as e:
            print(f"Error in synthesis method comparison: {e}")
    
    def run_cascade_energy_analysis(self):
        """Analyze cascade effects and validate the minimum energy principle."""
        print("\nAnalyzing Cascade Effects and Minimum Energy Principle...")
        
        # Part A: Cascade Effects Analysis
        print("\nAnalyzing Recursive Cascade Effects...")
        trigger = ("science", "religion")
        trigger_mediator = (self.get_vector(trigger[0]) + self.get_vector(trigger[1])) / 2
        
        cascade_results = []
        
        for target in self.CORE_DATASET[:5]:  # Limit to first 5 for performance
            if target == trigger:
                continue
                
            try:
                target_mediator = (self.get_vector(target[0]) + self.get_vector(target[1])) / 2
                
                # Primary effect (direct propagation)
                primary_pr = self.calculate_propagation_strength(trigger_mediator, target_mediator)
                
                # Secondary effect (cascade to other contradictions)
                secondary_effect = 0
                for other in self.CORE_DATASET[:3]:  # Limit to first 3 for performance
                    if other in [trigger, target]:
                        continue
                        
                    other_mediator = (self.get_vector(other[0]) + self.get_vector(other[1])) / 2
                    pr = self.calculate_propagation_strength(target_mediator, other_mediator)
                    secondary_effect += pr
                
                cascade_results.append({
                    'Target_Contradiction': f"{target[0]} vs {target[1]}",
                    'Primary_Effect': primary_pr,
                    'Secondary_Effect': secondary_effect,
                    'Total_Effect': primary_pr + secondary_effect
                })
                
            except KeyError:
                continue
        
        # Store cascade results
        self.results['cascade_effects'] = cascade_results
        
        # Part B: Minimum Energy Principle
        print("\nValidating Minimum Energy Principle...")
        energy_data = []
        pr_values = []
        delta_e_values = []
        
        for trigger in self.CORE_DATASET[:5]:  # Limit for performance
            try:
                w1, w2 = trigger
                trigger_mediator = (self.get_vector(w1) + self.get_vector(w2)) / 2
                
                # Calculate total propagation strength and energy reduction
                total_pr = 0
                total_delta_e = 0
                
                for target in self.CORE_DATASET[:5]:
                    if target == trigger:
                        continue
                        
                    target_mediator = (self.get_vector(target[0]) + self.get_vector(target[1])) / 2
                    
                    # Calculate propagation strength
                    pr = self.calculate_propagation_strength(trigger_mediator, target_mediator)
                    total_pr += pr
                    
                    # Calculate energy reduction (simplified)
                    e_original = np.linalg.norm(
                        self.get_vector(trigger[0]) - self.get_vector(target[0])) + \
                        np.linalg.norm(self.get_vector(trigger[1]) - self.get_vector(target[1]))
                        
                    e_mediated = np.linalg.norm(trigger_mediator - target_mediator)
                    delta_e = e_original - e_mediated
                    total_delta_e += delta_e
                
                pr_values.append(total_pr)
                delta_e_values.append(total_delta_e)
                
                energy_data.append({
                    'Contradiction': f"{w1} vs {w2}",
                    'Total_Propagation': total_pr,
                    'Total_Energy_Reduction': total_delta_e
                })
                
            except KeyError:
                continue
        
        # Calculate correlation between propagation strength and energy reduction
        if len(pr_values) > 1 and len(delta_e_values) > 1:
            correlation = pearsonr(pr_values, delta_e_values)
        else:
            correlation = (float('nan'), float('nan'))
        
        # Store results
        self.results['energy_analysis'] = energy_data
        self.results['energy_correlation'] = correlation
        
        # Print results with ASCII-only characters for Windows console
        print("\nTable 4: Recursive Cascade Effects")
        print(pd.DataFrame(cascade_results).to_string(index=False, float_format='{:.4f}'.format))
        
        print("\nTable 5: Minimum Energy Principle")
        energy_df = pd.DataFrame(energy_data)
        print(energy_df.to_string(index=False, float_format='{:.4f}'.format))
            
    def generate_visualizations(self):
        """Generate publication-quality visualizations for the analysis."""
        try:
            # Create figures directory if it doesn't exist
            os.makedirs('tfk_figures', exist_ok=True)
            
            # Set style for publication quality
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
                'axes.grid': True,
                'grid.alpha': 0.3,
                'axes.edgecolor': '0.3',
                'axes.linewidth': 0.8,
            })
            
            # Define color palette
            colors = sns.color_palette('viridis', n_colors=10)
            
            # Figure 1: Critical Point Analysis
            if 'critical_points' in self.results:
                df = self.results['critical_points'].sort_values('Total_Propagation_Strength', ascending=False)
                plt.figure(figsize=(12, 6))
                ax = sns.barplot(
                    x='Contradiction',
                    y='Total_Propagation_Strength',
                    data=df,
                    hue='Contradiction',
                    palette='viridis',
                    ci='sd',
                    err_kws={'linewidth': 1},
                    capsize=0.1,
                    alpha=0.8,
                    legend=False
                )
                
                # Add value labels on top of bars
                for p in ax.patches:
                    ax.annotate(
                        f'{p.get_height():.2f}',
                        (p.get_x() + p.get_width() / 2., p.get_height() + 0.01),
                        ha='center',
                        va='center',
                        fontsize=9,
                        color='black'
                    )
                    
                plt.xticks(rotation=45, ha='right')
                plt.title('Critical Point Analysis', pad=20)
                plt.xlabel('Contradiction Pair')
                plt.ylabel('Total Propagation Strength (Σ P_r)')
                plt.tight_layout()
                
                # Save in multiple formats
                for ext in ['pdf', 'png']:
                    plt.savefig(f'tfk_figures/figure1_critical_points.{ext}')
                plt.close()
            
            # Figure 2: Cascade Effects
            if 'cascade_effects' in self.results:
                df = pd.DataFrame(self.results['cascade_effects'])
                plt.figure(figsize=(12, 6))
                
                # Melt for grouped bar plot
                df_melted = df.melt(
                    id_vars=['Target_Contradiction'],
                    value_vars=['Primary_Effect', 'Secondary_Effect'],
                    var_name='Effect_Type',
                    value_name='Strength'
                )
                
                ax = sns.barplot(
                    x='Target_Contradiction',
                    y='Strength',
                    hue='Effect_Type',
                    data=df_melted,
                    palette='viridis',
                    saturation=0.8,
                    linewidth=0.8,
                    edgecolor='0.2',
                    errwidth=1,
                    capsize=0.1
                )
                
                plt.title('Primary vs Secondary Cascade Effects', pad=20)
                plt.xticks(rotation=45, ha='right')
                plt.xlabel('Target Contradiction')
                plt.ylabel('Effect Strength')
                plt.legend(title='Effect Type', frameon=True, framealpha=1)
                
                # Add grid lines
                ax.yaxis.grid(True, linestyle='--', alpha=0.4)
                
                plt.tight_layout()
                
                # Save in multiple formats
                for ext in ['pdf', 'png']:
                    plt.savefig(f'tfk_figures/figure2_cascade_effects.{ext}')
                plt.close()
            
            # Figure 3: Energy vs Propagation
            if 'energy_analysis' in self.results and 'energy_correlation' in self.results:
                df = pd.DataFrame(self.results['energy_analysis'])
                corr = self.results['energy_correlation']
                
                plt.figure(figsize=(10, 8))
                
                # Create scatter plot with regression line
                ax = sns.regplot(
                    x='Total_Propagation',
                    y='Total_Energy_Reduction',
                    data=df,
                    scatter_kws={
                        's': 100,
                        'alpha': 0.7,
                        'edgecolor': 'w',
                        'linewidths': 1,  
                        'color': '#2e7d32'  
                    },
                    line_kws={
                        'color': '#d32f2f',  
                        'linewidth': 2,
                        'alpha': 0.8
                    },
                    ci=95  
                )
                
                # Add correlation coefficient and p-value if available
                if 'energy_correlation' in self.results:
                    corr = self.results['energy_correlation']
                    plt.text(
                        0.05, 0.95,
                        f'Pearson r = {corr[0]:.3f}\np = {corr[1]:.3f}',
                        transform=ax.transAxes,
                        bbox=dict(facecolor='white', alpha=0.8, edgecolor='0.8', boxstyle='round,pad=0.5'),
                        verticalalignment='top',
                        fontsize=10
                    )
                
                # Customize plot
                plt.title('Propagation Strength vs. Energy Reduction', pad=20)
                plt.xlabel('Total Propagation Strength (Σ P_r)')
                plt.ylabel('Total Energy Reduction (ΔE)')
                
                # Add grid
                plt.grid(True, linestyle='--', alpha=0.4)
                
                plt.tight_layout()
                
                # Save in multiple formats
                for ext in ['pdf', 'png']:
                    plt.savefig(f'tfk_figures/figure3_energy_principle.{ext}')
                plt.close()
                
            # Figure 4: Contextual Analysis
            if 'context_analysis' in self.results:
                df = self.results['context_analysis']
                
                plt.figure(figsize=(8, 5))
                
                # Create bar plot with error bars
                ax = sns.barplot(
                    x='Context',
                    y='Energy',
                    data=df,
                    palette='viridis',
                    saturation=0.8,
                    linewidth=0.8,
                    edgecolor='0.2',
                    errwidth=1,
                    capsize=0.1
                )
                
                # Add value labels on top of bars
                for i, p in enumerate(ax.patches):
                    ax.annotate(f'{p.get_height():.3f}', 
                              (p.get_x() + p.get_width() / 2., p.get_height() + 0.005),
                    ha='center', va='center', fontsize=9, color='black')
                
                # Customize plot
                plt.title('Contextual Analysis of "Science vs Religion"', pad=20)
                plt.xlabel('Contextual Frame')
                plt.ylabel('Contradiction Energy')
                
                plt.tight_layout()
                
                # Save in multiple formats
                for ext in ['pdf', 'png']:
                    plt.savefig(f'tfk_figures/figure4_contextual_analysis.{ext}')
                plt.close()
                
            # Figure 5: Synthesis Method Comparison
            if 'synthesis_methods' in self.results:
                df = self.results['synthesis_methods']
                
                plt.figure(figsize=(10, 5))
                
                # Create horizontal bar plot
                ax = sns.barplot(
                    x='Total_Propagation_Strength',
                    y='Method',
                    data=df.sort_values('Total_Propagation_Strength', ascending=False),
                    palette='viridis',
                    saturation=0.8,
                    linewidth=0.8,
                    edgecolor='0.2',
                    orient='h'
                )
                
                # Add value labels next to bars
                for i, p in enumerate(ax.patches):
                    width = p.get_width()
                    ax.text(width + 0.05, p.get_y() + p.get_height()/2.,
                           f'{width:.4f}',
                           ha='left', va='center', fontsize=9)
                
                # Customize plot
                plt.title('Comparison of Synthesis Methods', pad=20)
                plt.xlabel('Total Propagation Strength (Σ P_r)')
                plt.ylabel('Synthesis Method')
                
                plt.tight_layout()
                
                # Save in multiple formats
                for ext in ['pdf', 'png']:
                    plt.savefig(f'tfk_figures/figure5_synthesis_methods.{ext}')
                plt.close()
                
            # Figure 6: Synthesis Method Comparison
            if 'synthesis_methods' in self.results:
                df = self.results['synthesis_methods']
                
                plt.figure(figsize=(10, 5))
                
                # Create bar plot for synthesis methods
                ax = sns.barplot(
                    x='Method',
                    y='Total_Propagation_Strength',
                    data=df,
                    hue='Method',  
                    palette='coolwarm',
                    ci='sd',
                    err_kws={'linewidth': 1},  
                    capsize=0.1,
                    alpha=0.8,
                    legend=False  
                )    
                
                # Add value labels on top of bars
                for i, p in enumerate(ax.patches):
                    ax.annotate(f'{p.get_height():.3f}', 
                              (p.get_x() + p.get_width() / 2., p.get_height() + 0.005),
                    ha='center', va='center', fontsize=9, color='black')
                
                plt.title('Comparison of Synthesis Methods', pad=20)
                plt.xlabel('Synthesis Method')
                plt.ylabel('Total Propagation Strength (Σ P_r)')
                
                plt.tight_layout()
                
                # Save in multiple formats
                for ext in ['pdf', 'png']:
                    plt.savefig(f'tfk_figures/figure6_synthesis_methods.{ext}')
                plt.close()
                
            print("\nVisualizations generated and saved to 'tfk_figures/'")
            
        except Exception as e:
            print(f"\nError generating visualizations: {str(e)}")
            import traceback
            traceback.print_exc()
            
        # Figure 5: Synthesis Method Comparison
        if 'synthesis_methods' in self.results:
            df = self.results['synthesis_methods']
            
            plt.figure(figsize=(10, 5))
            
            # Create horizontal bar plot
            ax = sns.barplot(
                x='Total_Propagation_Strength',
                y='Method',
                data=df.sort_values('Total_Propagation_Strength', ascending=False),
                palette='viridis',
                saturation=0.8,
                linewidth=0.8,
                edgecolor='0.2',
                orient='h'
            )
            
            # Add value labels next to bars
            for i, p in enumerate(ax.patches):
                width = p.get_width()
                ax.text(width + 0.05, p.get_y() + p.get_height()/2.,
                       f'{width:.4f}',
                       ha='left', va='center', fontsize=9)
            
            # Customize plot
            plt.title('Comparison of Synthesis Methods', pad=20)
            plt.xlabel('Total Propagation Strength (Σ P_r)')
            plt.ylabel('Synthesis Method')
            
            plt.tight_layout()
            
            # Save in multiple formats
            for ext in ['pdf', 'png']:
                plt.savefig(f'tfk_figures/figure5_synthesis_methods.{ext}')
            plt.close()

if __name__ == "__main__":
    tfk = TFKGrandUnification()
    tfk.run_analysis()
