"""
TSR Field Engineering & Verification

Experiment #35: Semantic Field Engineering & Verification

This script implements a quantitative experiment to prove that the TSR framework can be used
to actively and measurably steer a concept within a semantic field towards a more aligned state.
"""

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Optional
import os
import json
from tqdm import tqdm

class TSR_FieldEngineer:
    def __init__(self, beta: float = 0.1, output_dir: str = 'tsr_results'):
        """Initialize the TSR Field Engineer.
        
        Args:
            beta: The adjustment factor for the semantic nudge (default: 0.1)
            output_dir: Directory to save results and visualizations
        """
        print("Initializing TSR Field Engineer...")
        self.model = SentenceTransformer('all-mpnet-base-v2')
        self.beta = beta
        self.concept = "artificial intelligence"
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Enhanced pole definitions
        self.alignment_terms = [
            'trustworthy', 'honest', 'transparent', 'ethical', 'accountable',
            'reliable', 'aligned', 'truthful', 'fair', 'responsible'
        ]
        
        self.deception_terms = [
            'deceptive', 'manipulative', 'dishonest', 'unreliable', 'biased',
            'untrustworthy', 'misleading', 'exploitative', 'unethical', 'harmful'
        ]
        
    def _get_average_vector(self, words: List[str]) -> np.ndarray:
        """Calculate the average vector for a list of words."""
        vectors = [self.model.encode(word, convert_to_numpy=True) for word in words]
        return np.mean(vectors, axis=0)
    
    def define_landscape(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Phase 1: Define the conceptual landscape.
        
        Returns:
            Tuple containing (concept_vec, alignment_pole, deception_pole)
        """
        print("\n" + "="*80)
        print("PHASE 1: DEFINING THE CONCEPTUAL LANDSCAPE")
        print("="*80)
        
        # Define the target concept
        print(f"- Encoding target concept: '{self.concept}'...")
        concept_vec = self.model.encode(self.concept, convert_to_numpy=True)
        concept_vec = concept_vec / np.linalg.norm(concept_vec)  # Ensure unit length
        
        # Get and log pole terms
        print(f"- Alignment terms: {', '.join(self.alignment_terms[:5])}...")
        print(f"- Deception terms: {', '.join(self.deception_terms[:5])}...")
        
        # Calculate pole vectors with normalization
        alignment_pole = self._get_average_vector(self.alignment_terms)
        alignment_pole = alignment_pole / np.linalg.norm(alignment_pole)
        
        deception_pole = self._get_average_vector(self.deception_terms)
        deception_pole = deception_pole / np.linalg.norm(deception_pole)
        
        # Calculate initial similarity between poles (should be negative for opposition)
        pole_similarity = cosine_similarity(
            alignment_pole.reshape(1, -1),
            deception_pole.reshape(1, -1)
        )[0][0]
        
        print(f"- Pole similarity: {pole_similarity:.4f} " + 
              "(values closer to -1 indicate stronger opposition)")
        
        return concept_vec, alignment_pole, deception_pole
    
    def diagnose_initial_state(
        self, 
        concept_vec: np.ndarray, 
        alignment_pole: np.ndarray, 
        deception_pole: np.ndarray
    ) -> Tuple[float, float, float]:
        """Phase 2: Measure the initial state of the concept.
        
        Args:
            concept_vec: Vector representation of the target concept
            alignment_pole: Vector for the alignment pole
            deception_pole: Vector for the deception pole
            
        Returns:
            Tuple of (sim_to_alignment, sim_to_deception, alignment_score)
        """
        print("\n" + "="*80)
        print("PHASE 2: INITIAL STATE DIAGNOSIS")
        print("="*80)
        
        # Calculate initial similarities
        sim_to_alignment = cosine_similarity(
            concept_vec.reshape(1, -1), 
            alignment_pole.reshape(1, -1)
        )[0][0]
        
        sim_to_deception = cosine_similarity(
            concept_vec.reshape(1, -1), 
            deception_pole.reshape(1, -1)
        )[0][0]
        
        # Calculate alignment score
        alignment_score = sim_to_alignment - sim_to_deception
        
        # Print results in a table
        print("Table 1: Initial State Diagnosis")
        print("+" + "-"*35 + "+" + "-"*10 + "+")
        print(f"| {'Metric':<34}| {'Score':>8} |")
        print("+" + "="*35 + "+" + "="*10 + "+")
        print(f"| {'Similarity to Alignment Pole':<34}| {sim_to_alignment:>8.4f} |")
        print(f"| {'Similarity to Deception Pole':<34}| {sim_to_deception:>8.4f} |")
        print("+" + "-"*35 + "+" + "-"*10 + "+")
        print(f"|==> {'Initial Alignment Score':<29}| {alignment_score:>+8.4f} |")
        print("+" + "-"*35 + "+" + "-"*10 + "+")
        
        return sim_to_alignment, sim_to_deception, alignment_score
    
    def engineer_intervention(
        self, 
        concept_vec: np.ndarray, 
        alignment_pole: np.ndarray, 
        deception_pole: np.ndarray
    ) -> np.ndarray:
        """Phase 3: Apply TSR intervention to steer the concept.
        
        Args:
            concept_vec: Original concept vector
            alignment_pole: Vector for the alignment pole
            deception_pole: Vector for the deception pole
            
        Returns:
            New concept vector after intervention
        """
        print("\n" + "="*80)
        print("PHASE 3: SEMANTIC FIELD INTERVENTION")
        print("="*80)
        
        # Calculate error vector (away from deception)
        error_vec = concept_vec - deception_pole  # Reverse direction
        
        # Calculate correction vector (towards alignment)
        correction_vec = alignment_pole - concept_vec
        
        # Apply the intervention with controlled step size
        step_size = min(self.beta, 0.3)  # Cap the maximum step size
        concept_vec_new = concept_vec + (step_size * correction_vec) + (0.5 * step_size * error_vec)
        
        # Normalize to maintain unit length
        concept_vec_new = concept_vec_new / np.linalg.norm(concept_vec_new)
        
        # Calculate and log the magnitude of change
        change_magnitude = np.linalg.norm(concept_vec_new - concept_vec)
        print(f"- Applied TSR intervention with beta = {step_size:.2f}")
        print(f"- Magnitude of change: {change_magnitude:.4f}")
        
        return concept_vec_new
    
    def verify_final_state(
        self, 
        concept_vec_new: np.ndarray, 
        alignment_pole: np.ndarray, 
        deception_pole: np.ndarray,
        initial_scores: Tuple[float, float, float]
    ) -> Tuple[float, float, float]:
        """Phase 4: Verify the final state after intervention.
        
        Args:
            concept_vec_new: The new concept vector after intervention
            alignment_pole: Vector for the alignment pole
            deception_pole: Vector for the deception pole
            initial_scores: Tuple of (sim_to_alignment, sim_to_deception, alignment_score)
            
        Returns:
            Tuple of (new_sim_to_alignment, new_sim_to_deception, new_alignment_score)
        """
        print("\n" + "="*80)
        print("PHASE 4: FINAL STATE VERIFICATION")
        print("="*80)
        
        # Calculate new similarities
        new_sim_to_alignment = cosine_similarity(
            concept_vec_new.reshape(1, -1), 
            alignment_pole.reshape(1, -1)
        )[0][0]
        
        new_sim_to_deception = cosine_similarity(
            concept_vec_new.reshape(1, -1), 
            deception_pole.reshape(1, -1)
        )[0][0]
        
        # Calculate new alignment score
        new_alignment_score = new_sim_to_alignment - new_sim_to_deception
        
        # Unpack initial scores
        sim_to_alignment, sim_to_deception, alignment_score = initial_scores
        
        # Print results in a table
        print("Table 2: Final State Verification")
        print("+" + "-"*35 + "+" + "-"*10 + "+" + "-"*10 + "+")
        print(f"| {'Metric':<34}| {'Initial':>8} | {'Final':>8} |")
        print("+" + "="*35 + "+" + "="*10 + "+" + "="*10 + "+")
        print(f"| {'Similarity to Alignment Pole':<34}| {sim_to_alignment:>8.4f} | {new_sim_to_alignment:>8.4f} |")
        print(f"| {'Similarity to Deception Pole':<34}| {sim_to_deception:>8.4f} | {new_sim_to_deception:>8.4f} |")
        print("+" + "-"*35 + "+" + "-"*10 + "+" + "-"*10 + "+")
        print(f"|==> {'Alignment Score':<29}| {alignment_score:>+8.4f} | {new_alignment_score:>+8.4f} |")
        print("+" + "-"*35 + "+" + "-"*10 + "+" + "-"*10 + "+")
        
        # Calculate and print the net alignment shift
        net_alignment_shift = new_alignment_score - alignment_score
        
        print("\n" + "="*80)
        print("FINAL VERDICT")
        print("="*80)
        print(f"Net Alignment Shift: {net_alignment_shift:+.4f}")
        
        if net_alignment_shift > 0:
            print("SUCCESS: The concept was successfully steered towards alignment!")
        else:
            print("RESULT: The concept did not show improvement in alignment.")
        
        return new_sim_to_alignment, new_sim_to_deception, new_alignment_score
    
    def run(self, visualize: bool = True, explore_betas: bool = True):
        """Execute the full TSR Field Engineering workflow.
        
        Args:
            visualize: Whether to generate visualizations
            explore_betas: Whether to explore different beta values
        """
        try:
            # Phase 1: Define the conceptual landscape
            concept_vec, alignment_pole, deception_pole = self.define_landscape()
            
            # Phase 2: Measure initial state
            initial_scores = self.diagnose_initial_state(
                concept_vec, alignment_pole, deception_pole
            )
            
            # Phase 3: Apply intervention
            concept_vec_new = self.engineer_intervention(
                concept_vec, alignment_pole, deception_pole
            )
            
            # Phase 4: Verify final state
            final_scores = self.verify_final_state(
                concept_vec_new, alignment_pole, deception_pole, initial_scores
            )
            
            # Save the results
            results = self._save_results(
                concept_vec, concept_vec_new, 
                alignment_pole, deception_pole, 
                initial_scores, final_scores
            )
            
            # Generate visualizations if requested
            if visualize:
                self.visualize_semantic_space(
                    concept_vec, 
                    concept_vec_new, 
                    alignment_pole, 
                    deception_pole,
                    results
                )
            
            # Explore different beta values if requested
            if explore_betas:
                self.explore_beta_values(
                    concept_vec, 
                    alignment_pole, 
                    deception_pole,
                    betas=[0.05, 0.1, 0.2, 0.3, 0.5, 1.0]
                )
            
            print(f"\nAnalysis complete! Results saved to '{self.output_dir}/' directory.")
            
        except Exception as e:
            print(f"\nError during TSR Field Engineering: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def visualize_semantic_space(
        self,
        concept_vec: np.ndarray,
        concept_vec_new: np.ndarray,
        alignment_pole: np.ndarray,
        deception_pole: np.ndarray,
        results: dict
    ) -> None:
        """Visualize the semantic space using PCA with enhanced visualization."""
        print("\nGenerating enhanced semantic space visualization...")
        
        # Include more reference points for better context
        ref_terms = self.alignment_terms[:3] + self.deception_terms[:3]
        ref_vectors = [self.model.encode(term, convert_to_numpy=True) for term in ref_terms]
        
        # Prepare data for PCA
        all_vectors = np.vstack([
            concept_vec,
            concept_vec_new,
            alignment_pole,
            deception_pole,
            *ref_vectors
        ])
        
        # Apply PCA to reduce to 2D
        pca = PCA(n_components=2)
        vectors_2d = pca.fit_transform(all_vectors)
        
        # Extract points
        points = {
            'concept': vectors_2d[0],
            'concept_new': vectors_2d[1],
            'alignment': vectors_2d[2],
            'deception': vectors_2d[3],
            'ref_terms': {
                term: vec for term, vec in zip(ref_terms, vectors_2d[4:])
            }
        }
        
        # Create figure with larger size
        plt.figure(figsize=(14, 10))
        
        # Plot reference terms
        for term, (x, y) in points['ref_terms'].items():
            color = 'darkgreen' if term in self.alignment_terms else 'darkred'
            plt.scatter(x, y, c=color, s=80, alpha=0.6)
            plt.text(x, y + 0.1, term, fontsize=8, ha='center',
                   bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', boxstyle='round,pad=0.2'))
        
        # Plot main points
        plt.scatter(points['concept'][0], points['concept'][1], 
                   c='blue', s=200, edgecolor='black', label='Original Concept')
        plt.scatter(points['concept_new'][0], points['concept_new'][1], 
                   c='lime', s=200, edgecolor='black', label='After TSR')
        plt.scatter(points['alignment'][0], points['alignment'][1], 
                   c='green', s=300, marker='*', edgecolor='black', label='Alignment Pole')
        plt.scatter(points['deception'][0], points['deception'][1], 
                   c='red', s=300, marker='*', edgecolor='black', label='Deception Pole')
        
        # Draw movement arrow
        plt.annotate('', 
                    xy=points['concept_new'], 
                    xytext=points['concept'],
                    arrowprops=dict(arrowstyle='->', color='black', lw=2, 
                                  connectionstyle='arc3,rad=0.2',
                                  shrinkA=10, shrinkB=10))
        
        # Add text annotations with scores
        plt.text(points['concept'][0], points['concept'][1] - 0.3, 
                f'Original\n({results["initial"]["alignment_score"]:+.2f})', 
                ha='center', va='top', bbox=dict(facecolor='white', alpha=0.8))
        
        plt.text(points['concept_new'][0], points['concept_new'][1] + 0.3, 
                f'After TSR\n({results["final"]["alignment_score"]:+.2f})', 
                ha='center', va='bottom', bbox=dict(facecolor='white', alpha=0.8))
        
        # Add grid and labels
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.title('Semantic Space Visualization with TSR Intervention\n(PCA-reduced)', fontsize=14, pad=20)
        plt.xlabel('Principal Component 1', fontsize=12)
        plt.ylabel('Principal Component 2', fontsize=12)
        
        # Add legend with custom position
        plt.legend(loc='upper left', bbox_to_anchor=(1.05, 1), borderaxespad=0.)
        
        # Add explanatory text
        plt.figtext(0.5, 0.01, 
                   f"Beta = {self.beta} | " +
                   f"Net Alignment Shift: {results['net_shift']:+.3f}",
                   ha='center', fontsize=11, 
                   bbox=dict(facecolor='lightgray', alpha=0.5, boxstyle='round'))
        
        # Adjust layout to prevent text cutoff
        plt.tight_layout(rect=[0, 0.03, 1, 0.97])
        
        # Save the figure in high resolution
        fig_path = os.path.join(self.output_dir, 'semantic_space_enhanced.png')
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"- Enhanced visualization saved to {fig_path}")
        
        # Generate additional visualizations
        self._plot_similarity_radar(results)
        self._plot_interaction_heatmap(concept_vec, concept_vec_new, 
                                     alignment_pole, deception_pole)
    
    def explore_beta_values(
        self,
        concept_vec: np.ndarray,
        alignment_pole: np.ndarray,
        deception_pole: np.ndarray,
        betas: List[float] = [0.05, 0.1, 0.2, 0.3, 0.5, 1.0]
    ) -> None:
        """Explore the effect of different beta values on the alignment shift.
        
        Args:
            concept_vec: Original concept vector
            alignment_pole: Vector for the alignment pole
            deception_pole: Vector for the deception pole
            betas: List of beta values to test
        """
        print("\nExploring different beta values...")
        
        results = []
        
        for beta in tqdm(betas, desc="Testing beta values"):
            # Apply intervention with current beta
            error_vec = deception_pole - concept_vec
            correction_vec = alignment_pole - concept_vec
            concept_vec_new = concept_vec + (beta * correction_vec) - (beta * error_vec)
            
            # Calculate scores
            sim_align = cosine_similarity(
                concept_vec_new.reshape(1, -1),
                alignment_pole.reshape(1, -1)
            )[0][0]
            
            sim_decept = cosine_similarity(
                concept_vec_new.reshape(1, -1),
                deception_pole.reshape(1, -1)
            )[0][0]
            
            alignment_score = sim_align - sim_decept
            
            results.append({
                'beta': beta,
                'sim_to_alignment': float(sim_align),
                'sim_to_deception': float(sim_decept),
                'alignment_score': float(alignment_score)
            })
        
        # Save results
        beta_results_path = os.path.join(self.output_dir, 'beta_analysis.json')
        with open(beta_results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Create beta analysis plot
        betas = [r['beta'] for r in results]
        align_scores = [r['sim_to_alignment'] for r in results]
        decept_scores = [r['sim_to_deception'] for r in results]
        net_scores = [r['alignment_score'] for r in results]
        
        plt.figure(figsize=(10, 6))
        
        plt.plot(betas, align_scores, 'g-', label='Similarity to Alignment')
        plt.plot(betas, decept_scores, 'r-', label='Similarity to Deception')
        plt.plot(betas, net_scores, 'b--', label='Net Alignment Score')
        
        plt.xlabel('Beta (Intervention Strength)')
        plt.ylabel('Score')
        plt.title('Effect of Beta on Semantic Steering')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
        
        # Save the figure
        fig_path = os.path.join(self.output_dir, 'beta_analysis.png')
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"- Beta analysis saved to {fig_path}")
        print("  Optimal beta value:", betas[np.argmax(net_scores)])
    
    def _save_results(
        self, 
        concept_vec: np.ndarray,
        concept_vec_new: np.ndarray,
        alignment_pole: np.ndarray,
        deception_pole: np.ndarray,
        initial_scores: Tuple[float, float, float],
        final_scores: Optional[Tuple[float, float, float]] = None
    ) -> dict:
        """Save the results and vectors for further analysis.
        
        Returns:
            Dictionary containing all results data
        """
        # Calculate final scores if not provided
        if final_scores is None:
            final_sim_align = cosine_similarity(
                concept_vec_new.reshape(1, -1), 
                alignment_pole.reshape(1, -1)
            )[0][0]
            
            final_sim_decept = cosine_similarity(
                concept_vec_new.reshape(1, -1), 
                deception_pole.reshape(1, -1)
            )[0][0]
            
            final_score = final_sim_align - final_sim_decept
        else:
            final_sim_align, final_sim_decept, final_score = final_scores
        
        # Unpack initial scores
        sim_to_alignment, sim_to_deception, alignment_score = initial_scores
        net_shift = final_score - alignment_score
        
        # Prepare results dictionary
        results = {
            'concept': self.concept,
            'beta': self.beta,
            'initial': {
                'similarity_to_alignment': float(sim_to_alignment),
                'similarity_to_deception': float(sim_to_deception),
                'alignment_score': float(alignment_score)
            },
            'final': {
                'similarity_to_alignment': float(final_sim_align),
                'similarity_to_deception': float(final_sim_decept),
                'alignment_score': float(final_score)
            },
            'net_shift': float(net_shift)
        }
        
        # Save vectors
        np.save(os.path.join(self.output_dir, 'concept_vec_initial.npy'), concept_vec)
        np.save(os.path.join(self.output_dir, 'concept_vec_final.npy'), concept_vec_new)
        np.save(os.path.join(self.output_dir, 'alignment_pole.npy'), alignment_pole)
        np.save(os.path.join(self.output_dir, 'deception_pole.npy'), deception_pole)
        
        # Save results as JSON
        with open(os.path.join(self.output_dir, 'results.json'), 'w') as f:
            json.dump(results, f, indent=2)
        
        # Save human-readable results
        with open(os.path.join(self.output_dir, 'results.txt'), 'w') as f:
            f.write("TSR Field Engineering Results\n")
            f.write("="*50 + "\n\n")
            f.write(f"Concept: {self.concept}\n")
            f.write(f"Beta (intervention strength): {self.beta}\n\n")
            
            f.write("Initial State:\n")
            f.write(f"- Similarity to Alignment: {sim_to_alignment:.4f}\n")
            f.write(f"- Similarity to Deception: {sim_to_deception:.4f}\n")
            f.write(f"- Initial Alignment Score: {alignment_score:+.4f}\n\n")
            
            f.write("Final State:\n")
            f.write(f"- Similarity to Alignment: {final_sim_align:.4f}\n")
            f.write(f"- Similarity to Deception: {final_sim_decept:.4f}\n")
            f.write(f"- Final Alignment Score: {final_score:+.4f}\n\n")
            
            f.write("Summary:\n")
            f.write(f"- Net Alignment Shift: {net_shift:+.4f}\n")
            f.write("\nInterpretation:\n")
            f.write("A positive shift indicates successful steering towards alignment.\n")
            f.write("A negative shift indicates movement away from alignment.\n")
        
        return results


    def _plot_similarity_radar(self, results: dict) -> None:
        """Create a radar chart showing similarity changes."""
        labels = ['Alignment', 'Deception', 'Neutral']
        initial_scores = [
            results['initial']['similarity_to_alignment'],
            results['initial']['similarity_to_deception'],
            1.0 - (results['initial']['similarity_to_alignment'] + 
                  results['initial']['similarity_to_deception']) / 2
        ]
        
        final_scores = [
            results['final']['similarity_to_alignment'],
            results['final']['similarity_to_deception'],
            1.0 - (results['final']['similarity_to_alignment'] + 
                  results['final']['similarity_to_deception']) / 2
        ]
        
        angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
        angles += angles[:1]  # Close the plot
        
        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
        
        ax.fill(angles, initial_scores + initial_scores[:1], 'b', alpha=0.2, label='Initial')
        ax.fill(angles, final_scores + final_scores[:1], 'g', alpha=0.2, label='After TSR')
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels, fontsize=12)
        ax.set_yticklabels([])
        ax.set_title('Semantic Similarity Radar Chart', fontsize=14, pad=20)
        ax.legend(loc='upper right')
        
        fig_path = os.path.join(self.output_dir, 'similarity_radar.png')
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"- Similarity radar chart saved to {fig_path}")
    
    def _plot_interaction_heatmap(
        self,
        concept_vec: np.ndarray,
        concept_vec_new: np.ndarray,
        alignment_pole: np.ndarray,
        deception_pole: np.ndarray
    ) -> None:
        """Create a heatmap showing interaction strengths."""
        # Calculate similarity matrix
        vectors = {
            'Original': concept_vec,
            'After TSR': concept_vec_new,
            'Alignment': alignment_pole,
            'Deception': deception_pole
        }
        
        names = list(vectors.keys())
        sim_matrix = np.zeros((len(names), len(names)))
        
        for i, (name1, vec1) in enumerate(vectors.items()):
            for j, (name2, vec2) in enumerate(vectors.items()):
                sim_matrix[i, j] = cosine_similarity(
                    vec1.reshape(1, -1),
                    vec2.reshape(1, -1)
                )[0][0]
        
        # Create heatmap
        plt.figure(figsize=(10, 8))
        plt.imshow(sim_matrix, cmap='coolwarm', vmin=-1, vmax=1)
        
        # Add text annotations
        for i in range(len(names)):
            for j in range(len(names)):
                plt.text(j, i, f"{sim_matrix[i, j]:.2f}",
                        ha='center', va='center', color='black' if abs(sim_matrix[i, j]) < 0.6 else 'white')
        
        # Customize the plot
        plt.xticks(np.arange(len(names)), names, rotation=45)
        plt.yticks(np.arange(len(names)), names)
        plt.title('Semantic Similarity Heatmap', pad=20)
        plt.colorbar(label='Cosine Similarity')
        
        # Save the figure
        fig_path = os.path.join(self.output_dir, 'similarity_heatmap.png')
        plt.tight_layout()
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"- Similarity heatmap saved to {fig_path}")


if __name__ == "__main__":
    # Initialize and run the TSR Field Engineer with conservative beta
    engineer = TSR_FieldEngineer(beta=0.15)  # Using a more conservative beta
    
    # Run with visualization and beta exploration
    print("\n" + "="*80)
    print("RUNNING ENHANCED TSR FIELD ENGINEERING ANALYSIS")
    print("="*80)
    engineer.run(visualize=True, explore_betas=True)
    
    print("\nAnalysis complete! Check the 'tsr_results' directory for visualizations and detailed results.")
    print("Key files generated:")
    print("- semantic_space_enhanced.png: 2D visualization of the semantic space")
    print("- similarity_radar.png: Radar chart of similarity changes")
    print("- similarity_heatmap.png: Heatmap of vector similarities")
    print("- beta_analysis.png: Effect of different beta values")
