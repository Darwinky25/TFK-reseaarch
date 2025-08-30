import numpy as np
import gensim.downloader as api
from prettytable import PrettyTable
import matplotlib.pyplot as plt
import os

class MethodologicalPowerTest:
    def __init__(self, alpha=0.1):
        self.model = None
        self.alpha = alpha
        self.contradictions = [
            ("tradition", "innovation"),
            ("freedom", "security"),
            ("individual", "community"),
            ("stability", "change"),
            ("logic", "intuition"),
            ("theory", "practice"),
            ("art", "science"),
            ("work", "life"),
            ("nature", "nurture"),
            ("mind", "body"),
            ("emotion", "reason"),
            ("local", "global"),
            ("rights", "responsibilities"),
            ("competition", "cooperation"),
            ("risk", "safety")
        ]
        
    def load_model(self):
        """Load the GloVe model."""
        print("Loading GloVe model (glove-wiki-gigaword-300)...")
        self.model = api.load('glove-wiki-gigaword-300')
        
    def get_vector(self, word):
        """Get vector for a single word."""
        try:
            return self.model[word]
        except KeyError:
            print(f"  Warning: '{word}' not in vocabulary")
            return None
            
    def get_average_vector(self, words):
        """Calculate the average vector for a set of words."""
        vectors = []
        for word in words:
            vec = self.get_vector(word)
            if vec is not None:
                vectors.append(vec)
        return np.mean(vectors, axis=0) if vectors else None
    
    def calculate_propagation_strength(self, trigger, target):
        """Calculate the propagation strength from trigger to target."""
        trigger_vec = self.get_vector(trigger) if isinstance(trigger, str) else trigger
        target_vec = self.get_vector(target) if isinstance(target, str) else target
        
        if trigger_vec is None or target_vec is None:
            return 0.0
            
        # Calculate Euclidean distance
        distance = np.linalg.norm(trigger_vec - target_vec)
        # Calculate propagation strength (P_r)
        return np.exp(-self.alpha * distance)
    
    def calculate_system_propagation(self, mediator):
        """Calculate total propagation strength for all contradictions."""
        total = 0.0
        for pole_a, pole_b in self.contradictions:
            # Calculate P_r for both poles to the mediator
            pr_a = self.calculate_propagation_strength(pole_a, mediator)
            pr_b = self.calculate_propagation_strength(pole_b, mediator)
            total += pr_a + pr_b
        return total
    
    def run_experiment(self):
        """Run the methodological power test."""
        print("\n--- Methodological Power Test ---")
        print("Comparing three mediator types for the Driver's Dilemma...\n")
        
        # Define poles
        speed_terms = ["fast", "speed", "quick", "rapid", "accelerate"]
        safety_terms = ["safe", "safety", "careful", "cautious", "defensive"]
        
        # Calculate pole vectors
        print("- Calculating Pole A (Speed) and Pole B (Safety) vectors...")
        pole_a = self.get_average_vector(speed_terms)
        pole_b = self.get_average_vector(safety_terms)
        
        if pole_a is None or pole_b is None:
            print("Error: Could not calculate pole vectors.")
            return
        
        # Define mediators
        print("- Calculating Mediator A (Intuitive Answer)...")
        mediator_a_terms = ["awareness", "skill", "balance", "judgment", "control"]
        mediator_a = self.get_average_vector(mediator_a_terms)
        
        print("- Calculating Mediator B (Reasoning Method)...")
        mediator_b_terms = ["reason", "fact", "contrast", "instance", "example"]
        mediator_b = self.get_average_vector(mediator_b_terms)
        
        print("- Calculating Mediator C (Synthetic Control)...")
        mediator_c = (pole_a + pole_b) / 2  # Synthetic mediator
        
        # Calculate propagation strengths
        print("\n- Calculating propagation strengths...")
        results = []
        
        for name, mediator in [
            ("A: Intuitive Answer", mediator_a),
            ("B: Reasoning Method", mediator_b),
            ("C: Synthetic Control", mediator_c)
        ]:
            sigma_pr = self.calculate_system_propagation(mediator)
            results.append((name, sigma_pr))
        
        # Sort by propagation strength (highest first)
        results.sort(key=lambda x: x[1], reverse=True)
        
        # Display results
        self.display_results(results)
        
        # Plot results
        self.plot_results(results)
    
    def display_results(self, results):
        """Display the results in a formatted table."""
        # Create a table that works with Windows console encoding
        print("\n--- Propagation Strength Comparison ---")
        print("+" + "-"*30 + "+" + "-"*20 + "+")
        print(f"| {'Mediator Type':<28} | {'Total Expected Value':<18} |")
        print("+" + "-"*30 + "+" + "-"*20 + "+")
        
        for name, sigma_pr in results:
            print(f"| {name:<28} | {sigma_pr:18.4f} |")
        
        print("+" + "-"*30 + "+" + "-"*20 + "+")
        
        # Print interpretation
        print("\n--- Interpretation ---")
        best_name, best_score = results[0]
        print(f"\nThe most powerful mediator is: {best_name} (Sum P_r = {best_score:.4f})")
        
        if "Reasoning Method" in best_name:
            print("\nThis supports our hypothesis that reasoning methods are more powerful "
                  "mediators than intuitive answers or simple compromises. The process "
                  "of reasoning itself serves as a better hub for reconciling "
                  "contradictions than any specific answer.")
    
    def plot_results(self, results):
        """Create a bar chart of the results."""
        names = [name.split(": ")[1] for name, _ in results]
        values = [score for _, score in results]
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(names, values, color=['#4c72b0', '#55a868', '#c44e52'])
        
        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{height:.2f}',
                    ha='center', va='bottom')
        
        plt.title('Propagation Strength by Mediator Type', fontsize=14)
        plt.ylabel('Total Expected Value (Σ P_r)')
        plt.ylim(0, max(values) * 1.2)  # Add some padding at the top
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Save the plot
        plt.tight_layout()
        plot_path = os.path.join(os.getcwd(), 'mediator_comparison.png')
        plt.savefig(plot_path)
        print(f"\nPlot saved to: {plot_path}")

def main():
    experiment = MethodologicalPowerTest(alpha=0.1)
    experiment.load_model()
    experiment.run_experiment()

if __name__ == "__main__":
    main()
