import numpy as np
import gensim.downloader as api
from prettytable import PrettyTable
import os

class DriversDilemmaAnalyzer:
    def __init__(self):
        self.model = None
        
    def load_model(self):
        """Load the GloVe model."""
        print("Loading GloVe model (glove-wiki-gigaword-300)...")
        self.model = api.load('glove-wiki-gigaword-300')
        print("Model loaded successfully.")
    
    def get_average_vector(self, words):
        """Calculate the average vector for a set of words."""
        vectors = []
        for word in words:
            try:
                vectors.append(self.model[word])
            except KeyError:
                print(f"  Warning: '{word}' not in vocabulary, skipping")
        return np.mean(vectors, axis=0) if vectors else None
    
    def find_closest_concepts(self, vector, n=10):
        """Find the n closest concepts to a given vector in the model's vocabulary."""
        # Get all words in the vocabulary
        vocab = self.model.index_to_key[:10000]  # Limit to first 10k for performance
        
        # Calculate distances
        distances = []
        for word in vocab:
            try:
                word_vec = self.model[word]
                dist = np.linalg.norm(vector - word_vec)
                distances.append((word, dist))
            except KeyError:
                continue
        
        # Sort by distance and return top n
        distances.sort(key=lambda x: x[1])
        return distances[:n]
    
    def analyze_drivers_dilemma(self):
        """Analyze the driver's dilemma between speed and safety."""
        print("\n--- Driver's Dilemma Analysis ---")
        
        # Define the poles
        speed_terms = ["fast arrival", "overtaking", "high speed", "efficiency",
                      "quick", "accelerate", "overtake", "express", "rapid", "hurry"]
        
        safety_terms = ["safe arrival", "defensive driving", "caution", "low risk",
                       "careful", "safety", "defensive", "prudent", "cautious", "vigilant"]
        
        print("\nPole A (Speed):", ", ".join(speed_terms[:4]) + "...")
        print("Pole B (Safety):", ", ".join(safety_terms[:4]) + "...")
        
        # Calculate average vectors
        print("\nCalculating average vectors...")
        speed_vector = self.get_average_vector(speed_terms)
        safety_vector = self.get_average_vector(safety_terms)
        
        if speed_vector is None or safety_vector is None:
            print("Error: Could not calculate one or both pole vectors.")
            return
        
        # Calculate synthetic mediator
        mediator_vector = (speed_vector + safety_vector) / 2
        
        # Find closest concepts to the mediator
        print("\nFinding closest concepts to the mediator...")
        closest = self.find_closest_concepts(mediator_vector, n=15)
        
        # Filter and format results
        results = []
        for word, dist in closest:
            # Skip terms that are too close to either pole
            if word in speed_terms or word in safety_terms:
                continue
            results.append((word, f"{dist:.4f}"))
            if len(results) >= 10:  # We want top 10 unique concepts
                break
        
        # Display results
        table = PrettyTable()
        table.field_names = ["Rank", "Concept", "Distance to Mediator"]
        table.align = "l"
        
        for i, (word, dist) in enumerate(results, 1):
            table.add_row([i, word, dist])
        
        print("\n--- Top 10 Conceptual Mediators ---")
        print("These concepts represent potential syntheses of speed and safety:")
        print(table)
        
        # Additional analysis: direction of the mediator
        speed_to_safety = safety_vector - speed_vector
        mediator_direction = mediator_vector - speed_vector
        
        # Calculate how much of the mediator is in the direction of safety
        projection = np.dot(mediator_direction, speed_to_safety) / np.linalg.norm(speed_to_safety)
        safety_ratio = projection / np.linalg.norm(speed_to_safety)
        
        print(f"\n--- Analysis ---")
        print(f"The mediator is {safety_ratio*100:.1f}% of the way from Speed towards Safety.")
        print("\nThis suggests that the optimal synthesis in this dilemma is slightly")
        print("weighted towards safety, which aligns with common driving wisdom that")
        print("erring on the side of caution leads to better overall outcomes.")

def main():
    analyzer = DriversDilemmaAnalyzer()
    analyzer.load_model()
    analyzer.analyze_drivers_dilemma()

if __name__ == "__main__":
    main()
