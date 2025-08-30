import numpy as np
import gensim.downloader as api
from prettytable import PrettyTable
import spacy
from tqdm import tqdm
import os

class FilteredMediatorSearch:
    def __init__(self, top_n_candidates=2000, top_n_results=10):
        self.model = None
        self.nlp = None
        self.top_n_candidates = top_n_candidates
        self.top_n_results = top_n_results
        
    def load_models(self):
        """Load both the GloVe model and spaCy model."""
        print("Loading GloVe model (glove-wiki-gigaword-300)...")
        self.model = api.load('glove-wiki-gigaword-300')
        print("Loading POS tagging model (spaCy)...")
        self.nlp = spacy.load("en_core_web_sm")
    
    def get_average_vector(self, terms):
        """Calculate the average vector for a set of terms, splitting multi-word terms."""
        vectors = []
        for term in terms:
            # Split term into individual words
            words = term.split()
            term_vectors = []
            for word in words:
                try:
                    term_vectors.append(self.model[word])
                except KeyError:
                    print(f"  Warning: '{word}' not in vocabulary, skipping")
            if term_vectors:
                # Average vectors for multi-word terms
                vectors.append(np.mean(term_vectors, axis=0))
        return np.mean(vectors, axis=0) if vectors else None
    
    def find_closest_concepts(self, vector, n=10, filter_pos=True):
        """Find the n closest concepts to a given vector, optionally filtering by POS."""
        # Get all words in the vocabulary
        vocab = self.model.index_to_key[:10000]  # Limit to first 10k for performance
        
        # Calculate distances
        print(f"Finding top {self.top_n_candidates} raw candidates...")
        distances = []
        for word in tqdm(vocab, desc="Calculating distances"):
            try:
                word_vec = self.model[word]
                dist = np.linalg.norm(vector - word_vec)
                distances.append((word, dist, word_vec))
            except KeyError:
                continue
        
        # Sort by distance
        distances.sort(key=lambda x: x[1])
        
        # Apply POS filtering if requested
        if filter_pos:
            print(f"Filtering {self.top_n_candidates} candidates for Nouns and Adjectives...")
            filtered = []
            for word, dist, vec in tqdm(distances[:self.top_n_candidates], desc="POS filtering"):
                # Process the word with spaCy
                doc = self.nlp(word)
                # Check if the word is a noun or adjective
                if doc and doc[0].pos_ in ['NOUN', 'ADJ']:
                    filtered.append((word, dist, vec))
                    if len(filtered) >= self.top_n_results:
                        break
            return filtered
        else:
            return distances[:n]
    
    def analyze_drivers_dilemma(self):
        """Analyze the driver's dilemma using filtered mediator search."""
        print("\n--- Driver's Dilemma Analysis (V2 with POS Filtering) ---")
        
        # Define the poles with more terms for better vector representation
        # Multi-word terms are kept as is and will be split in get_average_vector
        speed_terms = ["fast", "arrival", "overtaking", "high", "speed", "efficiency",
                      "quick", "accelerate", "overtake", "express", "rapid", "hurry",
                      "velocity", "swift", "brisk", "expedite"]
        
        safety_terms = ["safe", "arrival", "defensive", "driving", "caution", "low", "risk",
                       "careful", "safety", "defensive", "prudent", "cautious", "vigilant",
                       "security", "protection", "precaution", "carefulness", "alertness"]
        
        print("\nPole A (Speed):", ", ".join(speed_terms[:5]) + "...")
        print("Pole B (Safety):", ", ".join(safety_terms[:5]) + "...")
        
        # Calculate average vectors
        print("\nCalculating pole vectors and synthetic mediator...")
        speed_vector = self.get_average_vector(speed_terms)
        safety_vector = self.get_average_vector(safety_terms)
        
        if speed_vector is None or safety_vector is None:
            print("Error: Could not calculate one or both pole vectors.")
            return
        
        # Calculate synthetic mediator (midpoint between the two poles)
        mediator_vector = (speed_vector + safety_vector) / 2
        
        # Find closest concepts with POS filtering
        filtered_results = self.find_closest_concepts(
            mediator_vector, 
            n=self.top_n_candidates,
            filter_pos=True
        )
        
        # Display results
        table = PrettyTable()
        table.field_names = ["Rank", "Concept", "Distance to Mediator"]
        table.align = "l"
        
        for i, (word, dist, _) in enumerate(filtered_results[:self.top_n_results], 1):
            table.add_row([i, word, f"{dist:.4f}"])
        
        print("\n--- Top 10 Filtered Conceptual Mediators (Nouns & Adjectives) ---")
        print(table)
        
        # Additional analysis: direction of the mediator
        speed_to_safety = safety_vector - speed_vector
        mediator_direction = mediator_vector - speed_vector
        
        # Calculate how much of the mediator is in the direction of safety
        projection = np.dot(mediator_direction, speed_to_safety) / np.linalg.norm(speed_to_safety)
        safety_ratio = projection / np.linalg.norm(speed_to_safety)
        
        print(f"\n--- Analysis ---")
        print(f"The mediator is {safety_ratio*100:.1f}% of the way from Speed towards Safety.")
        print("\nThese filtered results reveal more substantive, domain-relevant concepts")
        print("that represent the synthesis of speed and safety in driving.")

def main():
    analyzer = FilteredMediatorSearch(top_n_candidates=2000, top_n_results=10)
    analyzer.load_models()
    analyzer.analyze_drivers_dilemma()

if __name__ == "__main__":
    main()
