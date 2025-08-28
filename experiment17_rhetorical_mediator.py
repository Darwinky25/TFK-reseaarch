import numpy as np
import gensim.downloader as api
from sklearn.metrics.pairwise import euclidean_distances
from prettytable import PrettyTable

def calculate_distance(vec1, vec2):
    """Calculate Euclidean distance between two vectors."""
    return np.linalg.norm(vec1 - vec2)

def calculate_mediator(vec1, vec2):
    """Calculate the mediator vector between two vectors."""
    return (vec1 + vec2) / 2

def get_pole_vector(model, word_list):
    """Calculate the average vector for a list of words."""
    vectors = []
    for word in word_list:
        try:
            vectors.append(model[word])
        except KeyError:
            print(f"Warning: '{word}' not found in the model's vocabulary.")
    
    if not vectors:
        raise ValueError("No valid words found in the model's vocabulary.")
    
    return np.mean(vectors, axis=0)

def find_closest_concepts(model, target_vector, top_n=10, vocab_limit=20000):
    """Find the closest concepts to the target vector in the model's vocabulary."""
    distances = []
    vocab = list(model.key_to_index.keys())[:vocab_limit]
    
    for word in vocab:
        try:
            word_vec = model[word]
            dist = calculate_distance(target_vector, word_vec)
            distances.append((word, dist))
        except KeyError:
            continue
    
    # Sort by distance and return top_n results
    distances.sort(key=lambda x: x[1])
    return distances[:top_n]

def main():
    print("Loading GloVe model (glove-wiki-gigaword-300)...")
    model = api.load("glove-wiki-gigaword-300")
    print("Model loaded successfully.\n")
    
    print("--- Rhetorical Mediator Analysis ---")
    print("Analyzing the tension between 'Technical' and 'Impactful' writing styles...\n")
    
    # Define rhetorical poles
    pole_a = ['technical', 'precise', 'rigorous', 'quantitative', 
              'analytical', 'methodology', 'data']
    pole_b = ['inspiring', 'impactful', 'narrative', 'emotional', 
              'elegant', 'profound', 'discovery']
    
    # Calculate pole vectors
    print("- Calculating pole vectors...")
    vector_pole_a = get_pole_vector(model, pole_a)
    vector_pole_b = get_pole_vector(model, pole_b)
    
    # Calculate synthetic mediator
    print("- Calculating synthetic mediator...")
    mediator = calculate_mediator(vector_pole_a, vector_pole_b)
    
    # Find closest concepts to the mediator
    print("- Searching for the 10 closest conceptual mediators in the vocabulary...\n")
    closest_concepts = find_closest_concepts(model, mediator)
    
    # Prepare and display results
    table = PrettyTable()
    table.field_names = ["Rank", "Concept (Mediator)", "Distance"]
    table.align["Concept (Mediator)"] = "l"
    table.align["Distance"] = "r"
    
    for i, (word, dist) in enumerate(closest_concepts, 1):
        table.add_row([i, word, f"{dist:.4f}"])
    
    print("--- Top 10 Conceptual Mediators for Scientific Writing ---")
    print(table)
    
    print("""
--- Conclusion ---
These concepts represent the ideal synthesis of rigor and narrative. They will serve as the guiding principles for drafting the TFK research paper, ensuring it is both scientifically valid and intellectually compelling.""")

if __name__ == "__main__":
    main()
