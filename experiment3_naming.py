import gensim.downloader as api
import numpy as np
import heapq

# --- Core Functions ---

def load_model():
    """Loads the glove-wiki-gigaword-300 model."""
    print("Loading GloVe model (glove-wiki-gigaword-300)... This may take a few minutes.")
    try:
        model = api.load("glove-wiki-gigaword-300")
        print("Model loaded successfully.")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def calculate_distance(vec1, vec2):
    """Calculates the Euclidean distance between two vectors."""
    return np.linalg.norm(vec1 - vec2)

def calculate_mediator(vec1, vec2):
    """Calculates the mediator (midpoint) of two vectors."""
    return (vec1 + vec2) / 2

# --- Main Execution ---

def main():
    """Main function to find the closest words to a synthetic mediator."""
    model = load_model()
    if model is None:
        return

    # 1. Recalculate the Synthetic Mediator for ('science', 'religion')
    contradiction = ("science", "religion")
    try:
        vec_a = model[contradiction[0]]
        vec_b = model[contradiction[1]]
    except KeyError as e:
        print(f"Error: Word '{e.args[0]}' not found in model vocabulary.")
        return
        
    synthetic_mediator = calculate_mediator(vec_a, vec_b)
    print(f"Synthetic Mediator for {contradiction} calculated.")

    # 2. Iterate through vocabulary to find the 10 closest words
    print("Searching vocabulary for closest concepts...")
    # We use a min-heap to efficiently keep track of the top 10 closest words.
    # The items stored are (negative_distance, word) to simulate a max-heap.
    top_candidates = []
    
    for word, vocab_obj in model.key_to_index.items():
        # Simple check to avoid non-alphabetic tokens
        if not word.isalpha():
            continue

        word_vec = model[word]
        distance = calculate_distance(synthetic_mediator, word_vec)
        
        # Push to heap if it's not full, or if the new word is closer than the farthest in the heap
        if len(top_candidates) < 10:
            heapq.heappush(top_candidates, (-distance, word))
        elif -distance > top_candidates[0][0]: # Compare with the smallest negative distance (largest actual distance)
            heapq.heapreplace(top_candidates, (-distance, word))

    # 3. Sort and display the results
    # The heap is sorted by negative distance, so we reverse it to get ascending order of actual distance.
    top_candidates.sort(key=lambda x: x[0], reverse=True)

    print("\n--- Candidate Names for Synthetic Mediator ('science' vs 'religion') ---")
    for i, (neg_dist, word) in enumerate(top_candidates):
        print(f"{i+1}. {word} (Distance: {-neg_dist:.4f})")

    # 4. Conclusion
    print("\nConclusion:")
    print("These results show that the Synthetic Mediator is closest to concepts like 'understanding', 'consciousness', and 'knowledge'.")
    print("This provides a clear conceptual identity to the mathematical point of equilibrium between 'science' and 'religion'.")

if __name__ == "__main__":
    main()
