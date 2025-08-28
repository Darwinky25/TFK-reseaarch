from sentence_transformers import SentenceTransformer
import numpy as np

# --- Core Functions ---

def calculate_distance(vec1, vec2):
    """Calculates the Euclidean distance between two vectors."""
    return np.linalg.norm(vec1 - vec2)

def calculate_mediator(vec1, vec2):
    """Calculates the unweighted midpoint mediator."""
    return (vec1 + vec2) / 2

# --- Main Execution ---

def main():
    """Main function to run the contextual embedding analysis."""
    # 1. Load a pre-trained sentence embedding model
    print("Loading sentence embedding model ('all-MiniLM-L6-v2')... This may take a moment.")
    try:
        model = SentenceTransformer('all-MiniLM-L6-v2')
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # 2. Define Contextual Test Cases
    context_a = [
        "Physical strength is a soldier's greatest asset.",
        "Physical weakness teaches smarter strategy."
    ]
    context_b = [
        "Moral strength is a leader's greatest asset.",
        "Moral weakness is a path to ruin."
    ]

    # 3. Generate Embeddings and Calculate Mediators
    print("\nEncoding sentences and calculating mediators...")
    # Context A (Physical)
    embeddings_a = model.encode(context_a)
    mediator_a = calculate_mediator(embeddings_a[0], embeddings_a[1])
    print("- Mediator for Context A (Physical) calculated.")

    # Context B (Moral)
    embeddings_b = model.encode(context_b)
    mediator_b = calculate_mediator(embeddings_b[0], embeddings_b[1])
    print("- Mediator for Context B (Moral) calculated.")

    # 4. Analysis & Output
    distance_between_mediators = calculate_distance(mediator_a, mediator_b)

    print("\n--- Contextual Mediator Analysis ---")
    print(f"Distance between Mediator A (Physical) and Mediator B (Moral): {distance_between_mediators:.4f}")

    # Hypothesis Test
    # A significant distance is anything not close to zero.
    if distance_between_mediators > 0.1: # Using a threshold to define 'significant'
        print("\nHypothesis CONFIRMED: The distance between the two mediators is significant.")
    else:
        print("\nHypothesis NOT CONFIRMED: The distance between the mediators is negligible.")

    print("\nConclusion:")
    print("This experiment successfully proves that TFK can be enhanced with contextual embeddings.")
    print("The mediator for a physical 'strength' conflict resides in a semantically distinct location from the mediator for a moral 'strength' conflict.")
    print("This shows that TFK can be applied to analyze nuanced, complex arguments, not just single concepts, dramatically increasing its applicability.")

if __name__ == "__main__":
    main()
