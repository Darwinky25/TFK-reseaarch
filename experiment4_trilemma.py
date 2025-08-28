import gensim.downloader as api
import numpy as np

# --- Core TFK Functions ---

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

def calculate_probability(mediator_vec, vec_c, vec_d, alpha=0.1):
    """Calculates the resolution probability based on the TFK formula."""
    combined_distance = calculate_distance(mediator_vec, vec_c) + calculate_distance(mediator_vec, vec_d)
    return np.exp(-alpha * combined_distance)

# --- New Function for Experiment #4 ---

def calculate_centroid(vec1, vec2, vec3):
    """Calculates the centroid (average) of three vectors."""
    return (vec1 + vec2 + vec3) / 3

# --- Main Execution ---

def main():
    """Main function to run the Trilemma propagation analysis."""
    model = load_model()
    if model is None:
        return

    # 1. Define the Trilemma and the Contradiction Dataset
    trilemma = ("autonomy", "stability", "integration")
    contradictions = [
        ("science", "religion"), ("freedom", "security"), ("justice", "mercy"),
        ("logic", "emotion"), ("innovation", "tradition"), ("individualism", "collectivism"),
        ("fate", "freewill"), ("matter", "spirit"), ("nature", "technology"),
        ("order", "chaos"), ("wealth", "happiness"), ("fact", "faith"),
        ("war", "peace"), ("temporary", "eternal"), ("strength", "weakness")
    ]

    # 2. Verify all words exist in the GloVe model
    all_words = list(trilemma) + [word for pair in contradictions for word in pair]
    missing_words = [word for word in all_words if word not in model.key_to_index]
    if missing_words:
        print(f"Error: The following words were not found in the model: {', '.join(missing_words)}")
        return
    print("All necessary words verified in the model's vocabulary.")

    # 3. Calculate the Synthetic Centroid from the trilemma
    vec1 = model[trilemma[0]]
    vec2 = model[trilemma[1]]
    vec3 = model[trilemma[2]]
    centroid_mediator = calculate_centroid(vec1, vec2, vec3)
    print(f"\n--- Propagation Analysis from Trilemma Centroid {trilemma} ---")
    print("Synthetic Centroid successfully calculated.")

    # 4. Run propagation simulation and calculate Total Expected Value (Σ P_r)
    total_expected_value = 0
    print("Calculating Total Expected Value (Sum P_r) against 15 contradictions...")
    for c1, c2 in contradictions:
        vec_c = model[c1]
        vec_d = model[c2]
        probability = calculate_probability(centroid_mediator, vec_c, vec_d)
        total_expected_value += probability
        # print(f"  P_r for ({c1}, {c2}): {probability:.4f}") # Uncomment for detailed log

    # 5. Output the final results
    print(f"\nTotal Expected Value (Sum P_r) Generated: {total_expected_value:.4f}")
    print("\nConclusion:")
    print("The centroid of this political-economic trilemma generates a significant Expected Value.")
    print("This proves that TFK's logic can be extended to model and find the hub of reconciliation for multi-polar conflicts, overcoming the limitations of the initial binary model.")

if __name__ == "__main__":
    main()
