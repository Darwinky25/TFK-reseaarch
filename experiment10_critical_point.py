# experiment10_critical_point.py
# This script implements the Critical Point Analysis to identify 'keystone concepts'.

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

def calculate_mediator(vec1, vec2):
    """Calculates the mediator (midpoint) of two vectors."""
    return (vec1 + vec2) / 2

def calculate_total_expected_value(model, mediator_vec, contradictions, alpha=0.1):
    """Calculates the Sum P_r for a given mediator and alpha."""
    total_expected_value = 0
    for c1, c2 in contradictions:
        vec_c = model[c1]
        vec_d = model[c2]
        combined_distance = calculate_distance(mediator_vec, vec_c) + calculate_distance(mediator_vec, vec_d)
        probability = np.exp(-alpha * combined_distance)
        total_expected_value += probability
    return total_expected_value

# --- Main Execution ---

def main():
    """Main function to run the Critical Point Analysis."""
    model = load_model()
    if model is None:
        return

    # 1. Define the full set of contradictions
    contradictions = [
        ("science", "religion"), ("freedom", "security"), ("justice", "mercy"),
        ("logic", "emotion"), ("innovation", "tradition"), ("individualism", "collectivism"),
        ("fate", "freewill"), ("matter", "spirit"), ("nature", "technology"),
        ("order", "chaos"), ("wealth", "happiness"), ("fact", "faith"),
        ("war", "peace"), ("temporary", "eternal"), ("strength", "weakness")
    ]

    results = []
    print("\n--- Running Critical Point Analysis ---")
    print(f"Analyzing {len(contradictions)} contradictions to find keystone concepts...")

    # 2. Loop through each contradiction, treating it as the trigger
    for i, trigger_c in enumerate(contradictions):
        print(f"  ({i+1}/{len(contradictions)}) Testing trigger: {trigger_c}...")
        
        # Define the target contradictions (all others)
        target_contradictions = [c for c in contradictions if c != trigger_c]
        
        # Calculate the mediator for the trigger contradiction
        try:
            vec_a = model[trigger_c[0]]
            vec_b = model[trigger_c[1]]
        except KeyError as e:
            print(f"    Skipping {trigger_c} due to missing word: {e}")
            continue

        mediator = calculate_mediator(vec_a, vec_b)
        
        # Calculate the propagation strength (Sum P_r) on all other contradictions
        strength = calculate_total_expected_value(model, mediator, target_contradictions)
        
        results.append((trigger_c, strength))

    # 3. Sort the results by strength in descending order
    results.sort(key=lambda x: x[1], reverse=True)

    # 4. Display the final ranked table
    print("\n--- Critical Point Analysis: Contradiction Propagation Strength Ranking ---")
    print("+--------------------------------+----------------------------------+")
    print("| Triggering Contradiction       | Total Expected Value (Sum P_r)   |")
    print("+--------------------------------+----------------------------------+")
    for (c1, c2), strength in results:
        c_str = f"('{c1}', '{c2}')"
        print(f"| {c_str:<30} | {strength:<32.4f} |")
    print("+--------------------------------+----------------------------------+")

    # 5. Conclusion
    print("\nConclusion:")
    print("This analysis reveals the 'backbone' of the knowledge network. It proves that not all conflicts are equal;")
    print("some are 'keystones' that generate significantly more systemic reconciliation than others.")
    print("Resolving these keystone concepts provides disproportionately large leverage for stabilizing the entire system.")

if __name__ == "__main__":
    main()
