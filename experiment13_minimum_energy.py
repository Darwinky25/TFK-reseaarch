# experiment13_minimum_energy.py
# This script tests the Principle of Minimum Energy in the TFK system.

import gensim.downloader as api
import numpy as np
from scipy.stats import pearsonr

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
    """Calculates the Euclidean distance, representing the 'Energy' of a contradiction."""
    return np.linalg.norm(vec1 - vec2)

def calculate_mediator(vec1, vec2):
    """Calculates the mediator (midpoint) of two vectors."""
    return (vec1 + vec2) / 2

def calculate_probability(mediator_vec, vec_c, vec_d, alpha=0.1):
    """Calculates the resolution probability based on the TFK formula."""
    combined_distance = calculate_distance(mediator_vec, vec_c) + calculate_distance(mediator_vec, vec_d)
    return np.exp(-alpha * combined_distance)

# --- Main Execution ---

def main():
    """Main function to run the Minimum Energy Principle analysis."""
    model = load_model()
    if model is None:
        return

    contradictions = [
        ("science", "religion"), ("freedom", "security"), ("justice", "mercy"),
        ("logic", "emotion"), ("innovation", "tradition"), ("individualism", "collectivism"),
        ("fate", "freewill"), ("matter", "spirit"), ("nature", "technology"),
        ("order", "chaos"), ("wealth", "happiness"), ("fact", "faith"),
        ("war", "peace"), ("temporary", "eternal"), ("strength", "weakness")
    ]

    # 1. Calculate Initial Total System Energy
    total_system_energy = 0
    for c1, c2 in contradictions:
        total_system_energy += calculate_distance(model[c1], model[c2])
    
    print("--- Analisis Prinsip Energi Minimum ---")
    print(f"Energi Total Sistem Awal: {total_system_energy:.4f}")

    results = []
    # 2. Loop through each contradiction as a trigger
    for trigger_c in contradictions:
        # a. Calculate Propagation Strength (Σ P_r)
        target_contradictions = [c for c in contradictions if c != trigger_c]
        mediator = calculate_mediator(model[trigger_c[0]], model[trigger_c[1]])
        propagation_strength = 0
        for c1, c2 in target_contradictions:
            propagation_strength += calculate_probability(mediator, model[c1], model[c2])

        # b. Calculate Expected Energy Reduction (ΔE)
        # Start with the energy of the trigger itself, as it's fully resolved.
        energy_reduction = calculate_distance(model[trigger_c[0]], model[trigger_c[1]])
        for c1, c2 in target_contradictions:
            # For others, the reduction is their energy * their probability of resolution
            p_r = calculate_probability(mediator, model[c1], model[c2])
            e_c = calculate_distance(model[c1], model[c2])
            energy_reduction += p_r * e_c
            
        results.append((trigger_c, propagation_strength, energy_reduction))

    # 3. Prepare data for output and correlation
    results.sort(key=lambda x: x[1], reverse=True) # Sort by propagation strength
    strengths = [res[1] for res in results]
    reductions = [res[2] for res in results]

    # 4. Display the results table
    print("\n+--------------------------------+------------------+------------------+")
    print("| Kontradiksi Pemicu             | Sum P_r (Riak)   | Delta E (Reduksi)|")
    print("+--------------------------------+------------------+------------------+")
    for (c1, c2), strength, reduction in results:
        c_str = f"('{c1}', '{c2}')"
        print(f"| {c_str:<30} | {strength:<16.4f} | {reduction:<16.4f} |")
    print("+--------------------------------+------------------+------------------+")

    # 5. Calculate and report correlation
    correlation, _ = pearsonr(strengths, reductions)
    print("\n--- Analisis Korelasi ---")
    print(f"Koefisien Korelasi Pearson (Riak vs. Reduksi): {correlation:.4f}")

    # 6. Conclusion
    print("\nConclusion:")
    if correlation > 0.9:
        print("The extremely high positive correlation strongly supports the Principle of Minimum Energy.")
        print("This suggests that the TFK system naturally favors pathways of change that are most efficient at reducing total systemic conflict.")
    else:
        print("The correlation is not strong enough to definitively support the Principle of Minimum Energy.")

if __name__ == "__main__":
    main()
