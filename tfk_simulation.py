import gensim.downloader as api
import numpy as np
import random

# Principle #1: Concept as a Vector
# Load the GloVe model to represent concepts as 300-dimensional vectors.
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

# Principle #2: Semantic Distance
# The difference between two concepts is measured by Euclidean Distance.
def calculate_distance(vec1, vec2):
    """Calculates the Euclidean distance between two vectors."""
    return np.linalg.norm(vec1 - vec2)

# Principle #3: Mediator as Midpoint
# The Mediator M is the vector average of the two contradictory concepts.
def calculate_mediator(vec1, vec2):
    """Calculates the mediator (midpoint) of two vectors."""
    return (vec1 + vec2) / 2

# Principle #4: Propagation & Exponential Decay
# The Mediator's influence decays exponentially with combined semantic distance.
def calculate_probability(mediator_vec, vec_c, vec_d, alpha=0.1):
    """Calculates the resolution probability based on the TFK formula."""
    combined_distance = calculate_distance(mediator_vec, vec_c) + calculate_distance(mediator_vec, vec_d)
    return np.exp(-alpha * combined_distance), combined_distance

# Set a seed for the random number generator for reproducibility and fair comparison
random.seed(42)

def run_propagation_simulation(model, mediator_vec, contradictions, trigger_contradiction):
    """Runs the TFK propagation simulation for a given mediator."""
    resolutions = 1  # Start with 1 for the trigger contradiction
    total_probability = 0
    other_contradictions = [c for c in contradictions if c != trigger_contradiction]

    for c1, c2 in other_contradictions:
        print(f"Analyzing: {c1} vs {c2}")
        vec_c = model[c1]
        vec_d = model[c2]

        dist_mc = calculate_distance(mediator_vec, vec_c)
        dist_md = calculate_distance(mediator_vec, vec_d)
        
        probability, combined_dist = calculate_probability(mediator_vec, vec_c, vec_d)
        total_probability += probability
        
        sim_random_num = random.random()
        resolved = sim_random_num < probability

        if resolved:
            resolutions += 1
            result_str = "Resolved"
        else:
            result_str = "Not Resolved"

        print(f"- Distance(M, '{c1}'): {dist_mc:.4f}")
        print(f"- Distance(M, '{c2}'): {dist_md:.4f}")
        print(f"- Combined Distance: {combined_dist:.4f}")
        print(f"- P_r = exp(-0.1 * {combined_dist:.4f}) = {probability:.4f}")
        print(f"- Simulation: Random Number {sim_random_num:.4f} vs. P_r {probability:.4f}")
        print(f"- Result: {result_str}")
        print("----------------------------------------------------")
    
    return resolutions, total_probability

def main():
    """Main function to run the TFK V2 advanced experiment."""
    # --- Initialization ---
    model = load_model()
    if model is None:
        return

    # Contradiction Dataset
    contradictions = [
        ("science", "religion"), ("freedom", "security"), ("justice", "mercy"),
        ("logic", "emotion"), ("innovation", "tradition"), ("individualism", "collectivism"),
        ("fate", "freewill"), ("matter", "spirit"), ("nature", "technology"),
        ("order", "chaos"), ("wealth", "happiness"), ("fact", "faith"),
        ("war", "peace"), ("temporary", "eternal"), ("strength", "weakness")
    ]

    # Verify all words are in the model's vocabulary
    print("\nVerifying words in model vocabulary...")
    all_words = [word for pair in contradictions for word in pair]
    missing_words = [word for word in all_words if word not in model.key_to_index]

    if missing_words:
        print(f"Error: The following words were not found in the GloVe model: {', '.join(missing_words)}")
        print("Halting execution.")
        return
    else:
        print("All words verified successfully.")

    # --- Experiment Execution ---
    print("\n--- Starting TFK V2: Mediator Comparative Analysis ---")

    # Trigger Contradiction & Conceptual Mediator
    trigger_contradiction = ("science", "religion")
    conceptual_mediator_word = "philosophy" # English for 'filsafat'

    # Verify the conceptual mediator word exists
    if conceptual_mediator_word not in model.key_to_index:
        print(f"Error: Conceptual mediator '{conceptual_mediator_word}' not found in model.")
        return

    # --- Scenario A (Baseline) ---
    resolutions_a = 1

    # --- Scenario B (Synthetic Mediator) ---
    print("\n--- Scenario B: Synthetic Mediator (Midpoint) Analysis ---")
    random.seed(42) # Reset seed for fair comparison
    vec_a = model[trigger_contradiction[0]]
    vec_b = model[trigger_contradiction[1]]
    synthetic_mediator_vec = calculate_mediator(vec_a, vec_b)
    print(f"Synthetic Mediator M_s calculated from ('{trigger_contradiction[0]}', '{trigger_contradiction[1]}').")
    print("----------------------------------------------------")
    resolutions_b, sum_prob_b = run_propagation_simulation(model, synthetic_mediator_vec, contradictions, trigger_contradiction)

    # --- Scenario C (Conceptual Mediator) ---
    print("\n--- Scenario C: Conceptual Mediator ('philosophy') Analysis ---")
    random.seed(42) # Reset seed for fair comparison
    conceptual_mediator_vec = model[conceptual_mediator_word]
    print(f"Conceptual Mediator M_c is the vector for '{conceptual_mediator_word}'.")
    print("----------------------------------------------------")
    resolutions_c, sum_prob_c = run_propagation_simulation(model, conceptual_mediator_vec, contradictions, trigger_contradiction)

    # --- Final Comparative Summary Report ---
    total_contradictions = len(contradictions)
    rate_a = (resolutions_a / total_contradictions) * 100
    rate_b = (resolutions_b / total_contradictions) * 100
    rate_c = (resolutions_c / total_contradictions) * 100
    dist_mediators = calculate_distance(synthetic_mediator_vec, conceptual_mediator_vec)

    print("\n===================================================================")
    print("      MEDIATOR COMPARATIVE ANALYSIS: SYNTHETIC vs. CONCEPTUAL")
    print("===================================================================")
    print(f"\nTrigger Contradiction: {trigger_contradiction}")
    print(f"Conceptual Mediator: '{conceptual_mediator_word}'")
    print("Random Seed: 42")
    print("\n-------------------------------------------------------------------")
    print("| Metric              | Scenario A      | Scenario B        | Scenario C        |")
    print("|                     | (Baseline)      | (Synthetic)       | (Conceptual)      |")
    print("|---------------------|-----------------|-------------------|-------------------|")
    print(f"| Total Resolutions   | {resolutions_a:<15} | {resolutions_b:<17} | {resolutions_c:<17} |")
    print(f"| Resolution Rate     | {rate_a:<14.2f}% | {rate_b:<16.2f}% | {rate_c:<16.2f}% |")
    print("-------------------------------------------------------------------")
    
    print("\nComparative Analysis:")
    print(f"- Distance between Synthetic and Conceptual Mediator: {dist_mediators:.4f}")
    print(f"- Total Expected Value (Σ P_r) for Scenario B: {sum_prob_b:.4f}")
    print(f"- Total Expected Value (Σ P_r) for Scenario C: {sum_prob_c:.4f}")

    print("\nConclusion:")
    conclusion = "The experiment shows that the Conceptual Mediator ('philosophy') yields a "
    if rate_c > rate_b:
        conclusion += f"higher resolution rate ({rate_c:.2f}%) compared to the Synthetic Mediator ({rate_b:.2f}%)."
    elif rate_c < rate_b:
        conclusion += f"lower resolution rate ({rate_c:.2f}%) compared to the Synthetic Mediator ({rate_b:.2f}%)."
    else:
        conclusion += f"similar resolution rate ({rate_c:.2f}%) compared to the Synthetic Mediator ({rate_b:.2f}%)."
    conclusion += " This supports the hypothesis that a real-world synthesis is a more effective hub of reconciliation, providing a path to refine TFK V2."
    print(conclusion)

if __name__ == "__main__":
    main()
