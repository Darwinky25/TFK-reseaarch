# tfk_validation_suite.py
# A comprehensive script to run all key TFK analyses, generate empirical data,
# and produce publication-ready tables and visualizations.

import gensim.downloader as api
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
from sklearn.cluster import KMeans

# --- Part 1: Initialization & Core Functions ---

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

def define_contradictions():
    """Returns the list of 15 core contradictions."""
    return [
        ("science", "religion"), ("freedom", "security"), ("justice", "mercy"),
        ("logic", "emotion"), ("innovation", "tradition"), ("individualism", "collectivism"),
        ("fate", "freewill"), ("matter", "spirit"), ("nature", "technology"),
        ("order", "chaos"), ("wealth", "happiness"), ("fact", "faith"),
        ("war", "peace"), ("temporary", "eternal"), ("strength", "weakness")
    ]

def calculate_distance(vec1, vec2):
    return np.linalg.norm(vec1 - vec2)

def calculate_mediator(vec1, vec2):
    return (vec1 + vec2) / 2

def calculate_probability(mediator_vec, vec_c, vec_d, alpha=0.1):
    combined_distance = calculate_distance(mediator_vec, vec_c) + calculate_distance(mediator_vec, vec_d)
    return np.exp(-alpha * combined_distance)

# --- Part 2: Analysis Engine ---

def run_critical_point_analysis(model, contradictions):
    results = []
    for trigger_c in contradictions:
        mediator = calculate_mediator(model[trigger_c[0]], model[trigger_c[1]])
        sum_pr = sum(calculate_probability(mediator, model[c1], model[c2]) for c1, c2 in contradictions if (c1, c2) != trigger_c)
        results.append({"Trigger Contradiction": f"('{trigger_c[0]}', '{trigger_c[1]}')", "Total Expected Value (Sum P_r)": sum_pr})
    return pd.DataFrame(results).sort_values(by="Total Expected Value (Sum P_r)", ascending=False).reset_index(drop=True)

def run_cluster_analysis(model, contradictions):
    concepts = sorted(list(set(c for pair in contradictions for c in pair)))
    vectors = np.array([model[concept] for concept in concepts])
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10).fit(vectors)
    concept_to_cluster = {concept: label for concept, label in zip(concepts, kmeans.labels_)}

    trigger_c = ('science', 'religion')
    trigger_cluster = concept_to_cluster[trigger_c[0]]
    mediator = calculate_mediator(model[trigger_c[0]], model[trigger_c[1]])

    intra_cluster_prs = [calculate_probability(mediator, model[c1], model[c2]) for c1, c2 in contradictions if c1 != trigger_c[0] and concept_to_cluster[c1] == trigger_cluster]
    inter_cluster_prs = [calculate_probability(mediator, model[c1], model[c2]) for c1, c2 in contradictions if concept_to_cluster[c1] != trigger_cluster]

    avg_intra_pr = np.mean(intra_cluster_prs)
    avg_inter_pr = np.mean(inter_cluster_prs)

    return pd.DataFrame([{"Category": "Intra-Cluster", "Average P_r": avg_intra_pr}, {"Category": "Inter-Cluster", "Average P_r": avg_inter_pr}])

def run_recursive_cascade_analysis(model, contradictions, trigger=("science", "religion")):
    # Initial resolution
    p_trigger = 1.0
    mediator_primary = calculate_mediator(model[trigger[0]], model[trigger[1]])
    
    # Primary cascade
    secondary_contradictions = [c for c in contradictions if c != trigger]
    primary_cascade_results = []
    for c1, c2 in secondary_contradictions:
        p_secondary = calculate_probability(mediator_primary, model[c1], model[c2])
        primary_cascade_results.append({"Secondary Contradiction": f"('{c1}', '{c2}')", "P_secondary": p_secondary})
    
    # Secondary cascade (E_contribution)
    total_e_contribution = 0
    for res in primary_cascade_results:
        c1, c2 = eval(res["Secondary Contradiction"]) # Quick way to get tuple back
        mediator_secondary = calculate_mediator(model[c1], model[c2])
        e_contribution = sum(calculate_probability(mediator_secondary, model[oc1], model[oc2]) for oc1, oc2 in secondary_contradictions if (oc1, oc2) != (c1, c2))
        res["E_contribution"] = res["P_secondary"] * e_contribution
        total_e_contribution += res["E_contribution"]

    df = pd.DataFrame(primary_cascade_results)
    sum_pr_secondary = df['P_secondary'].sum()
    summary = {"Initial Resolution": p_trigger, "Primary Effect (Sum P_r)": sum_pr_secondary, "Secondary Effect (Sum E)": total_e_contribution}
    return df, summary

def run_minimum_energy_analysis(model, contradictions, critical_point_df):
    results = []
    for trigger_c_str in critical_point_df["Trigger Contradiction"]:
        trigger_c = eval(trigger_c_str)
        mediator = calculate_mediator(model[trigger_c[0]], model[trigger_c[1]])
        energy_reduction = calculate_distance(model[trigger_c[0]], model[trigger_c[1]])
        for c1, c2 in contradictions:
            if (c1, c2) != trigger_c:
                p_r = calculate_probability(mediator, model[c1], model[c2])
                e_c = calculate_distance(model[c1], model[c2])
                energy_reduction += p_r * e_c
        results.append({"Trigger Contradiction": trigger_c_str, "Delta E (Reduction)": energy_reduction})
    
    energy_df = pd.DataFrame(results)
    merged_df = pd.merge(critical_point_df, energy_df, on="Trigger Contradiction")
    correlation, _ = pearsonr(merged_df["Total Expected Value (Sum P_r)"], merged_df["Delta E (Reduction)"])
    return merged_df, correlation

# --- Part 3: Output Generation & Visualization ---

def create_critical_point_plot(df):
    plt.figure(figsize=(12, 8))
    sns.barplot(x="Total Expected Value (Sum P_r)", y="Trigger Contradiction", data=df, palette="viridis")
    plt.title("Figure 1: Contradiction Propagation Strength (Σ P_r)", fontsize=16)
    plt.xlabel("Total Expected Value (Sum P_r)", fontsize=12)
    plt.ylabel("Trigger Contradiction", fontsize=12)
    plt.tight_layout()
    plt.savefig("figure_1_critical_points.png")
    print("Saved figure_1_critical_points.png")

def create_cluster_plot(df):
    plt.figure(figsize=(8, 6))
    sns.barplot(x="Category", y="Average P_r", data=df, palette="coolwarm")
    plt.title("Figure 2: Evidence of Knowledge Modularity", fontsize=16)
    plt.xlabel("Cluster Category", fontsize=12)
    plt.ylabel("Average Resolution Probability (P_r)", fontsize=12)
    plt.tight_layout()
    plt.savefig("figure_2_cluster_influence.png")
    print("Saved figure_2_cluster_influence.png")

def create_cascade_plot(summary):
    labels = list(summary.keys())
    values = list(summary.values())
    plt.figure(figsize=(8, 6))
    bars = plt.bar(["Total Cascade Effect"], [sum(values)], color='lightgray')
    plt.bar(["Total Cascade Effect"], [values[0]], color='skyblue', label=labels[0])
    plt.bar(["Total Cascade Effect"], [values[1]], bottom=[values[0]], color='salmon', label=labels[1])
    plt.bar(["Total Cascade Effect"], [values[2]], bottom=[values[0]+values[1]], color='lightgreen', label=labels[2])
    plt.title("Figure 3: Recursive Cascade and Criticality", fontsize=16)
    plt.ylabel("Effect Magnitude")
    plt.legend()
    plt.tight_layout()
    plt.savefig("figure_3_cascade_effect.png")
    print("Saved figure_3_cascade_effect.png")

def create_energy_scatterplot(df, correlation):
    plt.figure(figsize=(10, 8))
    sns.regplot(x="Delta E (Reduction)", y="Total Expected Value (Sum P_r)", data=df)
    plt.title(f"Figure 4: Proof of the Principle of Minimum Semantic Energy (r = {correlation:.4f})", fontsize=16)
    plt.xlabel("System Energy Reduction (Delta E)", fontsize=12)
    plt.ylabel("Propagation Strength (Sum P_r)", fontsize=12)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("figure_4_minimum_energy.png")
    print("Saved figure_4_minimum_energy.png")

def main():
    # --- Part 1: Initialization ---
    print("Initializing TFK Validation Suite...")
    model = load_model()
    if model is None: return
    contradictions = define_contradictions()

    # --- Part 2: Running Analyses ---
    print("\nRunning Critical Point Analysis (Exp. 10)...")
    critical_point_df = run_critical_point_analysis(model, contradictions)

    print("Running Semantic Cluster Analysis (Exp. 11)...")
    cluster_results_df = run_cluster_analysis(model, contradictions)

    print("Running Recursive Cascade Analysis (Exp. 6)...")
    cascade_df, cascade_summary = run_recursive_cascade_analysis(model, contradictions, trigger=("science", "religion"))

    print("Running Minimum Energy Principle Analysis (Exp. 13)...")
    energy_df, correlation = run_minimum_energy_analysis(model, contradictions, critical_point_df)

    # --- Part 3: Generating Output ---
    print("\n" + "="*80)
    print("    TFK VALIDATION SUITE: FINAL EMPIRICAL DATA OUTPUT")
    print("="*80)

    # Output 1
    print("\n--- Table 1: Critical Point Analysis ---")
    print(critical_point_df.to_string())
    create_critical_point_plot(critical_point_df)

    # Output 2
    print("\n--- Table 2: Semantic Cluster Analysis ---")
    print(cluster_results_df.to_string())
    create_cluster_plot(cluster_results_df)

    # Output 3
    print("\n--- Table 3: Recursive Cascade Analysis ---")
    print(cascade_df[['Secondary Contradiction', 'P_secondary', 'E_contribution']].to_string())
    print(f"\nSummary: {cascade_summary}")
    create_cascade_plot(cascade_summary)

    # Output 4
    print("\n--- Table 4: Minimum Energy Principle Analysis ---")
    print(f"Pearson Correlation Coefficient: {correlation:.4f}")
    create_energy_scatterplot(energy_df, correlation) # Plot first with original column name

    # Renaming for clarity in the final printed table
    energy_df.rename(columns={"Total Expected Value (Sum P_r)": "Sum P_r (Ripple)"}, inplace=True)
    print(energy_df.to_string())

    print("\nAll analyses complete. Tables printed above. Figures saved to disk.")

if __name__ == "__main__":
    main()
