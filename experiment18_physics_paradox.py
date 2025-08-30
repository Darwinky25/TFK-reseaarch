import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from prettytable import PrettyTable
import torch

def load_model():
    print("Loading sentence transformer model (all-mpnet-base-v2)...")
    return SentenceTransformer('all-mpnet-base-v2')

def get_pole_vector(model, phrases):
    """Calculate the average vector for a list of phrases."""
    embeddings = model.encode(phrases, convert_to_tensor=True)
    # Ensure we're working with CPU tensors for compatibility
    if torch.cuda.is_available():
        embeddings = embeddings.cpu()
    return torch.mean(embeddings, dim=0).numpy()

def calculate_mediator(vec1, vec2):
    """Calculate the mediator vector between two vectors."""
    return (vec1 + vec2) / 2

def find_closest_concepts(model, target_vector, concept_list, top_n=10):
    """Find the closest concepts to the target vector from a list."""
    # Get embeddings and ensure they're on CPU
    concept_embeddings = model.encode(concept_list, convert_to_tensor=True)
    if torch.cuda.is_available():
        concept_embeddings = concept_embeddings.cpu()
    
    # Convert target vector to tensor and ensure it's on CPU
    target_tensor = torch.tensor(target_vector, dtype=torch.float32).unsqueeze(0)
    
    # Calculate cosine similarities using PyTorch for device compatibility
    concept_embeddings = concept_embeddings.to(torch.float32)
    target_tensor = target_tensor.to(torch.float32)
    
    # Normalize vectors for cosine similarity
    concept_norms = torch.norm(concept_embeddings, p=2, dim=1, keepdim=True)
    target_norm = torch.norm(target_tensor, p=2, dim=1, keepdim=True)
    
    # Handle zero norms to avoid division by zero
    concept_norms[concept_norms == 0] = 1e-10
    target_norm[target_norm == 0] = 1e-10
    
    # Calculate cosine similarity
    normalized_concepts = concept_embeddings / concept_norms
    normalized_target = target_tensor / target_norm
    similarities = torch.mm(normalized_concepts, normalized_target.T).squeeze()
    
    # Get top N most similar concepts
    top_indices = torch.topk(similarities, min(top_n, len(concept_list))).indices
    return [(concept_list[i], similarities[i].item()) for i in top_indices]

def main():
    # Initialize model
    model = load_model()
    
    # Define the three axes of contradiction
    axes = {
        'Axis 1 (Nature of Spacetime)': {
            'GR': ["smooth spacetime", "dynamic geometry", "continuous field"],
            'QM': ["quantized space", "fixed background", "discrete packets"]
        },
        'Axis 2 (Nature of Reality)': {
            'GR': ["local realism", "objective reality", "deterministic evolution"],
            'QM': ["quantum uncertainty", "probabilistic wave-function", "entangled non-locality"]
        },
        'Axis 3 (Role of Observer)': {
            'GR': ["observer independent", "absolute framework"],
            'QM': ["measurement problem", "observer effect"]
        }
    }
    
    # Define physics concepts to search
    physics_concepts = [
        'information', 'entropy', 'holography', 'computation', 'geometry',
        'symmetry', 'string', 'loop', 'causality', 'logic', 'consciousness',
        'energy', 'field', 'spacetime', 'quantum gravity', 'entanglement',
        'emergent spacetime', 'quantum information', 'holographic principle',
        'non-locality', 'wave function', 'decoherence', 'complementarity',
        'duality', 'supersymmetry', 'quantum field theory', 'black hole',
        'wormhole', 'multiverse', 'holographic universe', 'quantum foam'
    ]
    
    print("\n--- Physics Paradox Analysis: Searching for the Theory of Everything ---")
    print("Calculating pole vectors and axis mediators...\n")
    
    axis_mediators = {}
    
    # Calculate mediators for each axis
    for axis_name, poles in axes.items():
        print(f"Processing {axis_name}...")
        
        # Get pole vectors
        gr_vector = get_pole_vector(model, poles['GR'])
        qm_vector = get_pole_vector(model, poles['QM'])
        
        # Calculate axis mediator
        mediator = calculate_mediator(gr_vector, qm_vector)
        axis_mediators[axis_name] = mediator
        
        print(f"- {axis_name} mediator calculated")
    
    # Calculate total centroid
    print("\nCalculating total conflict centroid...")
    total_centroid = np.mean(list(axis_mediators.values()), axis=0)
    
    # Find closest concepts
    print("Searching for conceptual neighbors...\n")
    closest_concepts = find_closest_concepts(model, total_centroid, physics_concepts)
    
    # Display results
    table = PrettyTable()
    table.field_names = ["Rank", "Concept", "Similarity"]
    table.align["Concept"] = "l"
    table.align["Similarity"] = "r"
    
    for i, (concept, similarity) in enumerate(closest_concepts, 1):
        table.add_row([i, concept, f"{similarity:.4f}"])
    
    print("--- Top Concepts Closest to the Quantum-Relativity Centroid ---")
    print(table)
    
    print("""
--- Analysis ---
This analysis maps the conceptual space between General Relativity and Quantum Mechanics.
The centroid represents a theoretical 'balance point' between these frameworks.

Key observations:
1. Higher similarity scores indicate concepts that may bridge the GR-QM divide.
2. The top concepts suggest potential directions for a Theory of Everything.
3. This is a topological analysis, not a physical theory - it reveals conceptual
   relationships in the semantic space of physics language.""")

if __name__ == "__main__":
    main()
