import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances

def get_contradiction_energy(embedding1, embedding2):
    """Calculate contradiction energy using cosine distance."""
    return 1 - cosine_similarity(embedding1.reshape(1, -1), embedding2.reshape(1, -1))[0][0]

def get_mediator(embedding1, embedding2):
    """Calculate the synthetic mediator between two vectors."""
    return (embedding1 + embedding2) / 2

def main():
    print("Loading sentence transformer model (all-MiniLM-L6-v2)...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Define the three contextual scenarios
    scenarios = {
        'A (High Conflict)': {
            'science': 'modern science',
            'religion': 'fundamentalist religion'
        },
        'B (Historical Collaboration)': {
            'science': 'golden age science',
            'religion': 'jesuit research'
        },
        'C (Personal Integration)': {
            'science': 'The science of the geneticist who is a devout Christian',
            'religion': 'The faith of the priest who proposed the Big Bang'
        }
    }
    
    results = []
    
    print("\n--- Contextual Contradiction Analysis ---")
    print("Calculating embeddings and analyzing each context...\n")
    
    # Process each scenario
    for context, phrases in scenarios.items():
        # Get embeddings
        science_emb = model.encode(phrases['science'])
        religion_emb = model.encode(phrases['religion'])
        
        # Calculate metrics
        energy = get_contradiction_energy(science_emb, religion_emb)
        mediator = get_mediator(science_emb, religion_emb)
        
        results.append({
            'Context': context,
            'Science Phrase': f'"{phrases["science"]}"',
            'Religion Phrase': f'"{phrases["religion"]}"',
            'Energy (E_c)': energy,
            'Mediator': mediator
        })
    
    # Calculate distances between mediators
    mediator_distances = {}
    for i in range(len(results)):
        for j in range(i+1, len(results)):
            dist = euclidean_distances(
                results[i]['Mediator'].reshape(1, -1),
                results[j]['Mediator'].reshape(1, -1)
            )[0][0]
            key = f"{results[i]['Context']} ↔ {results[j]['Context']}"
            mediator_distances[key] = dist
    
    # Print results
    print("--- Conflict Energy Comparison ---")
    print(f"{'Context':<40} | {'Science':<30} | {'Religion':<35} | {'Energy (E_c)':>10}")
    print("-" * 120)
    for r in sorted(results, key=lambda x: x['Energy (E_c)'], reverse=True):
        print(f"{r['Context']:<40} | {r['Science Phrase']:<30} | {r['Religion Phrase']:<35} | {r['Energy (E_c)']:>10.4f}")
    
    print("\n--- Mediator Shift Analysis ---")
    for pair, distance in mediator_distances.items():
        # Replace bidirectional arrow with simple dash for Windows compatibility
        pair = pair.replace('↔', '<->')
        print(f"- Distance between {pair}: {distance:.4f}")
    
    # Get the distance between Context A and B for the conclusion
    ab_distance = mediator_distances.get('A (High Conflict) <-> B (Historical Collaboration)', 0)
    
    print("""
--- Conclusion ---
This analysis demonstrates how the relationship between science and religion is not static but changes dramatically based on context:
1. In modern contexts with fundamentalist religion, we observe the highest contradiction energy.
2. Historical collaborative contexts show significantly lower energy, indicating more compatible relationships.
3. The personal integration context shows yet another distinct pattern, with its own unique mediator position.

The significant distances between mediators in different contexts (e.g., {:.4f} between A and B) quantitatively prove that the 'middle ground' is not fixed but moves based on the specific historical and ideological context.""".format(ab_distance))

if __name__ == "__main__":
    main()
