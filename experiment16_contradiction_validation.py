import numpy as np
import gensim.downloader as api
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

def main():
    print("Loading GloVe model (glove-wiki-gigaword-300)...")
    model = api.load("glove-wiki-gigaword-300")
    
    # Standard contradictions dataset
    contradictions = [
        ('man', 'woman'), ('old', 'new'), ('order', 'chaos'),
        ('good', 'evil'), ('life', 'death'), ('love', 'hate'),
        ('war', 'peace'), ('hot', 'cold'), ('light', 'dark'),
        ('happy', 'sad'), ('rich', 'poor'), ('strong', 'weak'),
        ('fast', 'slow'), ('open', 'closed'), ('simple', 'complex'),
        # Adding some additional pairs for comparison
        ('strength', 'weakness'), ('science', 'religion'),
        ('wealth', 'happiness')
    ]
    
    results = []
    
    print("\n--- Contradiction Dataset Validation Analysis ---")
    print("Calculating semantic relationships...\n")
    
    for word1, word2 in contradictions:
        try:
            # Get word vectors
            v1 = model[word1]
            v2 = model[word2]
            
            # Calculate metrics
            cos_sim = cosine_similarity(v1.reshape(1, -1), v2.reshape(1, -1))[0][0]
            euc_dist = np.linalg.norm(v1 - v2)
            
            # Classify relationship type based on cosine similarity
            if cos_sim < -0.1:
                rel_type = "Strong Contradiction"
            elif cos_sim < 0.1:
                rel_type = "Weak Contradiction"
            elif cos_sim < 0.3:
                rel_type = "Neutral/Unrelated"
            elif cos_sim < 0.6:
                rel_type = "Mildly Related"
            else:
                rel_type = "Strongly Related"
            
            results.append({
                'Concept Pair': f"({word1}, {word2})",
                'Cosine Similarity': cos_sim,
                'Euclidean Distance': euc_dist,
                'Relationship Type': rel_type
            })
            
        except KeyError as e:
            print(f"Warning: Words not found in vocabulary - {word1} and/or {word2}")
    
    # Create and display results table
    df = pd.DataFrame(results)
    df = df.sort_values('Cosine Similarity', ascending=True)  # Most contradictory first
    
    # Print formatted table
    print("--- Contradiction Validation Results ---")
    print(f"{'Concept Pair':<25} | {'Cosine Similarity':>18} | {'Euclidean Distance':>18} | {'Relationship Type'}")
    print("-" * 90)
    
    for _, row in df.iterrows():
        print(f"{row['Concept Pair']:<25} | {row['Cosine Similarity']:>18.4f} | {row['Euclidean Distance']:>18.4f} | {row['Relationship Type']}")
    
    # Summary statistics
    print("\n--- Summary of Relationship Types ---")
    print(df['Relationship Type'].value_counts())
    
    print("""
--- Key Findings ---
1. Strong Contradictions (cosine similarity < -0.1):
   - These pairs show clear opposition in semantic space
   - They are strong candidates for TFK analysis

2. Weak Contradictions (-0.1 < cosine similarity < 0.1):
   - These pairs show some opposition but may have more complex relationships
   - They might need additional context for meaningful analysis

3. Neutral/Related Pairs (cosine similarity > 0.1):
   - These pairs are not true contradictions
   - They may represent related concepts, synonyms, or co-occurring ideas
   - Consider removing or re-evaluating these for contradiction analysis
""")
    
    # Save results to CSV for further analysis
    df.to_csv('contradiction_validation_results.csv', index=False)
    print("Detailed results saved to 'contradiction_validation_results.csv'")

if __name__ == "__main__":
    main()
