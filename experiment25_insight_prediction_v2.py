import numpy as np
import gensim.downloader as api
from tqdm import tqdm
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import spacy
import random

class ImprovedInsightPredictor:
    def __init__(self, n_contradictions=200, vocab_size=1000):
        self.model = None
        self.nlp = None
        self.n_contradictions = n_contradictions
        self.vocab_size = vocab_size
        self.contradictions = []
        self.vector_cache = {}
        
    def load_models(self):
        print("Loading models...")
        self.model = api.load('glove-wiki-gigaword-300')
        self.nlp = spacy.load("en_core_web_sm")
        
    def prepare_vocabulary(self):
        print("Preparing vocabulary...")
        self.vocab = [w for w in self.model.index_to_key[:self.vocab_size] 
                     if ' ' not in w and w.isalpha()]
        
        # Cache normalized vectors
        for word in self.vocab:
            vec = self.model[word]
            self.vector_cache[word] = vec / np.linalg.norm(vec)
            
    def generate_contradictions_cluster(self):
        print("Generating contradictions via clustering...")
        vectors = np.array([self.vector_cache[w] for w in self.vocab])
        kmeans = KMeans(n_clusters=2, n_init=10)
        labels = kmeans.fit_predict(vectors)
        
        cluster0 = [w for w, l in zip(self.vocab, labels) if l == 0]
        cluster1 = [w for w, l in zip(self.vocab, labels) if l == 1]
        
        contradictions = []
        for i in range(min(self.n_contradictions, len(cluster0), len(cluster1))):
            w1, w2 = cluster0[i], cluster1[i]
            sim = np.dot(self.vector_cache[w1], self.vector_cache[w2])
            if -0.3 < sim < 0.7:  # Filter by semantic similarity
                contradictions.append((w1, w2))
                
        return contradictions[:self.n_contradictions]
    
    def calculate_metrics(self):
        print("\nCalculating metrics...")
        results = []
        
        for w1, w2 in tqdm(self.contradictions, desc="Processing"):
            try:
                vec = (self.vector_cache[w1] + self.vector_cache[w2]) / 2
                metrics = {'contradiction': (w1, w2), 'sum_pr': 0, 'sum_delta_e': 0}
                
                for w3, w4 in self.contradictions[:50]:  # Limit comparisons
                    if (w1, w2) == (w3, w4):
                        continue
                        
                    target_vec = (self.vector_cache[w3] + self.vector_cache[w4]) / 2
                    dist = np.linalg.norm(vec - target_vec)
                    metrics['sum_pr'] += np.exp(-0.1 * dist)
                    
                    # Simple energy reduction
                    d_ab = np.linalg.norm(vec - target_vec)
                    metrics['sum_delta_e'] += d_ab  # Simplified for demo
                    
                results.append(metrics)
            except KeyError:
                continue
                
        return pd.DataFrame(results)
    
    def run(self):
        print("\n--- Improved Large-Scale Analysis ---")
        self.load_models()
        self.prepare_vocabulary()
        self.contradictions = self.generate_contradictions_cluster()
        
        df = self.calculate_metrics()
        if df is None or df.empty:
            print("Error: No results.")
            return
            
        # Normalize and calculate SPS
        df[['norm_pr', 'norm_delta_e']] = MinMaxScaler().fit_transform(
            df[['sum_pr', 'sum_delta_e']])
        df['sps'] = df[['norm_pr', 'norm_delta_e']].mean(axis=1)
        df = df.sort_values('sps', ascending=False)
        
        # Save and show results
        df.to_csv('improved_rankings.csv', index=False)
        print("\nTop 10 Contradictions:")
        print(df[['contradiction', 'sps']].head(10).to_string(index=False))
        
        # Simple plot
        plt.figure(figsize=(10, 6))
        plt.scatter(df['sum_pr'], df['sum_delta_e'], alpha=0.6)
        plt.xlabel('Propagation Strength')
        plt.ylabel('Energy Reduction')
        plt.title('Improved Analysis Results')
        plt.savefig('improved_analysis.png')
        print("\nAnalysis complete. Results saved.")

if __name__ == "__main__":
    predictor = ImprovedInsightPredictor(n_contradictions=100)  # Reduced for speed
    predictor.run()
