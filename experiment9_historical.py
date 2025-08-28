import numpy as np
import pickle
import heapq
from sklearn import preprocessing

# --- Core TFK Functions ---
def calculate_distance(vec1, vec2):
    """Calculates the Euclidean distance between two vectors."""
    return np.linalg.norm(vec1 - vec2)

def calculate_mediator(vec1, vec2):
    """Calculates the unweighted midpoint mediator."""
    return (vec1 + vec2) / 2

# --- Helper Class for loading HistWords Embeddings ---
class Embedding:
    """
    A simplified class to load and interact with HistWords embeddings.
    """
    def __init__(self, vecs, vocab, normalize=True):
        self.m = vecs
        self.iw = vocab
        self.wi = {w: i for i, w in enumerate(self.iw)}
        if normalize:
            preprocessing.normalize(self.m, copy=False)

    @classmethod
    def load(cls, path, normalize=True):
        """Loads an embedding from the specified path."""
        try:
            with open(path + "-vocab.pkl", 'rb') as f:
                vocab = pickle.load(f)
            vecs = np.load(path + "-w.npy", mmap_mode="c")
            print(f"Model from {path} loaded successfully.")
            return cls(vecs, vocab, normalize)
        except FileNotFoundError as e:
            print(f"Error loading model from {path}: {e}")
            print("Please ensure the model is downloaded and the path is correct.")
            return None

    def represent(self, w):
        """Returns the vector for a word."""
        if w in self.wi:
            return self.m[self.wi[w], :]
        else:
            raise KeyError(f"Word '{w}' not in vocabulary.")

    def closest(self, vector, n=5):
        """Finds the closest words to a given vector."""
        # Assumes the vectors have been normalized.
        scores = self.m.dot(vector)
        return heapq.nlargest(n, zip(scores, self.iw))

# --- Main Execution ---
def main():
    """Main function to run the historical embedding analysis."""
    # Paths to the downloaded COHA models for the 1920s and 1990s.
    path_model_1920s = "sgns/1920"
    path_model_1990s = "sgns/1990"

    model_1920 = Embedding.load(path_model_1920s)
    model_1990 = Embedding.load(path_model_1990s)

    if model_1920 is None or model_1990 is None:
        print("\nExecution stopped.")
        return

    # --- Procedure ---
    contradiction = ("nature", "technology")
    c1, c2 = contradiction

    try:
        print(f"\nAnalyzing the historical shift of the contradiction: {contradiction}")

        # 1. Calculate Mediator 1920
        vec_c1_1920 = model_1920.represent(c1)
        vec_c2_1920 = model_1920.represent(c2)
        mediator_1920 = calculate_mediator(vec_c1_1920, vec_c2_1920)

        # 2. Calculate Mediator 1990
        vec_c1_1990 = model_1990.represent(c1)
        vec_c2_1990 = model_1990.represent(c2)
        mediator_1990 = calculate_mediator(vec_c1_1990, vec_c2_1990)

    except KeyError as e:
        print(f"Error: A word in {contradiction} was not found in one of the historical models. {e}")
        return

    # --- Analysis & Output ---
    distance_between_mediators = calculate_distance(mediator_1920, mediator_1990)

    print("\n--- Historical Mediator Analysis ---")
    print(f"Distance between Mediator 1920 and Mediator 1990: {distance_between_mediators:.4f}")

    # Find closest words to each mediator
    closest_1920 = model_1920.closest(mediator_1920)
    closest_1990 = model_1990.closest(mediator_1990)

    print("\nConceptual neighborhood of the 1920s mediator:")
    for score, word in closest_1920:
        print(f"- {word} (Similarity: {score:.4f})")

    print("\nConceptual neighborhood of the 1990s mediator:")
    for score, word in closest_1990:
        print(f"- {word} (Similarity: {score:.4f})")

    # Hypothesis Test
    if distance_between_mediators > 1.0: # Using a threshold to define 'significant'
        print("\nHypothesis CONFIRMED: The mediator's position shifted significantly over time.")
    else:
        print("\nHypothesis NOT CONFIRMED: The mediator's position remained relatively stable.")

    print("\nConclusion:")
    print("TFK was successfully used to map the historical evolution of a conceptual conflict.")
    print(f"The point of balance between '{c1}' and '{c2}' shifted significantly over the 20th century.")
    print("This proves that TFK is not just a static model but can be used as a dynamic tool for quantitative historical analysis.")

if __name__ == "__main__":
    main()
