from client import TFIDFVectorizerEngine

def main():
    engine = TFIDFVectorizerEngine()
    docs = [
        "AI agents execute autonomous workflows",
        "Agents optimize distributed systems",
        "Quantum mechanics and entanglement"
    ]
    vectors = engine.fit_transform(docs)
    sim12 = engine.cosine_similarity(vectors[0], vectors[1])
    sim13 = engine.cosine_similarity(vectors[0], vectors[2])
    print("TF-IDF & Cosine Similarity Verification:")
    print(f"Similarity (Doc0 vs Doc1): {sim12}")
    print(f"Similarity (Doc0 vs Doc2): {sim13}")

if __name__ == "__main__":
    main()
