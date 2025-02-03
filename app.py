# Importing the necessary modules
import os
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer  # For semantic embeddings
from joblib import Parallel, delayed  # For parallel processing
from collections import defaultdict  # For efficient grouping of documents

# List all text files in the current directory
student_files = [doc for doc in os.listdir() if doc.endswith('.txt')]

# Read the content of each text file
student_notes = [open(_file, encoding='utf-8').read() for _file in student_files]

# Load pre-trained Sentence-BERT model for semantic embeddings (larger model for better accuracy)
model = SentenceTransformer('all-mpnet-base-v2')

# Generate semantic embeddings for all documents
embeddings = model.encode(student_notes)

# Set to store plagiarism results
plagiarism_results = []

# Threshold for considering plagiarism (e.g., 0.7 means 70% similarity)
PLAGIARISM_THRESHOLD = 0.7

# Function to compute cosine similarity between two documents
def similarity(doc1, doc2):
    return cosine_similarity([doc1], [doc2])[0][0]

# Function to determine the optimal number of clusters using the Elbow Method
def find_optimal_clusters(embeddings, max_clusters=None):
    n_samples = len(embeddings)
    if max_clusters is None:
        max_clusters = min(10, n_samples - 1)  # Ensure max_clusters <= n_samples - 1
    wcss = []  # Within-Cluster-Sum-of-Squares
    silhouette_scores = []
    cluster_range = range(2, max_clusters + 1)  # Start from 2 clusters
    for n_clusters in cluster_range:
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        kmeans.fit(embeddings)
        wcss.append(kmeans.inertia_)
        if n_clusters > 1:  # Silhouette Score requires at least 2 clusters
            silhouette_scores.append(silhouette_score(embeddings, kmeans.labels_))
    # Find the optimal number of clusters using the Silhouette Score
    if silhouette_scores:
        optimal_clusters = cluster_range[np.argmax(silhouette_scores)]
        print(f"Optimal number of clusters: {optimal_clusters}")
    else:
        optimal_clusters = 2  # Default to 2 clusters if Silhouette Score cannot be computed
        print("Could not compute Silhouette Score. Defaulting to 2 clusters.")
    return optimal_clusters

# Perform clustering with the optimal number of clusters
optimal_clusters = find_optimal_clusters(embeddings)
kmeans = KMeans(n_clusters=optimal_clusters, random_state=42)
clusters = kmeans.fit_predict(embeddings)

# Group documents by their cluster for efficient comparison
cluster_groups = defaultdict(list)
for idx, cluster_id in enumerate(clusters):
    cluster_groups[cluster_id].append((student_files[idx], embeddings[idx], student_notes[idx]))

# Function to compare documents within the same cluster
def compare_documents_in_cluster(cluster_docs):
    results = []
    # Compare each document in the cluster with every other document in the same cluster
    for i in range(len(cluster_docs)):
        for j in range(i + 1, len(cluster_docs)):
            student_a, embedding_a, note_a = cluster_docs[i]
            student_b, embedding_b, note_b = cluster_docs[j]
            sim_score = similarity(embedding_a, embedding_b)
            if sim_score > PLAGIARISM_THRESHOLD:  # Only consider scores above the threshold
                results.append({
                    "id": f"{student_a.split('.')[0]}-{student_b.split('.')[0]}",
                    "source_documents": sorted([student_a, student_b]),
                    "similarity_score": sim_score,
                    "copied_texts": (note_a, note_b)
                })
    return results

# Function to check for plagiarism among student files
def check_plagiarism():
    global cluster_groups
    # Use parallel processing to compare documents within each cluster
    results = Parallel(n_jobs=-1)(delayed(compare_documents_in_cluster)(cluster_docs)
                                  for cluster_docs in cluster_groups.values())
    # Flatten the results and add them to the list
    for sublist in results:
        plagiarism_results.extend(sublist)
    return plagiarism_results

# Print the plagiarism results
print("Plagiarism Results:")
if not plagiarism_results:
    print("No plagiarism detected.")
else:
    for idx, result in enumerate(check_plagiarism(), start=1):
        print(f"ID: {idx}")
        print(f"Source Documents: {', '.join(result['source_documents'])}")
        print(f"Similarity Score: {result['similarity_score']:.2f}")
        print(f"Copied Texts:\nDocument A: {result['copied_texts'][0][:100]}...\nDocument B: {result['copied_texts'][1][:100]}...")
        print("-" * 80)