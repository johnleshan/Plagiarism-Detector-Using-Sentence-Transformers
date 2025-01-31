# Importing the necessary modules
import os
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer  # For semantic embeddings
from joblib import Parallel, delayed  # For parallel processing
import numpy as np

# List all text files in the current directory
student_files = [doc for doc in os.listdir() if doc.endswith('.txt')]

# Read the content of each text file
student_notes = [open(_file, encoding='utf-8').read() for _file in student_files]

# Load pre-trained Sentence-BERT model for semantic embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')  # Lightweight and efficient model

# Generate semantic embeddings for all documents
embeddings = model.encode(student_notes)

# Set to store plagiarism results
plagiarism_results = set()

# Threshold for considering plagiarism (e.g., 0.7 means 70% similarity)
PLAGIARISM_THRESHOLD = 0.7

# Number of clusters for K-Means (adjust based on dataset size)
NUM_CLUSTERS = 5

# Perform K-Means clustering on the embeddings
kmeans = KMeans(n_clusters=NUM_CLUSTERS, random_state=42)
clusters = kmeans.fit_predict(embeddings)

# Function to compute cosine similarity between two documents
def similarity(doc1, doc2):
    return cosine_similarity([doc1], [doc2])[0][0]

# Function to compare documents within the same cluster
def compare_documents_in_cluster(cluster_id, student_files, embeddings, clusters):
    results = []
    # Get indices of documents in the current cluster
    cluster_indices = np.where(clusters == cluster_id)[0]
    # Compare each document in the cluster with every other document in the same cluster
    for i in cluster_indices:
        for j in cluster_indices:
            if i != j:  # Avoid self-comparison
                sim_score = similarity(embeddings[i], embeddings[j])
                if sim_score > PLAGIARISM_THRESHOLD:  # Only consider scores above the threshold
                    student_pair = sorted((student_files[i], student_files[j]))
                    results.append((student_pair[0], student_pair[1], sim_score))
    return results

# Function to check for plagiarism among student files
def check_plagiarism():
    global embeddings, student_files, clusters
    # Use parallel processing to compare documents within each cluster
    results = Parallel(n_jobs=-1)(delayed(compare_documents_in_cluster)(cluster_id, student_files, embeddings, clusters)
                                  for cluster_id in range(NUM_CLUSTERS))
    # Flatten the results and add them to the set
    for sublist in results:
        for item in sublist:
            plagiarism_results.add(item)
    return plagiarism_results

# Print the plagiarism results
print("Plagiarism Results (Student A, Student B, Similarity Score):")
for data in check_plagiarism():
    print(data)