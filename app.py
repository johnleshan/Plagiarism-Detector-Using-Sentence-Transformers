# Importing necessary modules
import os
import warnings

# Suppress TensorFlow deprecation warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress INFO and WARNING messages
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', module='tensorflow')

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from joblib import Parallel, delayed
import numpy as np

# Get list of student files ending with '.txt'
student_files = [doc for doc in os.listdir() if doc.endswith('.txt')]
student_notes = [open(_file, encoding='utf-8').read() for _file in student_files]

# Function to extract overlapping sentences between two texts
def extract_overlapping_sentences(text_a, text_b):
    sentences_a = text_a.split('. ')
    sentences_b = text_b.split('. ')
    overlapping = set(sentences_a).intersection(set(sentences_b))
    return '. '.join(overlapping)

# === TF-IDF Similarity ===
def vectorize_tfidf(Text): 
    return TfidfVectorizer().fit_transform(Text).toarray()

def similarity_tfidf(doc1, doc2): 
    return cosine_similarity([doc1, doc2])[0][1]

vectors_tfidf = vectorize_tfidf(student_notes)
s_vectors_tfidf = list(zip(student_files, vectors_tfidf))

# === Sentence-BERT Embeddings ===
model = SentenceTransformer('all-mpnet-base-v2')
embeddings_sbert = model.encode(student_notes)

# === Clustering with Sentence-BERT ===
# Function to find optimal number of clusters using Elbow Method and Silhouette Score
def find_optimal_clusters(data, max_k=10):
    n_samples = len(data)  # Number of documents
    if n_samples < 2:
        raise ValueError("Not enough samples to perform clustering.")
    
    # Ensure max_k does not exceed n_samples - 1
    max_k = min(max_k, n_samples - 1)
    
    iters = range(2, max_k + 1)  # Start from 2 clusters up to max_k
    sse = []
    silhouette_scores = []

    for k in iters:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(data)
        sse.append(kmeans.inertia_)
        
        # Compute Silhouette Score only if k < n_samples
        if 2 <= k < n_samples:
            score = silhouette_score(data, kmeans.labels_)
            silhouette_scores.append(score)

    # Use the "elbow" point in SSE or highest Silhouette Score to choose optimal k
    if silhouette_scores:
        optimal_k = np.argmax(silhouette_scores) + 2  # Adding 2 because range starts at 2
    else:
        optimal_k = min(max_k, n_samples - 1)  # Default to max_k if no Silhouette Scores are available

    return optimal_k

optimal_num_clusters = find_optimal_clusters(embeddings_sbert)

# Perform K-Means clustering on the embeddings
kmeans = KMeans(n_clusters=optimal_num_clusters, random_state=42)
clusters = kmeans.fit_predict(embeddings_sbert)

# Organize files into their respective clusters
clustered_files = {i: [] for i in range(optimal_num_clusters)}
for idx, cluster in enumerate(clusters):
    clustered_files[cluster].append((student_files[idx], embeddings_sbert[idx]))

# === Plagiarism Detection ===
plagiarism_results = set()
id_counter = 1

def calculate_similarity(file_a, embedding_a, file_b, embedding_b, method="sbert"):
    global id_counter
    if method == "sbert":
        sim_score = cosine_similarity([embedding_a], [embedding_b])[0][0]
    else:
        sim_score = similarity_tfidf(embedding_a, embedding_b)
    
    if sim_score >= 0.7:  # Only consider scores above a threshold
        text_a = next(note for fname, note in zip(student_files, student_notes) if fname == file_a)
        text_b = next(note for fname, note in zip(student_files, student_notes) if fname == file_b)
        
        overlapping_text = extract_overlapping_sentences(text_a, text_b)
        
        result = (id_counter, file_a, file_b, sim_score, overlapping_text)
        id_counter += 1
        return result
    return None

# Parallel processing to check plagiarism within each cluster
def process_cluster(cluster_data):
    results = []
    files, embeddings = zip(*cluster_data)
    for i in range(len(files)):
        for j in range(i + 1, len(files)):
            result = calculate_similarity(files[i], embeddings[i], files[j], embeddings[j])
            if result:
                results.append(result)
    return results

# Run plagiarism detection in parallel across clusters
results = Parallel(n_jobs=-1)(
    delayed(process_cluster)(clustered_files[cluster]) for cluster in clustered_files
)

# Flatten the results from all clusters
for cluster_result in results:
    for res in cluster_result:
        plagiarism_results.add(res)

# === TF-IDF Plagiarism Detection ===
for student_a, text_vector_a in s_vectors_tfidf:
    new_vectors = s_vectors_tfidf.copy()
    current_index = new_vectors.index((student_a, text_vector_a))
    del new_vectors[current_index]
    for student_b, text_vector_b in new_vectors:
        result = calculate_similarity(student_a, text_vector_a, student_b, text_vector_b, method="tfidf")
        if result:
            plagiarism_results.add(result)

# Print plagiarism results
print(f"{'ID':<5} {'Source Document A':<25} {'Source Document B':<25} {'Similarity Score':<15} {'Copied Text'}")
for data in sorted(plagiarism_results, key=lambda x: x[3], reverse=True):
    print(f"{data[0]:<5} {data[1]:<25} {data[2]:<25} {data[3]:<15.2f} {data[4][:50]}...")  # Truncate copied text to 50 chars