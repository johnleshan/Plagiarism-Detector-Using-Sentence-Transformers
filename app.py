# Importing the necessary modules
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# List all text files in the current directory
student_files = [doc for doc in os.listdir() if doc.endswith('.txt')]

# Read the content of each text file
student_notes = [open(_file, encoding='utf-8').read() for _file in student_files]

# Function to convert text to TF-IDF vectors
def vectorize(Text):
    return TfidfVectorizer().fit_transform(Text).toarray()

# Function to compute cosine similarity between two documents
def similarity(doc1, doc2):
    return cosine_similarity([doc1, doc2])

# Convert the student notes to TF-IDF vectors
vectors = vectorize(student_notes)

# Pair each student file with its corresponding vector
s_vectors = list(zip(student_files, vectors))

# Set to store plagiarism results
plagiarism_results = set()

# Function to check for plagiarism among student files
def check_plagiarism():
    global s_vectors
    for student_a, text_vector_a in s_vectors:
        # Create a copy of the vectors list and remove the current student's vector
        new_vectors = s_vectors.copy()
        current_index = new_vectors.index((student_a, text_vector_a))
        del new_vectors[current_index]
        
        # Compare the current student's vector with all other students' vectors
        for student_b, text_vector_b in new_vectors:
            # Compute the similarity score
            sim_score = similarity(text_vector_a, text_vector_b)[0][1]
            # Sort the student pair to avoid duplicate entries
            student_pair = sorted((student_a, student_b))
            # Store the result as a tuple
            score = (student_pair[0], student_pair[1], sim_score)
            # Add the result to the set
            plagiarism_results.add(score)
    return plagiarism_results

# Print the plagiarism results
for data in check_plagiarism():
    print(data)