import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import faiss
from fuzzywuzzy import process

# 1. Load and preprocess the product data
def load_data(file_path):
    """ Load and preprocess product data from CSV file """
    data = pd.read_csv(file_path)
    data['Name'] = data['Name'].fillna("").str.lower()  # Lowercase and handle missing names
    data['Description'] = data['Description'].fillna("").str.lower()
    data['Category'] = data['Category'].fillna("").str.lower()
    data['Subcategory'] = data['Subcategory'].fillna("").str.lower()
    data['Price'] = data['Price'].fillna(0)  # Fill missing prices with 0 or handle as needed
    return data

# Example: Load your grocery data
data = load_data('df_midhackuse.csv')

# 2. Vectorize Product Data using TF-IDF
def vectorize_data(data):
    """ Vectorize product names, descriptions, and categories using TF-IDF """
    vectorizer = TfidfVectorizer(stop_words='english')
    text_data = data['Name'] + " " + data['Category'] + " " + data['Subcategory'] + " " + data['Description']
    vectors = vectorizer.fit_transform(text_data)
    return vectors, vectorizer

# Get TF-IDF vectors for the dataset
vectors, vectorizer = vectorize_data(data)

# 3. Dimensionality Reduction using TruncatedSVD
def reduce_dimensions(vectors, n_components=128):
    """ Reduce dimensionality of sparse vectors using TruncatedSVD """
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    reduced_vectors = svd.fit_transform(vectors)
    return reduced_vectors, svd

# Perform dimensionality reduction and save SVD model
reduced_vectors, svd = reduce_dimensions(vectors)

# 4. Build FAISS Index for Semantic Search
def build_faiss_index(reduced_vectors):
    """ Build a FAISS index for reduced dimensional vectors """
    reduced_vectors = np.array(reduced_vectors, dtype=np.float32)  # Ensure float32
    index = faiss.IndexFlatL2(reduced_vectors.shape[1])  # L2 similarity
    index.add(reduced_vectors)  # Add reduced vectors to index
    return index

# Build the FAISS index
index = build_faiss_index(reduced_vectors)

# 5. Semantic Search for Relevant Products
def search_products(query, vectorizer, index, svd, top_k=10):
    """ Perform a semantic search for the most relevant products """
    query_vector = vectorizer.transform([query])  # Transform query to sparse vector
    query_reduced = svd.transform(query_vector)  # Reduce dimensionality using the same SVD model
    query_reduced = np.array(query_reduced, dtype=np.float32)  # Ensure float32 type
    distances, indices = index.search(query_reduced, top_k)  # Search for top_k matches
    return indices, distances

# 6. Fuzzy Matching for Misspelled Queries
def fuzzy_search(query, data):
    """ Perform fuzzy search to correct spelling mistakes in product names """
    product_names = data['Name'].tolist()
    best_match = process.extractOne(query, product_names)
    return best_match

# 7. Handle Semantic Queries for Sizes, Categories, or Similar Products
# 7. Handle Semantic Queries for Sizes, Categories, or Similar Products
def handle_semantic_queries(query, data):
    """
    Handle semantic queries like "small pack sizes" or "beverages".
    Returns a filtered dataset based on size or Category keywords.
    """
    query = query.lower()

    # Check for specific queries like "rice"
    if "rice" in query:
        filtered_data = data[data['Description'].str.contains("rice", case=False)]
        return filtered_data
    
    # Check for size-related queries
    if "small pack" in query or "small size" in query:
        filtered_data = data[data['Description'].str.contains("100ml|200ml|small")]
        return filtered_data
    
    # Check for Category-related queries
    elif "beverages" in query:
        filtered_data = data[data['Category'].str.contains("juice|milk|beverage|drink")]
        return filtered_data
    
    # Default: return entire dataset
    return data

# 8. Combined Search: Fuzzy + Semantic + Similar Products
def combined_search(query, vectorizer, index, svd, reduced_vectors, data, top_k=10, fuzzy_threshold=80):
    """
    Perform combined fuzzy and semantic search with support for:
    1. Misspelled queries.
    2. Semantic understanding of sizes, categories, or similar products.
    """
    # Step 1: Perform fuzzy search for Name corrections
    fuzzy_match = fuzzy_search(query, data)
    
    if fuzzy_match and fuzzy_match[1] >= fuzzy_threshold:
        print(f"Fuzzy Match Found: {fuzzy_match[0]} with score: {fuzzy_match[1]}")
        indices, distances = search_products(fuzzy_match[0], vectorizer, index, svd, top_k)
    else:
        print("No strong fuzzy match found. Performing direct semantic search...")
        indices, distances = search_products(query, vectorizer, index, svd, top_k)
    
    # Step 2: Filter results based on semantic intent
    result_indices = indices[0]
    filtered_data = handle_semantic_queries(query, data.iloc[result_indices])
    
    return filtered_data.head(top_k)

# 9. Display Top Results
def display_top_results(query, vectorizer, index, svd, data, top_k=10):
    """
    Display the top results for the user's query.
    """
    results = combined_search(query, vectorizer, index, svd, reduced_vectors, data, top_k)
    print("\nTop Matching Results:")
    print(results[['Name', 'Category', 'Description', 'Price']])

# 10. Main function for user interaction
def main():
    query = input("Enter your product query: ")
    display_top_results(query, vectorizer, index, svd, data, top_k=10)

if __name__ == "__main__":
    main()
