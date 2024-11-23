# from flask import Flask, request, jsonify
# import pandas as pd
# from sentence_transformers import SentenceTransformer, util
# import torch
# from fuzzywuzzy import process

# app = Flask(__name__)

# data_path = "filtered_dataset.csv" 
# data = pd.read_csv(data_path)

# # Load the SentenceTransformer model
# model = SentenceTransformer('all-MiniLM-L6-v2')
# # Precompute embeddings for product descriptions
# # query="which ciy is the most populated?"
# # xq=model.encode(query)
# # print(xq)
# product_name = data['Name'].fillna('').tolist()
# product_embeddings = model.encode(product_name, convert_to_tensor=True)

# print(product_embeddings)
# # # # Optional: Save embeddings for future use
# # torch.save(product_embeddings, "models/embeddings.pt")

# # # Translation map for handling Romanized Hindi words
# translation_map = {"chawal": "rice", "jeera": "cumin", "atta": "flour", "dal": "lentils",
#             'dalchini': 'cinnamon',
#             'doodh': 'milk',
#             'dahi': 'yogurt',
#             'paneer': 'cottage cheese',
#             'makhan': 'butter',
#             'aloo': 'potato',
#             'pyaaz': 'onion',
#             'tamatar': 'tomato',
#             'gobhi': 'cauliflower',
#             'seb': 'apple',
#             'kela': 'banana',
#             'aam': 'mango'
# }

# # # Semantic Search Endpoint
# @app.route('/semantic_search', methods=['POST'])
# def semantic_search():
#     query = request.json.get('query', '')
#     top_k = request.json.get('top_k', 10)

#     # Encode the query
#     query_embedding = model.encode(query, convert_to_tensor=True)

#     # Compute cosine similarity between query and product embeddings
#     scores = util.cos_sim(query_embedding, product_embeddings).squeeze()
#     top_results = torch.topk(scores, k=top_k)

#     # Fetch top results
#     results = []
#     for idx in top_results.indices:
#         product = data.iloc[idx.item()]
#         results.append({
#             "Name": product['Name'],
#             "Category": product['Category'],
#             "Price": product['Price'],
#             "Description": product['Description']
#         })

#     return jsonify(results)

# # Fuzzy Search Endpoint
# @app.route('/fuzzy_search', methods=['POST'])
# def fuzzy_search():
#     query = request.json.get('query', '')
#     top_k = request.json.get('top_k', 10)

#     # Match query with product names
#     product_names = data['Name'].fillna('').tolist()
#     fuzzy_results = process.extract(query, product_names, limit=top_k)

#     # Fetch results
#     results = []
#     for name, score in fuzzy_results:
#         product = data[data['Name'] == name].iloc[0]
#         results.append({
#             "Name": product['Name'],
#             "Category": product['Category'],
#             "Price": product['Price'],
#             "Description": product['Description']
#         })

#     return jsonify(results)

# # Translation Endpoint
# @app.route('/translate', methods=['POST'])
# def translate_query():
#     query = request.json.get('query', '')
#     words = query.split()
#     translated = [translation_map.get(word, word) for word in words]
#     return jsonify({"translated_query": " ".join(translated)})

# # Retailer-Based Filtering Endpoint
# @app.route('/filter_results', methods=['POST'])
# def filter_results():
#     results = request.json.get('results', [])
#     retailer_type = request.json.get('retailer_type', 'small-scale')

#     # Filter results based on retailer type and price
#     filtered_results = []
#     for product in results:
#         if retailer_type == 'small-scale' and product['Price'] <= 100:
#             filtered_results.append(product)
#         elif retailer_type == 'large-scale' and product['Price'] > 100:
#             filtered_results.append(product)

#     return jsonify(filtered_results)

# # Home Route (optional for testing)
# @app.route('/', methods=['GET'])
# def home():
#     return "Semantic Search API is running!"

# # Run Flask App
# if __name__ == '__main__':
#     app.run(debug=True, host='0.0.0.0', port=5000)


import pandas as pd
from sentence_transformers import SentenceTransformer, util
import torch

# 1. Initialize SentenceTransformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# 2. Function to clean input (basic text preprocessing)
def clean_input(input_text):
    input_text = input_text.strip() # Convert to lowercase and strip extra spaces
    # You can add more advanced cleaning here if needed (e.g., removing special characters)
    return input_text

# 3. Function to perform semantic search
def semantic_search(input_query, dataset_path, top_k=10):
    # Clean the input query
    cleaned_query = clean_input(input_query)
    
    # Generate embedding for the cleaned input query
    query_embedding = model.encode(cleaned_query, convert_to_tensor=True)
    
    # Load the pre-cleaned dataset
    data = pd.read_csv('filtered_dataset.csv')
    
    # Ensure 'Description' column exists in dataset, else adjust
    if 'Name' not in data.columns:
        raise ValueError("Dataset must contain a 'Name' column.")
    
    # 4. Generate embeddings for the dataset descriptions
    product_name = data['Name'].fillna('').tolist()  # Handle NaNs
    product_embeddings = model.encode(product_name, convert_to_tensor=True)
    
    # 5. Compute cosine similarity between the query and product descriptions
    scores = util.cos_sim(query_embedding, product_embeddings).squeeze()
    
    # 6. Get the top-k most similar products
    top_results = torch.topk(scores, k=top_k)
    
    # 7. Display results
    print(f"Top {top_k} results for '{input_query}':\n")
    for idx in top_results.indices:
        product = data.iloc[idx.item()]
        print(f"Name: {product['Name']}")
        print(f"Category: {product['Category']}")
        print(f"Description: {product['Description']}")
        print(f"Price: {product['Price']}")
        print('-' * 50)

# Example usage
if __name__ == '__main__':
    # Ask for user input
    input_query = input("Enter your search query: ")
    dataset_path = "filtered_dataset.csv"  # Path to your dataset (replace with your actual dataset)
    
    # Perform semantic search and display results
    semantic_search(input_query, dataset_path)
