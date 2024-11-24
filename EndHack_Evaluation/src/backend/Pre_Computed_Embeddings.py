# import pandas as pd
# from sentence_transformers import SentenceTransformer
# import torch

# # Load your dataset
# data_path = "filtered_dataset.csv"  # Replace with your dataset path
# data = pd.read_csv(data_path)

# # Initialize the SentenceTransformer model
# model = SentenceTransformer('all-MiniLM-L6-v2')

# # Precompute embeddings for Name, Category, and Description columns
# product_name = data['Name'].fillna('').tolist()
# product_category = data['Category'].fillna('').tolist()
# product_description = data['Description'].fillna('').tolist()

# # Generate embeddings for each field and concatenate them
# print("Generating embeddings...")
# name_embeddings = model.encode(product_name, convert_to_tensor=True)
# category_embeddings = model.encode(product_category, convert_to_tensor=True)
# description_embeddings = model.encode(product_description, convert_to_tensor=True)

# # Combine embeddings by summing them (or you can use concatenation)
# product_embeddings = name_embeddings + category_embeddings + description_embeddings

# # Save embeddings and the dataset to files
# torch.save(product_embeddings, "precomputed_embeddings.pt")
# data.to_csv("cleaned_dataset.csv", index=False)  # Save the cleaned dataset (optional)

# print("Embeddings saved to 'precomputed_embeddings.pt' and cleaned dataset saved to 'cleaned_dataset.csv'.")


import pandas as pd
from sentence_transformers import SentenceTransformer
import torch
from torch.utils.data import DataLoader

data = pd.read_csv("df_midhackuse.csv")  # Replace with your dataset path

# Ensure 'Name', 'Category', and 'Description' columns exist
if not {'Name', 'Category', 'Description'}.issubset(data.columns):
    raise ValueError("Dataset must contain 'Name', 'Category', and 'Description' columns.")

# Load the model
model = SentenceTransformer('paraphrase-MiniLM-L3-v2')


# Function to process embeddings in batches
def generate_embeddings_in_batches(sentences, model, batch_size=64):
    embeddings = []
    dataloader = DataLoader(sentences, batch_size=batch_size, shuffle=False)
    for batch in dataloader:
        batch_embeddings = model.encode(batch, convert_to_tensor=True)
        embeddings.append(batch_embeddings)
    return torch.cat(embeddings, dim=0)

# Combine fields into a single search corpus
data['SearchText'] = data[['Name', 'Category', 'Description']].fillna('').agg(' '.join, axis=1)

# Generate embeddings
print("Generating embeddings...")
search_texts = data['SearchText'].tolist()
product_embeddings = generate_embeddings_in_batches(search_texts, model)

# Save embeddings and dataset
print("Saving embeddings...")
torch.save(product_embeddings, "models/embeddings.pt")
data.to_csv("models/filtered_dataset_with_searchtext.csv", index=False)

print("Embeddings saved successfully!")
