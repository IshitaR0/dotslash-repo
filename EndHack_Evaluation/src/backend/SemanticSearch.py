"""
# Document Outline

## Imports and Setup
- Import necessary libraries for data processing, NLP, and machine learning.
- Ensure required NLTK resources are downloaded.

## EnhancedRetailSearch Class
### Initialization
- Store and preprocess the input DataFrame.
- Define mappings for size patterns and Hindi-to-English translations.
- Initialize TF-IDF vectorizer and Word2Vec model.

### Preprocessing Methods
- `preprocess_data`: Merge and standardize relevant data columns.
- `extract_size`: Extract size information from product descriptions.

### Query Processing Methods
- `translate_query`: Translate Hindi words in the query to English.
- `expand_query`: Expand queries with semantic variations using Word2Vec.

### Model Training
- `train_word2vec`: Train a Word2Vec model using the dataset.

### Search Methods
- `get_fuzzy_matches`: Perform fuzzy matching to find approximate matches.
- `calculate_semantic_similarity`: Compute semantic similarity using TF-IDF.
- `search`: Combine fuzzy and semantic search to return ranked results.

### Metrics and Evaluation
- `get_metrics`: Return performance metrics for the search engine.

## Demo and Testing
### `demo_search` Function
- Load sample data.
- Initialize and test the search engine with various queries.
- Print results and performance metrics.

## Main Execution
- Run the `demo_search` function when the script is executed as the main program.
"""

import pandas as pd
import numpy as np
import re
from fuzzywuzzy import fuzz, process
from nltk.tokenize import word_tokenize
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import TfidfVectorizer
import time
import nltk
from typing import List, Dict

# Ensure necessary NLTK resources are downloaded
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

class EnhancedRetailSearch:
    def __init__(self, df: pd.DataFrame):
        # Initialize the class with the input DataFrame
        self.df = df.copy()  # Create a copy of the DataFrame
        self.df.reset_index(inplace=True, drop=True)  # Reset the DataFrame index
        # Define size patterns for extracting size-related information
        self.size_patterns = {
            'kg': [r'\b(\d+\.?\d*)\s*kg\b', r'\b(\d+\.?\d*)\s*kilo\b'],
            'g': [r'\b(\d+\.?\d*)\s*g\b', r'\b(\d+\.?\d*)\s*gram\b'],
            'l': [r'\b(\d+\.?\d*)\s*l\b', r'\b(\d+\.?\d*)\s*litre\b'],
            'ml': [r'\b(\d+\.?\d*)\s*ml\b', r'\b(\d+\.?\d*)\s*millilitre\b']
        }
        # Define mappings to translate common Hindi terms to English
        self.word_mappings = {
            'chawal': 'rice', 'atta': 'flour', 'dal': 'lentils', 'chini': 'sugar',
            'namak': 'salt', 'mirch': 'chili', 'haldi': 'turmeric', 'tel': 'oil',
            'ghee': 'ghee', 'namkeen': 'snacks', 'chai': 'tea', 'doodh': 'milk',
            'paani': 'water', 'chota': 'small', 'bada': 'large', 'sabun': 'soap',
            'masala': 'spice', 'shakkar': 'sugar'
        }
        self.preprocess_data()  # Preprocess the input DataFrame
        # Initialize the TF-IDF vectorizer
        self.tfidf = TfidfVectorizer(stop_words='english')
        # Generate the TF-IDF matrix for the search text
        self.tfidf_matrix = self.tfidf.fit_transform(self.df['SearchText'])
        self.train_word2vec()  # Train the Word2Vec model
        # Initialize metrics to track performance
        self.metrics = {
            'query_times': [],
            'total_queries': 0,
            'successful_queries': 0
        }

    def preprocess_data(self):
        """Preprocess the dataset."""
        # Combine relevant columns into a single searchable text field
        self.df['SearchText'] = (
            self.df['Name'].fillna('') + ' ' +
            self.df['Description'].fillna('') + ' ' +
            self.df['Subcategory'].fillna('') + ' ' +
            self.df['Brand_Name'].fillna('') + ' ' +
            self.df['CommonOrGenericNameOfCommodity'].fillna('') + ' ' +
            self.df['NetQuantity'].fillna('')
        ).str.lower()
        # Extract size information from the Description column
        self.df['Size'] = self.df['Description'].apply(
            lambda x: self.extract_size(x) if isinstance(x, str) else 'Unknown'
        )

    def extract_size(self, description: str) -> str:
        """Extract size information from product description."""
        # Search for size patterns in the description
        for size_category, patterns in self.size_patterns.items():
            for pattern in patterns:
                match = re.search(pattern, description, re.IGNORECASE)
                if match:
                    return f"{match.group(1)} {size_category}"  # Return the matched size
        return 'Unknown'  # Default to 'Unknown' if no match is found

    def translate_query(self, query: str) -> str:
        """Translate Hindi words in query to English."""
        # Translate each word in the query based on word mappings
        words = query.lower().split()
        return ' '.join([self.word_mappings.get(word, word) for word in words])

    def expand_query(self, query: str) -> List[str]:
        """Expand query with variations and synonyms."""
        expanded_queries = [query]  # Start with the original query
        for term in query.split():
            try:
                # Get similar words from the Word2Vec model
                similar_words = self.word2vec_model.wv.most_similar(term, topn=3)
                expanded_queries.extend([word for word, _ in similar_words])
            except KeyError:
                # Ignore words not found in the Word2Vec model
                continue
        return list(set(expanded_queries))  # Remove duplicates and return

    def train_word2vec(self):
        """Train Word2Vec model for semantic search."""
        # Tokenize the SearchText field for Word2Vec training
        sentences = [word_tokenize(text) for text in self.df['SearchText']]
        # Train the Word2Vec model
        self.word2vec_model = Word2Vec(sentences, vector_size=100, window=5, min_count=1, workers=4)

    def get_fuzzy_matches(self, query: str, min_score: int = 60) -> List[int]:
        """Get fuzzy matches for query."""
        matches = []  # Initialize list to store matching indices and scores
        for idx, row in self.df.iterrows():
            # Calculate the fuzzy match score between the query and product name
            score = fuzz.partial_ratio(query.lower(), row['Name'].lower())
            if score >= min_score:  # Only include matches above the threshold
                matches.append((idx, score))
        # Sort matches by score in descending order
        return sorted(matches, key=lambda x: x[1], reverse=True)

    def calculate_semantic_similarity(self, query: str, doc: str) -> float:
        """Calculate semantic similarity between query and document."""
        # Transform query and document into TF-IDF vectors
        query_vector = self.tfidf.transform([query])
        doc_vector = self.tfidf.transform([doc])
        # Calculate the cosine similarity
        return (query_vector * doc_vector.T).toarray()[0][0]

    def search(self, query: str, top_n: int = 10) -> pd.DataFrame:
        """Perform an enhanced search."""
        start_time = time.time()  # Record the start time for performance tracking
        self.metrics['total_queries'] += 1  # Increment the query count
        query = self.translate_query(query)  # Translate the query
        expanded_queries = self.expand_query(query)  # Expand the query with synonyms

        results = []  # Initialize list for search results
        for variation in expanded_queries:
            results.extend(self.get_fuzzy_matches(variation))  # Get fuzzy matches for each variation

        # Remove duplicate indices while keeping the best scores
        unique_results = {idx: score for idx, score in results}
        # Combine fuzzy scores with semantic similarity scores
        combined_scores = [
            (idx, score + self.calculate_semantic_similarity(query, self.df.loc[idx, 'SearchText']) * 100)
            for idx, score in unique_results.items()
        ]
        # Sort by combined scores and limit to top_n results
        sorted_scores = sorted(combined_scores, key=lambda x: x[1], reverse=True)[:top_n]

        # Record the query time
        self.metrics['query_times'].append(time.time() - start_time)
        # Return the DataFrame rows corresponding to the top results
        return self.df.loc[[idx for idx, _ in sorted_scores]]

    def get_metrics(self):
        """Return performance metrics."""
        # Calculate the average query time
        avg_time = np.mean(self.metrics['query_times'])
        return {
            'average_query_time': avg_time,
            'total_queries': self.metrics['total_queries'],
            'successful_queries': self.metrics['successful_queries']
        }

def demo_search():
    """Demo function to test the search engine."""
    # Load dataset
    df = pd.read_csv('groceryData.csv')
    
    # Initialize search engine
    search_engine = EnhancedRetailSearch(df)
    
    # Test queries
    test_queries = [
        "Rice",                    # Basic search
        "Rise",                    # Fuzzy search
        "chawal",                  # Hindi search
        "beverages",               # Category search
        "small pack harpic",       # Size + brand search
        "bada paani bottle",       # Hindi + size + product
        "cleaning products",       # Category search
        "water bottle 1l",         # Product + size search
        "chota dettol",            # Hindi + brand
        "masala",                  # Hindi category
        "organic products",        # Semantic search
        "healthy snacks",          # Semantic category search
        "1kg sugar",               # Size + product search
    ]
    
    # Run test queries
    for query in test_queries:
        print(f"\nSearching for: {query}")
        results = search_engine.search(query)
        if len(results) > 0:
            print(results[['Name', 'Subcategory', 'Price']])  # Display relevant columns
        else:
            print("No results found")  # Handle cases with no matches
    
    # Print performance metrics
    print("\nPerformance Metrics:")
    print(search_engine.get_metrics())

if __name__ == "__main__":
    demo_search()  # Run the demo when the script is executed