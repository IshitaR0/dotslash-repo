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
        self.df = df.copy()
        self.df.reset_index(inplace=True, drop=True)
        self.size_patterns = {
            'kg': [r'\b(\d+\.?\d*)\s*kg\b', r'\b(\d+\.?\d*)\s*kilo\b'],
            'g': [r'\b(\d+\.?\d*)\s*g\b', r'\b(\d+\.?\d*)\s*gram\b'],
            'l': [r'\b(\d+\.?\d*)\s*l\b', r'\b(\d+\.?\d*)\s*litre\b'],
            'ml': [r'\b(\d+\.?\d*)\s*ml\b', r'\b(\d+\.?\d*)\s*millilitre\b']
        }
        self.word_mappings = {
            'chawal': 'rice', 'atta': 'flour', 'dal': 'lentils', 'chini': 'sugar',
            'namak': 'salt', 'mirch': 'chili', 'haldi': 'turmeric', 'tel': 'oil',
            'ghee': 'ghee', 'namkeen': 'snacks', 'chai': 'tea', 'doodh': 'milk',
            'paani': 'water', 'chota': 'small', 'bada': 'large', 'sabun': 'soap',
            'masala': 'spice', 'shakkar': 'sugar'
        }
        self.preprocess_data()
        self.tfidf = TfidfVectorizer(stop_words='english')
        self.tfidf_matrix = self.tfidf.fit_transform(self.df['SearchText'])
        self.train_word2vec()
        self.metrics = {
            'query_times': [],
            'total_queries': 0,
            'successful_queries': 0
        }

    def preprocess_data(self):
        """Preprocess the dataset."""
        self.df['SearchText'] = (
            self.df['Name'].fillna('') + ' ' +
            self.df['Description'].fillna('') + ' ' +
            self.df['Subcategory'].fillna('') + ' ' +
            self.df['Brand_Name'].fillna('') + ' ' +
            self.df['CommonOrGenericNameOfCommodity'].fillna('') + ' ' +
            self.df['NetQuantity'].fillna('')
        ).str.lower()
        self.df['Size'] = self.df['Description'].apply(
            lambda x: self.extract_size(x) if isinstance(x, str) else 'Unknown'
        )

    def extract_size(self, description: str) -> str:
        """Extract size information from product description."""
        for size_category, patterns in self.size_patterns.items():
            for pattern in patterns:
                match = re.search(pattern, description, re.IGNORECASE)
                if match:
                    return f"{match.group(1)} {size_category}"
        return 'Unknown'

    def translate_query(self, query: str) -> str:
        """Translate Hindi words in query to English."""
        words = query.lower().split()
        return ' '.join([self.word_mappings.get(word, word) for word in words])

    def expand_query(self, query: str) -> List[str]:
        """Expand query with variations and synonyms."""
        expanded_queries = [query]
        for term in query.split():
            try:
                similar_words = self.word2vec_model.wv.most_similar(term, topn=3)
                expanded_queries.extend([word for word, _ in similar_words])
            except KeyError:
                continue
        return list(set(expanded_queries))

    def train_word2vec(self):
        """Train Word2Vec model for semantic search."""
        sentences = [word_tokenize(text) for text in self.df['SearchText']]
        self.word2vec_model = Word2Vec(sentences, vector_size=100, window=5, min_count=1, workers=4)

    def get_fuzzy_matches(self, query: str, min_score: int = 40) -> List[int]:
        """Get fuzzy matches for query."""
        matches = []
        for idx, row in self.df.iterrows():
            score = fuzz.partial_ratio(query.lower(), row['Name'].lower())
            if score >= min_score:
                matches.append((idx, score))
        return sorted(matches, key=lambda x: x[1], reverse=True)

    def calculate_semantic_similarity(self, query: str, doc: str) -> float:
        """Calculate semantic similarity between query and document."""
        query_vector = self.tfidf.transform([query])
        doc_vector = self.tfidf.transform([doc])
        return (query_vector * doc_vector.T).toarray()[0][0]

    def search(self, query: str, top_n: int = 10, category: str = None, max_price: float = None) -> pd.DataFrame:
        """Perform an enhanced search."""
        start_time = time.time()
        self.metrics['total_queries'] += 1
        query = self.translate_query(query)
        expanded_queries = self.expand_query(query)

        # Create a copy of the dataframe to avoid modifying the original
        df_copy = self.df.copy()

        # Filter the dataset based on the category and price range
        if category:
            df_copy = df_copy[df_copy['Subcategory'].str.lower().isin(category)]
        if max_price:
            df_copy = df_copy[df_copy['Price'] <= max_price]

        results = []
        for variation in expanded_queries:
            results.extend(self.get_fuzzy_matches(variation))

        unique_results = {idx: score for idx, score in results}
        combined_scores = [
            (idx, score + self.calculate_semantic_similarity(query, self.df.loc[idx, 'SearchText']) * 100)
            for idx, score in unique_results.items()
        ]
        sorted_scores = sorted(combined_scores, key=lambda x: x[1], reverse=True)[:top_n]

        self.metrics['query_times'].append(time.time() - start_time)
        return self.df.loc[[idx for idx, _ in sorted_scores]]

    def get_metrics(self):
        """Return performance metrics."""
        avg_time = np.mean(self.metrics['query_times'])
        return {
            'average_query_time': avg_time,
            'total_queries': self.metrics['total_queries'],
            'successful_queries': self.metrics['successful_queries']
        }

def demo_search():
    """Demo function to test the search engine."""
    df = pd.read_csv('groceryData.csv')
    search_engine = EnhancedRetailSearch(df)

    # Test queries with the new format: "A"/"B"/"C"/"D" and food item
    test_queries = [
        ("A", "Rice"),                 # Category A (sweets and namkeen, price < 150)
        ("B", "Harpic"),               # Category B (price < 300)
        ("C", "Organic Oil"),          # Category C (price < 3000)
        ("D", "Sugar"),                # Category D (no price restriction)
        ("A", "Chai")                  # Category A (sweets and namkeen, price < 150)
    ]

    # Run test queries
    for category, food_item in test_queries:
        print(f"\nSearching for: {food_item} with filter: {category}")
        
        # Define filters based on the category
        if category == "A":
            results = search_engine.search(food_item, category=["Snacks and Namkeen"], max_price=150)
        elif category == "B":
            results = search_engine.search(food_item, max_price=300)
        elif category == "C":
            results = search_engine.search(food_item, max_price=3000)
        else:
            results = search_engine.search(food_item)  # No filter for category D

        # Display the results
        if len(results) > 0:
            print(results[['Name', 'Subcategory', 'Price']])
        else:
            print("No results found")
    
    # Print performance metrics
    print("\nPerformance Metrics:")
    print(search_engine.get_metrics())

if __name__ == "__main__":
    demo_search()
