# FIRST---------------------------------------------------------------------
# import pandas as pd
# from fuzzywuzzy import fuzz
# from typing import List

# class RetailSearch:
#     def __init__(self, df: pd.DataFrame):
#         self.df = df
        
#         # Unified Hindi-English mapping for grocery and retail terms
#         self.word_mappings = {
#             'chawal': 'rice', 'atta': 'flour', 'dal': 'lentils', 'chini': 'sugar', 'namak': 'salt',
#             'mirch': 'chilli', 'haldi': 'turmeric', 'tel': 'oil', 'ghee': 'clarified butter',
#             'namkeen': 'snacks', 'chai': 'tea', 'doodh': 'milk', 'paani': 'water',
#             'chota': 'small', 'bada': 'large', 'pack': 'pack', 'kirana': 'grocery',
#             'safai': 'cleaning', 'masale': 'spices', 'nashta': 'breakfast', 
#         }
        
#         # Categories and their Hindi equivalents
#         self.category_mappings = {
#             'grocery': ['Staples', 'Ready to Cook', 'Snacks and Namkeen'],
#             'cleaning': ['Cleaning & Household'],
#             'spices': ['Masala & Seasoning'],
#             'beverages': ['Energy and Soft Drinks', 'Tea and Coffee']
#         }
        
#         # Reverse mapping for English to Hindi
#         self.reverse_mappings = {v: k for k, v in self.word_mappings.items()}

#     def preprocess_query(self, query: str) -> List[str]:
#         """Convert query to lowercase and split into words."""
#         return query.lower().split()
    
#     def translate_query(self, query_words: List[str]) -> List[str]:
#         """Translate Hindi terms in the query to English."""
#         return [self.word_mappings.get(word, word) for word in query_words]
    
#     def fuzzy_match_score(self, query: str, target: str) -> int:
#         """Calculate fuzzy matching score between query and target."""
#         return fuzz.partial_ratio(query.lower(), target.lower())
    
#     def size_match(self, query: List[str], text: str) -> bool:
#         """Check for size-related terms in the query."""
#         size_terms = {
#             'chota': ['small', '100ml', '200ml'], 
#             'bada': ['large', '1l', '2l']
#         }
#         for hindi_term, sizes in size_terms.items():
#             if hindi_term in query:
#                 return any(size in text.lower() for size in sizes)
#         return False
    
#     def search(self, query: str, top_n: int = 10, min_score: int = 60) -> pd.DataFrame:
#         """
#         Search products using Hindi/English query with fuzzy matching.
        
#         Args:
#             query: Search query in Hindi (romanized) or English.
#             top_n: Number of results to return.
#             min_score: Minimum fuzzy match score to consider.
        
#         Returns:
#             DataFrame with matched products.
#         """
#         query_words = self.preprocess_query(query)
#         translated_words = self.translate_query(query_words)
#         query_str = ' '.join(translated_words)
        
#         scores = {}
        
#         for idx, row in self.df.iterrows():
#             max_score = 0
            
#             # Check name, description, and categories
#             for field in ['Name', 'Description', 'Category', 'Subcategory']:
#                 if pd.notna(row[field]):
#                     field_score = self.fuzzy_match_score(query_str, str(row[field]))
#                     max_score = max(max_score, field_score)
            
#             # Size matching adds a boost
#             if self.size_match(query_words, str(row['Name'])):
#                 max_score += 20
            
#             if max_score >= min_score:
#                 scores[idx] = max_score
        
#         # Sort and return top results
#         sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
#         top_indices = [idx for idx, score in sorted_scores[:top_n]]
#         results = self.df.loc[top_indices].copy()
#         results['match_score'] = [scores[idx] for idx in top_indices]
#         return results.sort_values('match_score', ascending=False)

# # Example Usage
# def demo_search():
#     # Load your dataset
#     df = pd.read_csv('final_dataset_1200am.csv')  # Replace with your actual dataset path
    
#     # Initialize RetailSearch
#     search_engine = RetailSearch(df)
    
#     # Test queries
#     queries = [
#         "chawal",  # Rice
#         "chota tel",  # Small oil
#         "masale",  # Spices
#         "bada paani bottle"  # Large water bottle
#     ]
    
#     for query in queries:
#         print(f"\nSearching for: {query}")
#         results = search_engine.search(query)
#         if not results.empty:
#             print(results[['Name', 'Price', 'match_score']].head())
#         else:
#             print("No results found")

# if __name__ == "__main__":
#     demo_search()

# SECOND----------------------------------------------------------------------
# from flask import Flask, request, jsonify
# import pandas as pd
# from fuzzywuzzy import fuzz
# from typing import List

# app = Flask(__name__)

# class RetailSearch:
#     def __init__(self, df: pd.DataFrame):
#         self.df = df
#         self.word_mappings = {
#             'chawal': 'rice', 'atta': 'flour', 'dal': 'lentils', 'chini': 'sugar', 'namak': 'salt',
#             'mirch': 'chilli', 'haldi': 'turmeric', 'tel': 'oil', 'ghee': 'clarified butter',
#             'namkeen': 'snacks', 'chai': 'tea', 'doodh': 'milk', 'paani': 'water',
#             'chota': 'small', 'bada': 'large', 'pack': 'pack', 'kirana': 'grocery',
#             'safai': 'cleaning', 'masale': 'spices', 'nashta': 'breakfast',
#         }

#     def preprocess_query(self, query: str) -> List[str]:
#         return query.lower().split()

#     def translate_query(self, query_words: List[str]) -> List[str]:
#         return [self.word_mappings.get(word, word) for word in query_words]

#     def fuzzy_match_score(self, query: str, target: str) -> int:
#         return fuzz.partial_ratio(query.lower(), target.lower())

#     def search(self, query: str, top_n: int = 10, min_score: int = 60) -> pd.DataFrame:
#         query_words = self.preprocess_query(query)
#         translated_words = self.translate_query(query_words)
#         query_str = ' '.join(translated_words)

#         scores = {}
#         for idx, row in self.df.iterrows():
#             max_score = 0
#             for field in ['Name', 'Description', 'Category', 'Subcategory']:
#                 if pd.notna(row[field]):
#                     field_score = self.fuzzy_match_score(query_str, str(row[field]))
#                     max_score = max(max_score, field_score)
#             if max_score >= min_score:
#                 scores[idx] = max_score

#         sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
#         top_indices = [idx for idx, score in sorted_scores[:top_n]]
#         results = self.df.loc[top_indices].copy()
#         results['match_score'] = [scores[idx] for idx in top_indices]
#         return results[['Name', 'Price', 'match_score']].to_dict(orient='records')

# @app.route('/search', methods=['POST'])
# def search():
#     data = request.get_json()  # Get the query from frontend
#     query = data.get('query', '')

#     # Load the dataset (assuming it's already loaded into df)
#     df = pd.read_csv('final_dataset_1200am.csv')  # Replace with your actual dataset path
#     search_engine = RetailSearch(df)

#     # Perform the search
#     results = search_engine.search(query)

#     return jsonify(results)  # Send results back to frontend as JSON

# if __name__ == "__main__":
#     app.run(debug=True)


# THIRD----------------------------------------------
from flask import Flask, request, jsonify
from flask_cors import CORS  # Make sure this import is here
import pandas as pd
from fuzzywuzzy import fuzz
from typing import List

# Initialize Flask app
app = Flask(__name__)

# Enable CORS for all routes
CORS(app)

class RetailSearch:
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.word_mappings = {
            'chawal': 'rice', 'atta': 'flour', 'dal': 'lentils', 'chini': 'sugar', 'namak': 'salt',
            'mirch': 'chilli', 'haldi': 'turmeric', 'tel': 'oil', 'ghee': 'clarified butter',
            'namkeen': 'snacks', 'chai': 'tea', 'doodh': 'milk', 'paani': 'water',
            'chota': 'small', 'bada': 'large', 'pack': 'pack', 'kirana': 'grocery',
            'safai': 'cleaning', 'masale': 'spices', 'nashta': 'breakfast',
        }
        self.reverse_mappings = {v: k for k, v in self.word_mappings.items()}

    def preprocess_query(self, query: str) -> List[str]:
        """Convert query to lowercase and split into words."""
        return query.lower().split()

    def translate_query(self, query_words: List[str]) -> List[str]:
        """Translate Hindi terms in the query to English."""
        return [self.word_mappings.get(word, word) for word in query_words]

    def fuzzy_match_score(self, query: str, target: str) -> int:
        """Calculate fuzzy matching score between query and target."""
        return fuzz.partial_ratio(query.lower(), target.lower())

    def size_match(self, query: List[str], text: str) -> bool:
        """Check for size-related terms in the query."""
        size_terms = {
            'chota': ['small', '100ml', '200ml'],
            'bada': ['large', '1l', '2l']
        }
        for hindi_term, sizes in size_terms.items():
            if hindi_term in query:
                return any(size in text.lower() for size in sizes)
        return False

    def search(self, query: str, top_n: int = 10, min_score: int = 60) -> pd.DataFrame:
        """Search products using Hindi/English query with fuzzy matching."""
        query_words = self.preprocess_query(query)
        translated_words = self.translate_query(query_words)
        query_str = ' '.join(translated_words)

        scores = {}
        for idx, row in self.df.iterrows():
            max_score = 0
            for field in ['Name', 'Description', 'Category', 'Subcategory']:
                if pd.notna(row[field]):
                    field_score = self.fuzzy_match_score(query_str, str(row[field]))
                    max_score = max(max_score, field_score)
            if self.size_match(query_words, str(row['Name'])):
                max_score += 20
            if max_score >= min_score:
                scores[idx] = max_score

        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        top_indices = [idx for idx, score in sorted_scores[:top_n]]
        results = self.df.loc[top_indices].copy()
        results['match_score'] = [scores[idx] for idx in top_indices]
        return results.sort_values('match_score', ascending=False)


# Load dataset
df = pd.read_csv('df_midhackuse.csv')  # Update with the actual dataset path
search_engine = RetailSearch(df)

@app.route('/search', methods=['POST'])
def search():
    try:
        # Get the query from the request body
        query = request.json.get('query', '').strip()
        if not query:
            return jsonify({'error': 'Query is required'}), 400

        # Perform search
        results = search_engine.search(query)

        # Format results as JSON
        response = results[['Name', 'Description', 'Price', 'match_score']].to_dict(orient='records')
        return jsonify(response), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5001)  # This runs the Flask app
