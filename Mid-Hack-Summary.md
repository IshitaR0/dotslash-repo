# **Mid-Hack Evaluation Submission**

## **Team Name**:  
Femme - Devs

## **Project Title**:  
**Semantic Search for Products - Bizom + New Frontiers**

---

1. **Relevance & Impact**:
   - Solves a critical problem in grocery retail by improving search accuracy and usability.
   - Enhances the shopping experience for consumers and businesses by overcoming keyword-based search limitations.

2. **Preliminary Research**:
   - Studied pain points in traditional search systems.
   - Explored fuzzy matching, semantic search, and regional language handling to align with consumer needs.

3. **Proposed Methodology**:
   - Combined traditional search (exact match) with advanced techniques (fuzzy search, embeddings).
   - Integrated Hindi vocabulary mapping for regional language queries.

4. **Team Collaboration**:
   - Clear task distribution and regular GitHub updates.
   - Maintained transparency in progress clear communication.

--- 

## **Progress Summary**

1. **GitHub Repository**:   
   - **Tag**: Mid-hack tagged at Commit.

2. **Implemented Features**:
   - **Data Cleaning and Preparation**:
     - Preprocessed a dataset of over 50,000 grocery products by handling missing values, currency normalization, and text standardization through EDA and general word popularity trend evaluation.
   - **Exact Word Search**:
     - Developed a working search system for exact keyword matches.
   - **Fuzzy Search - WIP**:
     - Implemented fuzzy logic to handle typos in user queries (e.g., "rise" correctly retrieves "rice").
   - **Regional Language Support -  WIP**:
     - Created Hindi-to-English mappings for common grocery terms (e.g., "chawal" → "rice").
     - Exploring transliteration support for broader coverage of Hindi vocabulary.
   - **Preliminary Front-End**:
     - Developed a simple UI to input queries and view certain items (not integrated).
   - **Semantic Search (Not Started)**:
     - Embedding creation exploration, using **SentenceTransformer**, **BERT**, **TF-IDF**, **Word-2-Vec** to support intent-based queries like "show beverages."

3. **Collaboration and Workflow**:
   - Clear division of responsibilities:
     - Developers handling data processing and feature implementation.
     - Designer focusing on front-end and UX.
     - Project Manager ensuring deliverable timelines.

---

## **Mid-Hack Deliverables**

1. **Code Functionality**:
   - Working **exact word search**.
   - FIrst iteration of **Frontend**.
2. **Repository Structure - WIP**:
   - `requirements.txt`: Lists all project dependencies.

3. **Documentation**:
   - **README.md - WIP**: Overview of the project, setup instructions, and features.

---


## **Challenges Faced**

1. **Handling Large Dataset**:
   - Optimizing search functions for a 50,000+ product dataset, evaluating company size for items to be shown (our current logic is in ascending order of price).

2. **Semantic Search Development**:
   - Creating and fine-tuning embeddings to interpret intent-based queries effectively, working in a dtaset where item name isnt the most intuitive, both hindi and english nme items present, with different spellings (e.g. chawal and chawwal).

3. **Regional Language Support**:
   - Expanding Hindi vocabulary mappings to cover diverse queries while exploring transliteration.

4. **Front-End Integration**:
   - Ensuring a smooth interface that supports all search functionalities.
   - Front end for working across devices

---

## **Planned Next Steps**

1. **Complete Semantic Search**:
   - Finalize embedding-based semantic search.
   - Add support for queries like "show beverages" or "small packs of Harpic."

2. **Implement Product Suggestions**:
   - Introduce alternative product recommendations when specific items are unavailable (e.g., "Lysol" when "Harpic" is out of stock).

3. **Enhance Hindi Language Support**:
   - Broaden the vocabulary dictionary.
   - Implement query transliteration for better handling of regional language searches.

4. **Improve UI/UX**:
   - Enhance front-end features like usiness type, price sorting, and detailed product results.

---

## **Future Vision**

1. **Scalability**:
   - Expand support for additional regional languages (e.g., Tamil, Telugu).
   - Optimize for larger datasets in multi-million product retail environments.

2. **API Integration**:
   - Still choosing which to finalise on.

3. **Advanced Recommendations**:
   - Use collaborative filtering to offer personalized product suggestions.



Here are some relevant GitHub repositories, libraries, and resources to help with your grocery-specific search engine project:

https://www.youtube.com/watch?v=vHJPaNBdOyA

https://www.geeksforgeeks.org/python-introduction-to-web-development-using-flask/#:~:text=Flask%20is%20a%20lightweight%20backend,learn%20and%20use%20for%20beginners.

https://www.kaggle.com/code/neesham/semantic-search-for-beginners

https://www.nltk.org/

1. *Fuzzy Search*: 
   - The [fuzzysearch Python library](https://github.com/taleinat/fuzzysearch) provides a way to implement approximate string matching, which is useful for handling spelling errors (e.g., "Rise" -> "Rice").
   - A [simple fuzzy search script](https://gist.github.com/umitdincel/89960b4f9c302c604443) demonstrates basic concepts of token-based fuzzy matching.

2. *Regional Language Support*:
   - The [Indic NLP Library](https://github.com/anoopkunchukuttan/indic_nlp_library) helps process and tokenize Indian languages like Hindi. It's beneficial for enabling Hindi-language queries.

3. *Semantic Search*:
   - [Haystack by deepset](https://github.com/deepset-ai/haystack): A robust framework for building semantic search solutions using embeddings, which is ideal for intent-based queries like "Give me beverages" or "Show me small pack sizes for Harpic".
   - [Sentence-Transformers](https://github.com/UKPLab/sentence-transformers): A library for creating embeddings to build semantic search systems effectively.

4. *Dataset Preparation and Management*:
   - The [Pandas Profiling library](https://github.com/pandas-profiling/pandas-profiling) helps analyze and clean datasets efficiently.