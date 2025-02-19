import re
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# Simple class to store term and its importance score
class TermScore:
    def __init__(self, term, score):
        self.term = term
        self.score = score
    
    def __str__(self):
        # Format like: word(0.123)
        return f"{self.term}({self.score:.3f})"

def clean_text(text):
    """
    Clean up text by removing special characters and converting to lowercase
    """
    # Remove anything that's not a letter or space
    cleaned = re.sub(r'[^a-zA-Z ]+', '', text)
    return cleaned.lower().strip()

def clean_texts_in_dataframe(df, text_column='text'):
    """
    Clean all texts in a DataFrame column
    """
    if text_column not in df.columns:
        print(f"Error: '{text_column}' column not found!")
        return None
        
    # Make a copy so we don't change the original
    df_copy = df.copy()
    df_copy[text_column] = df_copy[text_column].apply(clean_text)
    return df_copy

class TextProcessor:
    """
    Main class to handle text processing using TF-IDF
    """
    def __init__(self, max_features=None):
        # Initialize the TF-IDF vectorizer
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            stop_words='english',              
        
        )
        
    def calculate_tfidf(self, documents):
        # Convert documents to TF-IDF matrix
        tfidf_matrix = self.vectorizer.fit_transform(documents)
        
        # Get the words (features) that correspond to the matrix columns
        feature_names = self.vectorizer.get_feature_names_out()
        
        return {
            'matrix': tfidf_matrix.todense(),
            'words': feature_names
        }
    
    def process_documents(self, documents, max_words=100, show_scores=True):
        """
        Process documents and keep only the most important words based on TF-IDF
        """
        # First clean the texts
        cleaned_docs = [clean_text(doc) for doc in documents]
        
        # Calculate TF-IDF
        result = self.calculate_tfidf(cleaned_docs)
        matrix = result['matrix']
        words = result['words']
        
        processed_documents = []
        
        # Go through each document
        for doc_idx, doc_scores in enumerate(matrix):
            # Create a dictionary of word:score pairs
            word_scores = {}
            for word_idx, score in enumerate(doc_scores.T):
                word_scores[words[word_idx]] = float(score)
            
            # Sort words by score and keep top ones
            sorted_words = sorted(word_scores.items(), key=lambda x: x[1], reverse=True)
            top_words = dict(sorted_words[:max_words])
            
            # Process original document words
            processed_words = []
            for word in cleaned_docs[doc_idx].split():
                if word in top_words:
                    if show_scores:
                        # Add the word with its score
                        term = TermScore(word, top_words[word])
                        processed_words.append(str(term))
                    else:
                        processed_words.append(word)
            
            processed_documents.append(' '.join(processed_words))
        
        return processed_documents
    
    
    def get_word_scores(self, result, doc_index, min_score=0.0):
        """
        Get all words and their importance scores for a specific document
        """
        doc_scores = result['matrix'][doc_index]
        
        # Create list of word-score pairs
        word_scores = []
        for word_idx, score in enumerate(doc_scores.T):
            score_value = float(score)
            if score_value >= min_score:
                word_scores.append(
                    TermScore(result['words'][word_idx], score_value)
                )
        
        # Sort by score (highest first)
        return sorted(word_scores, key=lambda x: x.score, reverse=True)

def process_file(input_file, output_file, text_column='text', max_words=100):
    """
    Process text data from a file and save results
    """
    try:
        # Read the file (works with CSV or TSV)
        if input_file.endswith('.tsv'):
            df = pd.read_csv(input_file, sep='\t')
        else:
            df = pd.read_csv(input_file)
        
        # Process the texts
        processor = TextProcessor()
        df = clean_texts_in_dataframe(df, text_column)
        texts = df[text_column].tolist()
        
        # Get TF-IDF scores and process
        processed_texts = processor.process_documents(texts, max_words)
        
        # Update and save
        df[text_column] = processed_texts
        df.to_csv(output_file, sep='\t', index=False)
        
        return df
        
    except FileNotFoundError:
        print(f"Error: Couldn't find file: {input_file}")
        return None

# Example usage / testing
if __name__ == "__main__":
    # Test with some example documents
    test_docs = [
        "I enjoy reading about Machine Learning and Machine Learning is my PhD subject",
        "I would enjoy a walk in the park",
        "I was reading in the library"
    ]
    
    # Create test DataFrame
    test_df = pd.DataFrame({"text": test_docs})
    
    # Create processor and process documents
    processor = TextProcessor()
    
    # Clean the texts
    cleaned_df = clean_texts_in_dataframe(test_df)
    cleaned_texts = cleaned_df["text"].tolist()
    
    # Process documents keeping top 5 words with scores
    processed_texts = processor.process_documents(
        cleaned_texts,
        max_words=5,
        show_scores=True
    )

    #open complaint documents
    file_path = 'data/complaintDocs_BertForClass.txt'
    complaint_df = pd.read_csv(file_path, sep='\t', names=['docid', 'text', 'MotionResultCode'])
    docs_df = pd.DataFrame(complaint_df, columns=['text'])

    #process
    processor = TextProcessor()

    #clean 
    cleaned_df = clean_texts_in_dataframe(docs_df)
    cleaned_texts = cleaned_df["text"].tolist()

    #process and keep top 5 words
    processed_texts = processor.process_documents(
        cleaned_texts,
        max_words=5,
        show_scores=True
    )

    #results
    docs_df["processed_text"] = processed_texts
    print("\nProcessed Texts with Scores:")
    print(docs_df[["text", "processed_text"]])
    
    # Show results
    """
    test_df["processed_text"] = processed_texts
    print("\nProcessed Texts with Scores:")
    print(test_df[["text", "processed_text"]])
    """

