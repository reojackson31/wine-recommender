from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
nltk.download('wordnet')
nltk.download('omw-1.4')
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer

# Custom transformer for text preprocessing
class TextPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self, stop_words, lemmatizer):
        self.stop_words = stop_words
        self.lemmatizer = lemmatizer
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        processed_texts = []
        for text in X:
            text = re.sub('[^a-zA-Z]', ' ', text)
            text = text.lower()
            text_words = text.split()
            text_words = [word for word in text_words if word not in self.stop_words]
            text_words = [self.lemmatizer.lemmatize(word) for word in text_words]
            processed_texts.append(' '.join(text_words))
        return processed_texts


# Load data
data = pd.read_csv('wine_data.csv')

# Initialize the text preprocessor and TF-IDF vectorizer
stopword_list = stopwords.words('english')
lemmatizer = WordNetLemmatizer()
preprocessor = TextPreprocessor(stop_words=stopword_list, lemmatizer=lemmatizer)
vectorizer = TfidfVectorizer()
clf = OneVsRestClassifier(LogisticRegression(solver='saga', max_iter=1000, random_state=42))

# Create a pipeline
pipeline = Pipeline([
    ('text_preprocessing', preprocessor),
    ('tfidf_vectorization', vectorizer),
    ('classifier', clf)
])

multilabel_binarizer = MultiLabelBinarizer()
Y = multilabel_binarizer.fit_transform(data['variety'])

# Fit the pipeline to the 'description' data and transform it
pipeline.fit(data['description'], Y)

# Serialize the pipeline and the vectorized data
with open('classification_model.pkl', 'wb') as file:
    pickle.dump((pipeline, data, multilabel_binarizer), file)