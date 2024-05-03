import streamlit as st
import pickle
import numpy as np
import re
import base64
import warnings
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd 


warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore")
count=0

def increment():
    global count 
    count+=1
warnings.filterwarnings("ignore")

# Function to read and encode an image file in base64
def get_image_base64(path):
    with open(path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode('utf-8')

# Define the Class for text preprocessing
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

# Streamlit interface setup
st.title('Wine Finder: Find Your Flavor, Discover Your Perfect Wine')

# Display the image
image_path = "wine.png"
image_base64 = get_image_base64(image_path)
html_str = f"<img src='data:image/png;base64,{image_base64}' width='300'/>"
html_str = f"""
<div style="text-align: center;">
    <img src='data:image/png;base64,{image_base64}' width='300'/>
</div>
"""
st.markdown(html_str, unsafe_allow_html=True)

st.write('Describe your ideal wine: Include preferred flavors, notes, or aromas.')
# Initialize session state for recommendation type if not already set
if 'rec_type' not in st.session_state:
    st.session_state.rec_type = None

# Buttons for recommendation type
col1, col2 = st.columns(2)
with col1:
    if st.button('Search for categories'):
        st.session_state.rec_type = 'By Variety'
with col2:
    if st.button('Search for labels'):
        st.session_state.rec_type = 'By Name'

# Proceed with logic based on selection
if st.session_state.rec_type is not None:
    if st.session_state.rec_type == 'By Variety':
        
        # Load the saved components for variety
        with open("classification_model.pkl", 'rb') as file:
            pipeline, data, multilabel_binarizer = pickle.load(file)

        user_input_description = st.text_input("Enter your wine preferences",key=count)
        increment()
        if st.button('Recommend Wines By Category'):
            # Making prediction
            y_user_pred_proba = pipeline.predict_proba([user_input_description])  # Correct use of predict_proba
            
            # Get the indices of the probabilities sorted in descending order
            sorted_indices = np.argsort(-y_user_pred_proba, axis=1)[0]
            
            # Initialize a set to keep track of unique varieties recommended
            unique_varieties = set()
            unique_recommendations = []
            
            # Loop over sorted indices and get unique varieties
            for idx in sorted_indices:
                variety = data.iloc[idx]['variety']
                if variety not in unique_varieties:
                    unique_varieties.add(variety)
                    unique_recommendations.append(data.iloc[idx])
            
                    # Break the loop if we have enough recommendations
                    if len(unique_recommendations) == 5:
                        break
            
            # Convert the list of unique recommendations to a DataFrame for easy viewing
            unique_recommendations_df = pd.DataFrame(unique_recommendations)
            st.subheader('Your Personalized Wine Categories')
            st.write("Discover the wine styles that align with your taste preferences. Explore these top selections curated just for you:")
            i=0
            for index, row in unique_recommendations_df.iterrows():
                i+=1
                st.write(f"**{i}. {row['variety']}**")
                st.write(f"**Description:** {row['description']}")
                st.write("---")

    elif st.session_state.rec_type == 'By Name':
        
        # Load the pipeline and vectorized data for name
        with open('doc_similarity_model.pkl', 'rb') as file:
            pipeline, data_vectors, data = pickle.load(file)

        input_text = st.text_input("Enter your wine preferences by name",key=count)
        increment()
        if st.button('Recommend Wines by Label'):
            input_vector = pipeline.transform([input_text])
            cosine_sim_matrix = cosine_similarity(input_vector, data_vectors)
            top_5_indices = np
            # Calculate cosine similarity between the input text vector and all data_vectors
            cosine_sim_matrix = cosine_similarity(input_vector, data_vectors)
            
            # Find the indices of the top 5 recommendations based on cosine similarity
            top_5_indices = np.argsort(-cosine_sim_matrix, axis=1)[:, :5][0]
            
            # Retrieve the top 5 recommendations from the original data
            recommendations = data.iloc[top_5_indices]
            
            # Display the information about the top 5 wine recommendations by name
            st.subheader('Curated Wine Selections For You:')
            st.write('Your taste preferences have led us to these exquisite wine labels. Each selection is tailored to match your unique palate:')
            i=0
            for index, row in recommendations.iterrows():
                i+=1
                st.write(f"**{i}. {row['title']}**")
                st.write(f"**Category:** {row['variety']}")
                st.write(f"**Region:** {row['province']}")
                st.write(f"**Description:** {row['description']}")
                st.write(f"**Price:** ${row['price']}")
                st.write("---")                            