# Wine Recommender based on reviews from sommeliers

## 1. Problem Description
In an era of unprecedented choice, consumers are often overwhelmed, especially when selecting wine—a product category with over 1,500 options on average in American supermarkets. The key to navigating this vast array is personalization. A staggering 91% of consumers are drawn to brands that offer personalized recommendations (Accenture). Our wine recommendation system cuts through the complexity, offering a user-friendly digital service that adapts to individual tastes and budgets. As e-commerce wine sales show promising growth, our system is not just a convenience but a necessity, aligning with the consumer’s desire for a curated and efficient shopping experience. This innovative tool meets the modern demand for tailored experiences in the digital marketplace, positioning businesses at the forefront of the industry’s evolution and driving customer satisfaction and loyalty.

## 2. Solution Demo

https://github.com/reojackson31/wine-recommender/assets/148725712/ab51fb36-086d-4145-9699-7c3a3b152562

## 3. Analytics Approach

### 3.1 Step 1: Text Preprocessing
Preprocessing steps for wine descriptions in our dataset include converting it to lowercase, removing punctuations and stopwords, performing lemmatization, and vectorizing the data using TF-IDF and word2vec techniques to prepare it for modeling.

### 3.2 Step 2: Model Development
We trained 2 types of models for our recommendation system:

- Wine Category Prediction: This text classification model predicts the wine variety (e.g., Red Blend, Riesling, Chardonnay) based on the wine’s description. We tested three different algorithms: MultinomialNB, SGDClassifier, and LogisticRegression. The models were evaluated based on the hamming score and hamming loss, with LogisticRegression selected as the best model.

- Wine Label Recommendation: To recommend wine labels, we implemented a document similarity model that analyzes user input about their wine preferences (flavors, notes, aromas, etc.) and identifies the best match within our dataset. We experimented with five models - nearest neighbors using manhattan distance, euclidean distance, cosine similarity with TF-IDF vectorization and cosine similarity with word2vec. The cosine similarity approach with TF-IDF was selected since it gave the highest accuracy for recommendations with a test dataset.

### 3.3 Step 3: Frontend Integration and Deployment
After experimenting and selecting our final models, we created a pipeline and saved the models using pickle files. The final recommendation system was integrated into a user-friendly frontend web application developed with Streamlit. This interface accepts user preferences regarding wines, and runs the models to generate personalized wine recommendations to users.


## 4. Expected Impact
The deployment of our wine recommendation system is poised to make a significant impact in three key areas.

- Firstly, from a customer experience standpoint, the system will address the common issue of choice overload by delivering personalized wine selections. This targeted approach simplifies the decision-making process, allowing consumers to navigate the vast world of wine with ease and confidence.

- Secondly, the system is designed to optimize sales by providing recommendations that resonate with individual customer preferences, thereby driving higher conversion rates. Such personalization fosters a more engaging shopping experience that encourages purchase completion. 

- Lastly, the system will serve as a rich repository of consumer preference data, offering invaluable insights that can be leveraged to inform inventory management and strategic marketing initiatives. By aligning stock with consumer tastes and market demand, retailers can operate more efficiently and market their products more effectively. In essence, this recommendation system is not just a tool for simplification but a strategic asset that
contributes to a more intuitive, data-driven approach to the wine retail experience.


## 5. Details of Code Files:

1. text_eda.ipynb - Jupyter Notebook with exploratory data analysis done on text fields in the dataset including wine descriptions, variety, region of origin etc. EDA steps include creating word clouds, TF-IDF, n-gram analysis, MDS plot, word frequency bar plots etc.

2. classification_model_experiments.ipynb - Jupyter notebook with experiments done using different classification models to predict the variety of wine based on the description text. Models tested include MultinomialNB, SGDClassifier, and LogisticRegression.

3. classification_model_training.py - Python script for model training on the best classification model selected from experimentation (LogisticRegression). The code also includes creation of a pipeline, and saving the trained model and pipeline in a pickle file.

4. doc_similarity_model_experiments.ipynb - Jupyter notebook with experiments done using different document similarity models to find the best match for wines in the dataset based on an input text from the user. We experimented with five models - nearest neighbors using manhattan distance, euclidean distance, cosine similarity with TF-IDF vectorization and cosine similarity with word2vec.

5. doc_similarity_model_training.py - Python script for preprocessing steps and vectorizing the dataset. The code also includes creation of a pipeline, and saving the vectorized data and pipeline in a pickle file.

6. streamlit_wine_recommender.py - Python script for the frontend web application developed with Streamlit. This interface accepts user preferences regarding wines, and runs the models to generate personalized wine recommendations to users.


**Contributors:**
- [Oyundari Batbayar](https://github.com/obatbayar1)
- [Jaya Chaturvedi](https://github.com/Jaya2404)
- [Vinay Govias](https://github.com/vin1652)
- [Reo Paul Jackson](https://github.com/reojackson31)
- [Moiz Shaikh](https://github.com/MoizZahidShaikh)
- [Syed Ammad Sohail](https://github.com/ammadsohail99)


