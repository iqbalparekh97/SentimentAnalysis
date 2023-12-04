# SentimentAnalysis
Performing sentiment analysis using textual data from Spotify dataset.

# Abstract 
This project focused on sentiment analysis of user reviews for Spotify, utilizing a Logistic Regression model. The primary objective was to classify reviews into three sentiment categories: positive, neutral, and negative. Initially, a dataset of Spotify reviews was preprocessed, involving lowercasing and punctuation removal to clean the textual data. The reviews were then labeled based on their ratings: 4 or 5 as positive, 3 as neutral, and 1 or 2 as negative.

The outcomes of this project underscore the effectiveness of Logistic Regression in text classification tasks and reveal insights into the nuances of sentiment analysis, particularly in dealing with neutral sentiments. The findings and methodologies employed in this project provide a foundation for further research and development in the field of sentiment analysis and natural language processing.

# Data Description
The data extracted and cleaned in this project consisted of user reviews for Spotify. The data cleaning and preparation involved standardizing the text, simplifying it by removing unnecessary characters, and converting the raw text into a format suitable for machine learning, all while ensuring that the inherent sentiment information was accurately captured and labeled.

# Algorithm Description. 
The web application is driven by a series of algorithms, each playing a critical role in processing the input and delivering the output. Here's an outline of these algorithms:

1. User Interface Input Handling:
The web app provides an interface for users to input or upload Spotify reviews. This input is captured and sent to the backend for processing.

2. Text Preprocessing Algorithm:
- Lowercasing: Converts the input text to lowercase to maintain consistency.
- Punctuation Removal: Strips the text of punctuation marks to simplify the text data.
These steps ensure that the input text is standardized and cleaned, similar to the preprocessing done on the training dataset.

3. TF-IDF Vectorization Algorithm:
Converts the preprocessed text into a numerical format using the TF-IDF (Term Frequency-Inverse Document Frequency) method.
This algorithm is essential for transforming text into a feature vector that can be understood and processed by the machine learning model. It reflects the importance of each word in the context of the corpus used for training.

4. Sentiment Prediction Algorithm (Logistic Regression Model):
The Logistic Regression model, trained on the cleaned Spotify review dataset, is the core algorithm for sentiment prediction.
It takes the TF-IDF vector as input and predicts the sentiment class (positive, neutral, negative) based on the learned patterns from the training data.

5. Label Decoding Algorithm:
The predicted sentiment, represented as a numerical class, is translated back into a readable label (‘positive’, ‘neutral’, ‘negative’) using the label mapping established during the training phase.

6. Response Handling Algorithm:
Prepares the output, which includes the predicted sentiment, and possibly additional information like prediction confidence.
Sends this output back to the user interface, where it is displayed to the user.

# Tools Used
1 Machine Learning Code:

- Pandas: Used for data manipulation and analysis. In this code, it's used to load and preprocess the dataset.
- Scikit-learn (sklearn): A machine learning library used for various tasks. In this code, it includes:
    -train_test_split: Splits the dataset into training and testing sets.
    -TfidfVectorizer: Converts the textual data to a matrix of TF-IDF features.
    -LabelEncoder: Encodes target labels with value between 0 and n_classes-1.
    -LogisticRegression: The machine learning model used for classification.
    -classification_report: Used to measure the quality of predictions from the logistic regression model.
- NumPy: A library for large, multi-dimensional arrays and matrices, along with a large collection of high-level mathematical functions to operate on these arrays.
- Joblib: Used for saving the trained machine learning models and other components for later use.

2. Streamlit Code:

- Streamlit: An open-source app framework for Machine Learning and Data Science projects. In this code, it is used to create a web interface for the sentiment analysis model.
- Joblib: Here, it's used to load the previously saved machine learning models and components.
