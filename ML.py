# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
import numpy as np

#Loading the dataset
file_path = 'C:\\Users\\Hp\\Desktop\\sentimentanalysis\\reviews.csv'
spotify_dataset = pd.read_csv(file_path)

#Cleaning the textual data
spotify_dataset['Cleaned_Review'] = spotify_dataset['Review'].str.lower().str.replace('[^\w\s]', '', regex=True)

#Labeling the data based on the Rating
conditions = [
    (spotify_dataset['Rating'] >= 4),
    (spotify_dataset['Rating'] == 3),
    (spotify_dataset['Rating'] <= 2)
]
choices = ['positive', 'neutral', 'negative']
spotify_dataset['Sentiment'] = np.select(conditions, choices)

# Feature extraction using TF-IDF
tfidf = TfidfVectorizer(max_features=5000)
X = tfidf.fit_transform(spotify_dataset['Cleaned_Review'])
y = spotify_dataset['Sentiment']

# Label encoding for sentiment labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Splitting the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)


##Model Training and evaluation

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# Train the model
lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train, y_train)

# Evaluate the model
y_pred = lr_model.predict(X_test)
print(classification_report(y_test, y_pred))

## Saving the model values for streamlit integration
import joblib

# Example of saving the model and other components
joblib.dump(lr_model, 'logistic_regression_model.pkl')
joblib.dump(tfidf, 'tfidf_vectorizer.pkl')
joblib.dump(le, 'label_encoder.pkl')

