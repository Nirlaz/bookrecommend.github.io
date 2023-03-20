import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem.porter import PorterStemmer
import nltk
import pickle
from numpy import dot
from numpy.linalg import norm
import firebase_admin
from firebase_admin import credentials, db
from flask import Flask, request, jsonify

app = Flask(__name__)

# Initialize Firebase Admin SDK
cred = credentials.Certificate('C:/Users/N_.r_az/Desktop/project/projectcode/service-account-key.json')
firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://book-recommender-ab507-default-rtdb.firebaseio.com/'
})


def cosine_similarity(vector, vectors_list):
    similarity_scores = []
    for v in vectors_list:
        # Calculate dot product and magnitude of vectors
        dot_product = dot(vector, v)
        magnitude_product = norm(vector) * norm(v)
        # Calculate cosine similarity
        similarity = dot_product / magnitude_product
        similarity_scores.append(similarity)
    return similarity_scores


@app.route('/recommend', methods=['POST'])
def recommend_books():
    # Get the email from the request
    data = request.get_json()
    email = data['email']
    
    # Get the user's data from Firebase
    ref = db.reference('/Users')
    data = ref.get()
    udf = pd.DataFrame.from_dict(data)
    
    # Get the user's tags
    user_tags = udf.loc[udf['email'] == email, 'tags'].values[0]
    
    # Get the book data from Firebase
    ref = db.reference('/books-data')
    bdata = ref.get()
    bdf = pd.DataFrame.from_dict(bdata)
    
    # Create a dataframe with the user's data
    data = {
        "Title": email,
        "tags": user_tags,
    }
    df = pd.DataFrame(data, index=[0])
    
    # Create the count vectorizer and vectorize the data
    cv = CountVectorizer(max_features=5000, stop_words='english')
    vector = cv.fit_transform(df['tags']).toarray()
    
    # Calculate cosine similarity
    similarity = cosine_similarity(vector[0], vector)
    
    # Get the top 10 recommended books
    book_list = sorted(list(enumerate(similarity)), reverse=True, key=lambda x: x[1])[1:11]
    recommended_books = []
    for i in book_list:
        # Get the book information
        title = bdf.iloc[i[0]].Title
        author = bdf.iloc[i[0]].Author
        publisher = bdf.iloc[i[0]].Publisher
        faculty = bdf.iloc[i[0]].Faculty
        subject = bdf.iloc[i[0]].Subject
        # Create a dictionary representing the book
        book_dict = {
            'title': title,
            'author': author,
            'publisher': publisher,
            'faculty': faculty,
            'subject': subject
        }
        recommended_books.append(book_dict)
    return jsonify(recommended_books)


@app.route('/auto', methods=['POST'])
def auto_recommend():
    # Get the email from the request
    data = request.get_json()
    email = data['email']
    
    # Get the user's data from Firebase
    ref = db.reference('/Users')
    data = ref.get()
    udf = pd.DataFrame.from_dict(data)
    
    # Get the user's tags
    user_tags = udf.loc[udf['email'] == email, 'tags'].values[0]
    
    # Get the book data from Firebase
    ref = db.reference('/books-data')


if __name__ == "__main__":

    app.run(debug=True)
