import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem.porter import PorterStemmer
import nltk
from numpy import dot
import json
from numpy.linalg import norm
import pickle
import firebase_admin
from firebase_admin import credentials, db
from flask import Flask , request, jsonify
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

@app.route('/recommend')
def recommend():
    # Retrieve data from Firebase and convert to pandas dataframe
    ref = db.reference('/pbook-data')
    data = ref.get()
    df = pd.DataFrame.from_dict(data)

    ref = db.reference('/nayabook_data')
    bdata = ref.get()
    bdf = pd.DataFrame.from_dict(bdata)
    
    ref = db.reference('/Users')
    data = ref.get()
    udf = pd.DataFrame.from_dict(data)
    # book=datas['value']
# Create the count vectorizer and vectorize the data
    cv = CountVectorizer(max_features=5000, stop_words='english')
    vector = cv.fit_transform(df['tags']).toarray()
    book= request.args.get('query')
    b_index = df[df['Title'] == book].index[0]
# Calculate cosine similarity
    similarity = cosine_similarity(vector[b_index],vector)
    book_list = sorted(list(enumerate(similarity)),reverse=True, key=lambda x: x[1])[1:10]
    recommended_books=[]
    for i in book_list:
    # Get the book information
        title = df.iloc[i[0]].Title
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

@app.route('/update', methods =['POST'])
def update():
    ref = db.reference('/pre-processed_data')
    data = ref.get()
    df = pd.DataFrame.from_dict(data)

    ref = db.reference('/Users')
    data = ref.get()
    udf = pd.DataFrame.from_dict(data)
    data = request.get_json()
    email = data['email']
    book =data['value']
    # email = "check@gmail.com"
    # book= "The Temples of Lasha"
    ub_data = df.loc[df['Title'] == book, 'tags'].values[0]
    oub_data = udf.loc[udf['email'] == email, 'tags'].values[0]
    oub_index = str(udf[udf['email'] == email].index[0])
    data = {
        "email": email,
        "tags": ub_data + " " + oub_data
    }
    re = db.reference('/Users').child(oub_index)
    re.set(data)
    return 'updated'

@app.route ('/create', methods=['POST'])
def create_node():
    data = request.get_json()
    email = data['email']
    tags=data['faculty']
    data = {
        "email": email,
        "tags": tags,
    }
    # Get database reference to node
    ref = db.reference('/Users')
    # Check if node exists
    node = ref.get()
    if node is not None:
        datas = ref.get()
        ref = pd.DataFrame.from_dict(datas)
        ind=ref.index[-1]
        ind=ind + 1
        re = db.reference('/Users').child(str(ind))
        re.set(data)
        return data
    else:
        re = db.reference('/Users').child('0')
        re.set(data)
        return data
    
@app.route ('/clear', methods=['POST'])
def clear_history():
    ref = db.reference('/Users')
    data = ref.get()
    udf = pd.DataFrame.from_dict(data)

    data = request.get_json()
    email = data['email']
    # email= "check@gmail.com"
    oub_data = udf.loc[udf['email'] == email, 'tags'].values[0]
    oub_index = str(udf[udf['email'] == email].index[0])
    tags=oub_data.split()
    tags= tags[-1]
    data = {
        "email": email,
        "tags": tags,
    }
    re = db.reference('/Users').child(oub_index)
    re.set(data)
    return 'history cleared'

@app.route('/auto', methods=['POST'])
def hello_world():
    ref = db.reference('/pre-processed_data')
    data = ref.get()
    df = pd.DataFrame.from_dict(data)
    ind=df.index[-1]
    ind=[ind + 1]

    ref = db.reference('/books-data')
    bdata = ref.get()
    bdf = pd.DataFrame.from_dict(bdata)

    ref = db.reference('/Users')
    data = ref.get()
    udf = pd.DataFrame.from_dict(data)
    
    # email= "check@gmail.com"
    data = request.get_json()
    email = data['email']
    oub_data = udf.loc[udf['email'] == email, 'tags'].values[0]
    data = {
        "Title": email,
        "tags": oub_data,
    }
    new_df = pd.DataFrame(data, index=ind)
    df = df.append(new_df, ignore_index=True)
# Create the count vectorizer and vectorize the data
    cv = CountVectorizer(max_features=5000, stop_words='english')
    vector = cv.fit_transform(df['tags']).toarray()

# Calculate cosine similarity
    similarity = cosine_similarity(vector[ind],vector)
    book_list = sorted(list(enumerate(similarity)),reverse=True, key=lambda x: x[1])[1:10]
    recommended_books=[]
    for i in book_list:
    # Get the book information
        title = df.iloc[i[0]].Title
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

if __name__ == "__main__":

    app.run(host="0.0.0.0",port=5000)