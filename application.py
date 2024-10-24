from flask import Flask, request, jsonify
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import pickle

# Initialize Flask app
application = Flask(__name__)
app = application


# Load model and vectorizer
def load_model():
    with open('basic_classifier.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
    with open('count_vectorizer.pkl', 'rb') as vectorizer_file:
        vectorizer = pickle.load(vectorizer_file)
    return model, vectorizer


# Load the trained model and vectorizer
model, vectorizer = load_model()


@app.route('/')
def home():
    return jsonify({'message': 'Hello World!'})


@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    if 'text' not in data:
        return jsonify({'error': 'No text provided for prediction'}), 400

    text = data['text']
    transformed_text = vectorizer.transform([text])
    prediction = model.predict(transformed_text)[0]

    result = 'REAL' if prediction == 1 else 'FAKE'
    return jsonify({'prediction': result})


# WSGI entry point
if __name__ == '__main__':
    application.run(debug=True, host='0.0.0.0', port=8000)
