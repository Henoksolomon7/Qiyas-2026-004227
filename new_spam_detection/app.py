import joblib
import numpy as np
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# Load saved ML model and vectorizer
try:
    model = joblib.load('best_model.pkl')
    vectorizer = joblib.load('best_vectorizer.pkl')
    print("Model and Vectorizer loaded successfully!")
except Exception as e:
    print(f"Error loading model files: {e}")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    text = data.get('text', '')

    if not text.strip():
        return jsonify({'error': 'Please enter some text.'}), 400

    # Vectorize input text
    text_vec = vectorizer.transform([text])

    # Prediction
    prediction = int(model.predict(text_vec)[0])
    label = "Spam" if prediction == 1 else "Ham (Legitimate)"

    # Probability / Confidence scores (if supported by model)
    prob_spam, prob_ham = 0.5, 0.5
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(text_vec)[0]
        # Assuming index 1 is Spam, 0 is Ham
        prob_ham = float(probs[0])
        prob_spam = float(probs[1]) if len(probs) > 1 else 1.0 - prob_ham
    elif hasattr(model, "decision_function"):
        # For SVM / Perceptron linear decision distance
        score = float(model.decision_function(text_vec)[0])
        prob_spam = 1 / (1 + np.exp(-score))  # Sigmoid conversion
        prob_ham = 1.0 - prob_spam

    # Extract detected feature words for visualization
    feature_names = np.array(vectorizer.get_feature_names_out())
    nonzero_indices = text_vec.nonzero()[1]
    detected_words = list(feature_names[nonzero_indices])

    return jsonify({
        'prediction': label,
        'is_spam': prediction == 1,
        'prob_spam': round(prob_spam * 100, 2),
        'prob_ham': round(prob_ham * 100, 2),
        'words': detected_words
    })

if __name__ == '__main__':
    app.run(debug=True)