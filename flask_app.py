from flask import Flask, request, jsonify
import numpy as np
import pickle
from tensorflow.keras.models import load_model
import logging
from datetime import datetime

# Load the model and preprocessing artifacts
model = load_model('rnn_model.h5')

with open('tfidf_vectorizer.pkl', 'rb') as f:
    tfidf_vectorizer = pickle.load(f)

with open('label_encoding.pkl', 'rb') as f:
    label_encoding = pickle.load(f)

# Create a reverse label encoding map for decoding predictions
reverse_label_encoding = {v: k for k, v in label_encoding.items()}

# Configure logging
logging.basicConfig(filename='model_predictions.log', level=logging.INFO)

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    tokens = data['tokens']

    # Preprocess the input text
    text_input = ' '.join(tokens)
    X_tfidf = tfidf_vectorizer.transform([text_input])
    X_tfidf_reshaped = X_tfidf.toarray().reshape(1, 1, X_tfidf.shape[1])

    # Predict using the RNN model
    y_pred = model.predict(X_tfidf_reshaped)
    y_pred_classes = np.argmax(y_pred, axis=-1).flatten()

    # Decode the predictions
    decoded_predictions = [reverse_label_encoding[i] for i in y_pred_classes[:len(tokens)]]

    # Log the request and predictions
    log_entry = {
        'timestamp': datetime.now().isoformat(),
        'tokens': tokens,
        'predictions': decoded_predictions
    }
    logging.info(json.dumps(log_entry))

    return jsonify({
        'prediction': decoded_predictions
    })

if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)
