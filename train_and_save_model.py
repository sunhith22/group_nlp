import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import numpy as np
import pickle

# Load the dataset
df = pd.read_parquet('train-00000-of-00001.parquet')
df_test = pd.read_parquet('test-00000-of-00001.parquet')

# Define label encoding
label_encoding = {"B-O": 0, "B-AC": 1, "B-LF": 2, "I-LF": 3}

# Convert the ner_tags to numerical labels
train_label = [[label_encoding[tag] for tag in sample] for sample in df["ner_tags"]]
test_label = [[label_encoding[tag] for tag in sample] for sample in df_test["ner_tags"]]

# Extract text data for TF-IDF encoding
X_train = [' '.join(tokens) for tokens in df['tokens']]
X_test = [' '.join(tokens) for tokens in df_test['tokens']]

# TF-IDF encoding
tfidf_vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

# Prepare sequences for RNN
X_train_tfidf_seq = [tfidf_vectorizer.transform(tokens).toarray().reshape(-1, 1, 5000) for tokens in df['tokens']]
X_test_tfidf_seq = [tfidf_vectorizer.transform(tokens).toarray().reshape(-1, 1, 5000) for tokens in df_test['tokens']]

# Flatten the sequences and corresponding labels
X_train_tfidf_seq_flat = np.vstack(X_train_tfidf_seq)
y_train_flat = np.concatenate(train_label)

# Define the RNN model
model = Sequential([
    SimpleRNN(128, input_shape=(None, 5000), activation='relu'),
    Dense(256, activation='relu'),
    Dense(len(label_encoding), activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train_tfidf_seq_flat, y_train_flat, epochs=10, batch_size=32, verbose=1)

# Flatten the test sequences and corresponding labels
X_test_tfidf_seq_flat = np.vstack(X_test_tfidf_seq)
y_test_flat = np.concatenate(test_label)

# Evaluate the model on the test set
y_pred = model.predict(X_test_tfidf_seq_flat)
y_pred_classes = np.argmax(y_pred, axis=1)

# Calculate and print confusion matrix
conf_matrix = confusion_matrix(y_test_flat, y_pred_classes)
print("Confusion Matrix:")
print(conf_matrix)

# Print classification report
print(classification_report(y_test_flat, y_pred_classes, target_names=label_encoding.keys()))

# Calculate and print accuracy
accuracy = accuracy_score(y_test_flat, y_pred_classes)
print("Accuracy:", accuracy)

# Save the model
model.save('rnn_model.h5')

# Save the TF-IDF vectorizer
with open('tfidf_vectorizer.pkl', 'wb') as f:
    pickle.dump(tfidf_vectorizer, f)

# Save the label encoding
with open('label_encoding.pkl', 'wb') as f:
    pickle.dump(label_encoding, f)
