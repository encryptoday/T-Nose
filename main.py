import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load dataset
data = pd.read_csv('dataset/spamhamdata.csv', sep='\t', header=None, names=['Label', 'Message'])

# Preprocess the dataset
data['Label'] = data['Label'].map({'ham': 0, 'spam': 1})  # Encode labels as 0 (ham) and 1 (spam)

# Extract features and labels
X = data['Message']
y = data['Label']

# Convert text to TF-IDF features
vectorizer = TfidfVectorizer(stop_words='english', max_features=3000)
X_tfidf = vectorizer.fit_transform(X)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)

# Train the Logistic Regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Make predictions and calculate confidence levels
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]  # Probability of being spam

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=['Ham', 'Spam'])

# Output results
print(f"Accuracy: {accuracy * 100:.2f}%")
print("Confusion Matrix:")
print(conf_matrix)
print("Classification Report:")
print(report)

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Ham', 'Spam'], yticklabels=['Ham', 'Spam'])
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Plot confidence levels
plt.figure(figsize=(10, 6))
plt.hist(y_pred_proba, bins=30, color='purple', alpha=0.7)
plt.title("Distribution of Spam Confidence Levels")
plt.xlabel("Spam Confidence")
plt.ylabel("Frequency")
plt.show()

# Plot Predicted vs Actual results
plt.figure(figsize=(10, 6))
plt.scatter(range(len(y_test)), y_test, color='blue', label='Actual', alpha=0.7)
plt.scatter(range(len(y_pred)), y_pred, color='red', label='Predicted', alpha=0.7)
plt.title('Predicted vs Actual Results')
plt.xlabel('Index')
plt.ylabel('Label (0 = Ham, 1 = Spam)')
plt.legend()
plt.show()

# User input for predictions
while True:
    user_input = input("\nEnter a message to check if it's spam (or type 'exit' to quit): ")
    if user_input.lower() == 'exit':
        break
    user_features = vectorizer.transform([user_input])
    user_proba = model.predict_proba(user_features)[:, 1][0]
    user_prediction = "Spam" if user_proba > 0.5 else "Ham"
    print(f"Message: '{user_input}' | Prediction: {user_prediction} | Spam Confidence: {user_proba * 100:.2f}%")
