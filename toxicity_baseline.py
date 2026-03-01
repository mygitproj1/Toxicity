import pandas as pd
import re
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import multilabel_confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import classification_report

print("1. Loading the local train.csv dataset...")
# We only load train.csv because Kaggle's test.csv is missing the target labels!
df = pd.read_csv("data/train.csv").sample(10000, random_state=42)

# The 6 specific toxicity categories
toxicity_labels = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

print("2. Preprocessing and cleaning the text...")
def clean_text(text):
    """Basic text cleaning: lowercase, remove special characters and extra spaces."""
    text = str(text).lower()
    text = re.sub(r"[^a-zA-Z0-9 ]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# Apply the cleaning function
df['clean_text'] = df['comment_text'].apply(clean_text)

print("3. Splitting data into Train and Test sets...")
# Separate our text features (X) and our target answers (y)
X = df['clean_text']
y = df[toxicity_labels]

# This automatically shuffles the data and splits it 80% for training, 20% for testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("4. Vectorizing text (Converting words to numbers)...")
vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

print("5. Training the multi-label Machine Learning model...")
base_model = LogisticRegression(solver='liblinear')
model = OneVsRestClassifier(base_model)
model.fit(X_train_vec, y_train)

print("6. Evaluating accuracy...")
predictions = model.predict(X_test_vec)
print("\n--- Model Classification Report ---")
print(classification_report(y_test, predictions, target_names=toxicity_labels, zero_division=0))

print("\n7. Saving outputs for the UI...")
# Recombine the test data with our predictions so we can save it
results_df = pd.DataFrame({'clean_text': X_test})
for i, label in enumerate(toxicity_labels):
    results_df[f'actual_{label}'] = y_test[label].values
    results_df[f'pred_{label}'] = predictions[:, i]

results_df.to_csv("toxicity_model_results.csv", index=False)
joblib.dump(model, "toxicity_model.joblib")
joblib.dump(vectorizer, "tfidf_vectorizer.joblib")

print("Pipeline Complete! Models saved successfully.")

print("\n--- Generating Confusion Matrices ---")
# Generate the 6 separate matrices
matrices = multilabel_confusion_matrix(y_test, predictions)

# Set up a visual grid (2 rows, 3 columns) to display them all at once
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
axes = axes.ravel() # Flattens the grid array for easier looping

for i, label in enumerate(toxicity_labels):
    # Draw a heatmap for each individual matrix
    sns.heatmap(matrices[i], annot=True, fmt='d', cmap='Blues', ax=axes[i], 
                xticklabels=['Safe', 'Flagged'], 
                yticklabels=['Safe', 'Flagged'])
    
    axes[i].set_title(f'Category: {label.upper()}')
    axes[i].set_ylabel('Actual Human Label')
    axes[i].set_xlabel('Model Prediction')

plt.tight_layout()
plt.show() # This pops open the window with your charts!