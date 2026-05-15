# import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# load the dataset
df = pd.read_csv('reviews.csv')

# display first few rows
print("Dataset Sample:\n", df.head())

# define features and target
X = df['review_text']
y = df['verified_purchase']

# split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# initialize TF-IDF vectorizer
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)

# fit on training data
X_train_tfidf = vectorizer.fit_transform(X_train)

# transform test data
X_test_tfidf = vectorizer.transform(X_test)

# initialize logistic regression model
model = LogisticRegression()

# train model
model.fit(X_train_tfidf, y_train)

# make predictions
y_pred = model.predict(X_test_tfidf)

# evaluate model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f"\nAccuracy: {accuracy}")

print("\nClassification Report:\n", report)