# Fake-Product-Review-Detection-using-NLP-Techniques

## Project Overview

This project is a **Fake Review Detection System** built using **Python** and **Machine Learning**.
It analyzes product reviews and predicts whether a review is **Fake** or **Genuine**.

The model uses:

* **TF-IDF Vectorization** for text feature extraction
* **Logistic Regression** for classification

---

# Technologies Used

* Python
* Pandas
* Scikit-learn

---

# Dataset Information

The dataset file used is:

```text
reviews.csv
```

Dataset columns:

* `review_id`
* `product_name`
* `review_text`
* `rating`
* `verified_purchase`
* `review_date`
* `fake_review`

Target column:

```text
fake_review
```

Feature column:

```text
review_text
```

---

# Installation

Install required libraries using:

```bash
pip install pandas scikit-learn
```

---

# How to Run

Run the Python file:

```bash
python fake.py
```

---

# Project Workflow

1. Load dataset using Pandas
2. Split dataset into training and testing sets
3. Convert text reviews into numerical vectors using TF-IDF
4. Train Logistic Regression model
5. Predict fake or genuine reviews
6. Evaluate model accuracy

---

# Sample Code

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

df = pd.read_csv('reviews.csv')

X = df['review_text']
y = df['fake_review']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

vectorizer = TfidfVectorizer(stop_words='english')

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

model = LogisticRegression()

model.fit(X_train_tfidf, y_train)

y_pred = model.predict(X_test_tfidf)

accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)
```

---

# Output

The system displays:

* Model Accuracy
* Classification Report
* Predictions for fake reviews

Example:

```text
Accuracy: 0.92
```

---

# Future Improvements

* Use Deep Learning models
* Add Flask Web Interface
* Deploy on Cloud
* Real-time review prediction

---

# Author

Muhammad Suffiyan Rafi
Computer Science Student & Python Developer
