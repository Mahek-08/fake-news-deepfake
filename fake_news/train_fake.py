import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import joblib
from utils.preprocessing import clean_text



# 1) Load data
df = pd.read_csv('data/fake_news/train.csv')  # expects columns: text,label

# If dataset has separate title and text, combine them
if 'title' in df.columns:
    df['text'] = df['title'].fillna('') + ' ' + df['text'].fillna('')

# 2) Preprocess
df['text_clean'] = df['text'].astype(str).apply(clean_text)

# 3) Train/test split
X_train, X_test, y_train, y_test = train_test_split(df['text_clean'], df['label'], test_size=0.15, random_state=42, stratify=df['label'])

# 4) Vectorize
vec = TfidfVectorizer(max_features=30000, ngram_range=(1,2))
X_train_t = vec.fit_transform(X_train)
X_test_t = vec.transform(X_test)

# 5) Train a classifier
clf = LogisticRegression(max_iter=1000)
clf.fit(X_train_t, y_train)

# 6) Evaluate
preds = clf.predict(X_test_t)
print(classification_report(y_test, preds))

# 7) Save artifacts
joblib.dump(vec, 'fake_news/vectorizer.pkl')
joblib.dump(clf, 'fake_news/fake_news_model.pkl')
print('Saved model and vectorizer.')
