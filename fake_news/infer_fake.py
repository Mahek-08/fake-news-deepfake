# fake_news/infer_fake.py
import joblib
from utils.preprocessing import clean_text

vec = joblib.load('fake_news/vectorizer.pkl')
clf = joblib.load('fake_news/fake_news_model.pkl')


def predict_text(text: str):
    text_clean = clean_text(text)
    X = vec.transform([text_clean])
    pred = clf.predict(X)[0]
    prob = clf.predict_proba(X).max()
    return {'label': str(pred), 'score': float(prob)}

if __name__ == '__main__':
    sample = "President signs a new law to..."
    print(predict_text(sample))
