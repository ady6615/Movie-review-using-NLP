import os
import re
import argparse
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# NLTK for light preprocessing
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


import nltk

try:
    nltk.data.find('tokenizers/punkt')
except Exception:
    nltk.download('punkt')
try:
    nltk.data.find('tokenizers/punkt_tab/english')
except Exception:
    # punkt_tab is required on some NLTK installs
    nltk.download('punkt_tab')
try:
    nltk.data.find('corpora/stopwords')
except Exception:
    nltk.download('stopwords')
try:
    nltk.data.find('corpora/wordnet')
except Exception:
    nltk.download('wordnet')
# Try to download required NLTK data (harmless if already present)
'''try:
    nltk.data.find('tokenizers/punkt')
except Exception:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/stopwords')
except Exception:
    nltk.download('stopwords')
try:
    nltk.data.find('corpora/wordnet')
except Exception:
    nltk.download('wordnet')
'''
STOPWORDS = set(stopwords.words('english'))
LEMMATIZER = WordNetLemmatizer()


def clean_text(text: str) -> str:
    """Minimal cleaning: remove HTML, non-letters, lower, tokenize, remove stopwords, lemmatize."""
    text = str(text)
    text = re.sub(r'<.*?>', ' ', text)                # strip HTML
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)        # keep letters + spaces
    text = text.lower()
    tokens = nltk.word_tokenize(text)
    tokens = [LEMMATIZER.lemmatize(t) for t in tokens if (t not in STOPWORDS and len(t) > 1)]
    return ' '.join(tokens)


def load_dataset(path: str) -> pd.DataFrame:
    """Load a CSV and normalize to DataFrame(columns=['text','label'])."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found: {path}")
    df = pd.read_csv(path)
    # common layouts
    if {'review', 'sentiment'}.issubset(df.columns):
        out = df[['review', 'sentiment']].rename(columns={'review':'text', 'sentiment':'label'})
        return out
    if {'text', 'label'}.issubset(df.columns):
        return df[['text','label']]
    # fallback: pick first object column as text, and a small-cardinality column as label
    text_col = None
    for c in df.columns:
        if df[c].dtype == object:
            text_col = c
            break
    label_col = None
    for c in df.columns:
        if c != text_col and df[c].nunique() <= 5:
            label_col = c
            break
    if text_col is None or label_col is None:
        raise ValueError('Could not auto-detect text/label columns. Use a CSV with review/sentiment or text/label columns.')
    return df[[text_col, label_col]].rename(columns={text_col:'text', label_col:'label'})


def normalize_labels(y: pd.Series) -> pd.Series:
    """Convert a variety of label formats to binary 0/1 (0=negative, 1=positive)."""
    if y.dtype == object:
        y_low = y.str.lower().str.strip()
        mapping = {'positive':1, 'negative':0, 'pos':1, 'neg':0}
        if set(y_low.unique()).issubset(set(mapping.keys())):
            return y_low.map(mapping).astype(int)
        # sometimes labels are '1'/'0' strings
        try:
            return y.astype(int)
        except Exception:
            raise ValueError('Unrecognized label values. Expected positive/negative or numeric 0/1.')
    else:
        return y.astype(int)


def train_pipeline(args):
    print('Loading dataset...')
    df = load_dataset(args.data)
    print('Raw dataset shape:', df.shape)
    df['text'] = df['text'].fillna('').astype(str)
    print('Cleaning text (this may take a few minutes on large datasets)...')
    df['text_clean'] = df['text'].apply(clean_text)
    df = df[df['text_clean'].str.strip()!='']
    df['label'] = normalize_labels(df['label'])

    X = df['text_clean']
    y = df['label']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=42, stratify=y)

    print('Vectorizing text with TF-IDF...')
    vectorizer = TfidfVectorizer(max_features=args.max_features, ngram_range=(1,2), sublinear_tf=True)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    print('Training LogisticRegression...')
    model = LogisticRegression(max_iter=2000)
    model.fit(X_train_tfidf, y_train)

    print('Evaluating on test set...')
    y_pred = model.predict(X_test_tfidf)
    acc = accuracy_score(y_test, y_pred)
    print(f'Test accuracy: {acc:.4f}')
    print(classification_report(y_test, y_pred, digits=4))

    cm = confusion_matrix(y_test, y_pred)
    os.makedirs(args.outdir, exist_ok=True)
    # save confusion matrix
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['neg','pos'], yticklabels=['neg','pos'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    cm_path = os.path.join(args.outdir, 'confusion_matrix.png')
    plt.savefig(cm_path)
    plt.close()

    # save artifacts
    model_path = os.path.join(args.outdir, 'model.joblib')
    vec_path = os.path.join(args.outdir, 'vectorizer.joblib')
    joblib.dump(model, model_path)
    joblib.dump(vectorizer, vec_path)
    print('Saved model to', model_path)
    print('Saved vectorizer to', vec_path)
    # save a small CSV with cleaned text for reference
    sample_path = os.path.join(args.outdir, 'cleaned_sample.csv')
    df[['text','text_clean','label']].sample(n=min(1000, len(df))).to_csv(sample_path, index=False)
    print('Saved cleaned sample to', sample_path)
    print('Artifacts saved to', args.outdir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='IMDB_Dataset.csv', help='Path to CSV dataset')
    parser.add_argument('--outdir', type=str, default='artifacts', help='Where to save model artifacts')
    parser.add_argument('--max_features', type=int, default=10000)
    parser.add_argument('--test_size', type=float, default=0.2)
    args = parser.parse_args()
    train_pipeline(args)