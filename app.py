import streamlit as st
import joblib
import pandas as pd
import os

ARTIFACT_DIR = 'artifacts'
MODEL_PATH = os.path.join(ARTIFACT_DIR, 'model.joblib')
VECT_PATH = os.path.join(ARTIFACT_DIR, 'vectorizer.joblib')

@st.cache_resource
def load_artifacts():
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECT_PATH)
    return model, vectorizer

st.set_page_config(page_title='Movie Review Sentiment', layout='centered')
st.title('🎬 Movie Review Sentiment Analyzer')
st.write('Model: TF-IDF features + Logistic Regression (trained locally).')

if not os.path.exists(MODEL_PATH) or not os.path.exists(VECT_PATH):
    st.error('Model artifacts not found in `artifacts/` — run the training script first:\n`python train_model.py --data IMDB_Dataset.csv --outdir artifacts`')
else:
    model, vectorizer = load_artifacts()

    st.header('Single review prediction')
    review_text = st.text_area('Paste a review', height=160)
    if st.button('Predict'):
        if not review_text.strip():
            st.info('Please paste a review first.')
        else:
            X = vectorizer.transform([review_text])
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(X)[0]
                pred = int(model.predict(X)[0])
                label = 'Positive' if pred == 1 else 'Negative'
                st.markdown(f'**Prediction:** {label}')
                st.markdown(f'**Confidence:** Positive={proba[1]:.3f}  |  Negative={proba[0]:.3f}')
            else:
                pred = int(model.predict(X)[0])
                label = 'Positive' if pred == 1 else 'Negative'
                st.markdown(f'**Prediction:** {label}')

    st.header('Batch prediction (CSV)')
    uploaded = st.file_uploader('Upload a CSV with a text column (e.g. review or text).', type=['csv'])
    if uploaded is not None:
        df = pd.read_csv(uploaded)
        # try to find a text column
        text_col = None
        for candidate in ['review', 'text', 'review_text', 'content']:
            if candidate in df.columns:
                text_col = candidate
                break
        if text_col is None:
            text_cols = [c for c in df.columns if df[c].dtype == object]
            if len(text_cols) > 0:
                text_col = text_cols[0]
        if text_col is None:
            st.error('Could not find a text column in the uploaded CSV.')
        else:
            st.write('Using column:', text_col)
            texts = df[text_col].fillna('').astype(str).tolist()
            X = vectorizer.transform(texts)
            preds = model.predict(X)
            probs = model.predict_proba(X)
            df['pred_label'] = ['Positive' if p==1 else 'Negative' for p in preds]
            df['prob_positive'] = [float(p[1]) for p in probs]
            st.dataframe(df.head())
            csv_bytes = df.to_csv(index=False).encode('utf-8')
            st.download_button('Download predictions (CSV)', csv_bytes, file_name='predictions.csv')

    st.sidebar.header('Tips & Next steps')
    st.sidebar.write('- If you want better accuracy, try a transformer (Hugging Face `transformers`) or fine-tune a pre-trained BERT variant.')
    st.sidebar.write('- You can also experiment with different vectorizer settings (max_features, ngram_range).')
    st.sidebar.write('- For explainability, try LIME or SHAP on individual predictions.')
