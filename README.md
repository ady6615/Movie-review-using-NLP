# Movie-review-using-NLP
mini project on Natural Language Programming subject

Project: Sentiment Analysis on Movie Reviews
#
# 1) Prepare dataset: get IMDB dataset (Kaggle "IMDB Dataset.csv" contains `review` and `sentiment` columns)
# - Save the CSV in the project folder or provide full path to --data when running training script.
#
# 2) Install dependencies:
# pip install pandas scikit-learn joblib nltk streamlit matplotlib seaborn
#
# 3) Train model and generate artifacts:
# python train_model.py --data IMDB_Dataset.csv --outdir artifacts
#
# 4) Launch Streamlit app:
# streamlit run app.py
#
# 5) Optional improvements (next steps):
# - Replace TF-IDF + LR with a fine-tuned transformer (Hugging Face `transformers`) for SOTA accuracy.
# - Add cross-validation & hyperparameter search (GridSearchCV).
# - Add model explainability (LIME/SHAP) and an admin UI to inspect false positives/negatives.