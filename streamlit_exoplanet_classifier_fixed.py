# streamlit_exoplanet_classifier_fixed.py
# Usage: pip install -r requirements.txt
# Run: streamlit run streamlit_exoplanet_classifier_fixed.py

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

st.set_page_config(page_title='Exoplanet Classifier', layout='wide')
st.title('🔭 Exoplanet Classifier — Prototype')

st.markdown(
    """Upload a labeled exoplanet dataset (or use the example).
Columns commonly used:
- `orbital_period`, `transit_duration`, `planet_radius`, `transit_depth`, `stellar_radius`, `stellar_temp`
Labels: `confirmed`, `candidate`, `false_positive`
"""
)

# --- Sidebar: Dataset upload or example ---
st.sidebar.header('Data / Training')
uploaded_file = st.sidebar.file_uploader('cleaned_exoplanet_dataset.csv', type=['csv'])
use_example = st.sidebar.checkbox('Use example dataset', value=True if uploaded_file is None else False)

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.sidebar.success('Loaded uploaded CSV')
else:
    if use_example:
        # Synthetic dataset
        rng = np.random.RandomState(42)
        n = 400
        orbital_period = rng.exponential(scale=10, size=n) + rng.normal(0, 0.5, n)
        transit_duration = np.clip(0.1 + 0.02 * orbital_period + rng.normal(0,0.1,n), 0.02, None)
        planet_radius = np.clip(rng.normal(1.5, 0.7, n), 0.1, 15)
        transit_depth = (planet_radius**2) * 0.01 + rng.normal(0, 0.0005, n)
        stellar_radius = np.clip(rng.normal(1.0, 0.2, n), 0.1, 3)
        stellar_temp = np.clip(rng.normal(5600, 400, n), 3000, 8000)

        prob_planet = (planet_radius < 5) & (transit_depth > 0.0002) & (transit_duration > 0.03)
        labels = []
        for p in prob_planet:
            if rng.rand() < 0.02:
                labels.append('false_positive')
            else:
                labels.append('confirmed' if p else rng.choice(['candidate','false_positive'], p=[0.6,0.4]))

        df = pd.DataFrame({
            'orbital_period': orbital_period,
            'transit_duration': transit_duration,
            'planet_radius': planet_radius,
            'transit_depth': transit_depth,
            'stellar_radius': stellar_radius,
            'stellar_temp': stellar_temp,
            'label': labels
        })
        st.sidebar.success('Using synthetic example dataset')

st.subheader('Dataset preview')
st.write(df.head())

# Validate dataset
if 'label' not in df.columns:
    st.error('Dataset must have a `label` column.')
    st.stop()

# --- Sidebar: Features ---
all_features = [c for c in df.columns if c != 'label']
st.sidebar.header('Features')
selected_features = st.sidebar.multiselect('Select features for training', all_features, default=all_features)
if len(selected_features) == 0:
    st.error('Select at least one feature.')
    st.stop()

# Encode labels
label_mapping = {'confirmed':0, 'candidate':1, 'false_positive':2}
inv_label = {v:k for k,v in label_mapping.items()}
df['label_num'] = df['label'].map(label_mapping)

# Impute missing numeric values
X = df[selected_features].copy()
y = df['label_num']

imputer = SimpleImputer(strategy='median')
X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=selected_features)

# Train/test split
st.sidebar.header('Training parameters')
test_size = st.sidebar.slider('Test set fraction', 0.05, 0.5, 0.2, step=0.05)
n_estimators = st.sidebar.slider('RandomForest n_estimators', 10, 500, 100, step=10)
max_depth = st.sidebar.slider('RandomForest max_depth (0=None)', 0, 50, 0, step=1)
if max_depth == 0:
    max_depth = None
random_state = 42

X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=test_size, stratify=y, random_state=random_state)

# Pipeline with scaling + classifier
scaler = StandardScaler()
clf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=random_state)
pipe = Pipeline([('scaler', scaler), ('clf', clf)])
pipe.fit(X_train, y_train)

# --- Evaluation ---
y_pred = pipe.predict(X_test)
acc = accuracy_score(y_test, y_pred)
st.subheader('Model evaluation')
st.metric('Accuracy (test)', f'{acc:.3f}')

st.text('Classification report:')
report = classification_report(y_test, y_pred, target_names=[inv_label[i] for i in range(len(inv_label))])
st.text(report)

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=[inv_label[i] for i in range(len(inv_label))], 
            yticklabels=[inv_label[i] for i in range(len(inv_label))], ax=ax)
ax.set_xlabel('Predicted')
ax.set_ylabel('True')
st.pyplot(fig)

# Feature importances
st.subheader('Feature importances')
importances = pipe.named_steps['clf'].feature_importances_
feat_imp = pd.Series(importances, index=selected_features).sort_values(ascending=False)
st.bar_chart(feat_imp)

# --- Predictions ---
st.subheader('Make predictions')
st.markdown('Manual input or upload CSV for batch prediction.')

# Single prediction
with st.expander('Single prediction'):
    manual_vals = {}
    cols = st.columns(3)
    for i, f in enumerate(selected_features):
        with cols[i % 3]:
            manual_vals[f] = st.number_input(f, value=float(X[f].median()))
    if st.button('Predict single'):
        single_df = pd.DataFrame([manual_vals])
        pred = pipe.predict(single_df)[0]
        proba = pipe.predict_proba(single_df)[0]
        st.success(f'Predicted: {inv_label[pred]}')
        st.write({inv_label[i]: float(proba[i]) for i in range(len(proba))})

# Batch prediction
with st.expander('Batch prediction'):
    batch_file = st.file_uploader('Upload CSV for batch', type=['csv'], key='batch')
    if batch_file is not None:
        batch_df = pd.read_csv(batch_file)
        missing = [c for c in selected_features if c not in batch_df.columns]
        if missing:
            st.error(f'Missing columns: {missing}')
        else:
            X_batch = pd.DataFrame(imputer.transform(batch_df[selected_features]), columns=selected_features)
            preds = pipe.predict(X_batch)
            probs = pipe.predict_proba(X_batch)
            batch_df['pred_label'] = [inv_label[p] for p in preds]
            for i in range(len(inv_label)):
                batch_df[f'prob_{inv_label[i]}'] = probs[:,i]
            st.write(batch_df.head())
            st.download_button('Download predictions CSV', batch_df.to_csv(index=False).encode('utf-8'), file_name='predictions.csv')

# --- Save model ---
st.sidebar.header('Model persistence')
if st.sidebar.button('Save trained model'):
    joblib.dump(pipe, "exoplanet_rf_model.pkl")
    st.sidebar.success('Model saved as exoplanet_rf_model.pkl')

st.sidebar.markdown('---')
st.subheader('Dataset info')
st.write('Number of examples:', len(df))
st.write('Class distribution:')
st.write(df['label'].value_counts())
import os
save_path = os.path.join(os.getcwd(), "exoplanet_rf_model.pkl")
joblib.dump(pipe, save_path)
st.sidebar.success(f'Model saved at {save_path}')
st.subheader('Model evaluation')
st.metric('Accuracy (test)', f'{acc:.3f}')
