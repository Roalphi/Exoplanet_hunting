# streamlit_exoplanet_classifier_full.py
# Usage:
#   pip install -r requirements.txt
#   streamlit run streamlit_exoplanet_classifier_full.py
#
# requirements.txt should include:
# streamlit
# pandas
# numpy
# scikit-learn
# imbalanced-learn
# matplotlib
# seaborn
# joblib
# xgboost    # optional, only required if you choose XGBoost

import os
import streamlit as st
import pandas as pd
import numpy as np
from collections import Counter
import traceback

from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix, recall_score
from sklearn.ensemble import RandomForestClassifier
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
warnings.filterwarnings("ignore")

# -------------------------
# Defensive XGBoost import & version check
# -------------------------
try:
    import xgboost as xgb_pkg
    from xgboost import XGBClassifier
    # require xgboost >= 1.3 for modern fit() signature; adjust if needed
    ver_parts = xgb_pkg.__version__.split('.')
    XGB_VERSION = tuple(int(x) for x in ver_parts[:2])
    if XGB_VERSION >= (1, 3):
        XGB_AVAILABLE = True
    else:
        XGB_AVAILABLE = False
        st.sidebar.warning(
            f"Detected xgboost {xgb_pkg.__version__} — it's old and may not support some XGBClassifier keywords. "
            "XGBoost option will be hidden. Consider upgrading: pip install -U xgboost"
        )
except Exception as e:
    XGB_AVAILABLE = False
    # Only show a small message in the sidebar if Streamlit is running
    try:
        st.sidebar.info("XGBoost import failed or not installed — XGBoost option will be hidden.")
        st.sidebar.write(f"Import error: {str(e)}")
    except Exception:
        pass

# -------------------------
# Page config and title
# -------------------------
st.set_page_config(page_title='Exoplanet Classifier', layout='wide')
st.title('🔭 Exoplanet Classifier — Full')

st.markdown(
    "Upload a labeled exoplanet dataset or use the example. Required columns: "
    "`orbital_period, transit_duration, planet_radius, transit_depth, stellar_radius, stellar_temp, label`"
)

# -------------------------
# Data upload / example
# -------------------------
st.sidebar.header('Data / Options')
uploaded_file = st.sidebar.file_uploader('Upload labeled CSV', type=['csv'])
use_example = st.sidebar.checkbox('Use example dataset', value=(uploaded_file is None))

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        st.sidebar.success('Loaded uploaded CSV')
    except Exception as e:
        st.sidebar.error(f'Could not read CSV: {e}')
        st.stop()
else:
    if use_example:
        rng = np.random.RandomState(42)
        n = 2000
        orbital_period = rng.exponential(scale=10, size=n) + rng.normal(0, 0.5, n)
        transit_duration = np.clip(0.1 + 0.02 * orbital_period + rng.normal(0,0.12,n), 0.02, None)
        planet_radius = np.clip(rng.normal(1.8, 1.0, n), 0.05, 20)
        transit_depth = np.abs((planet_radius**2) * 0.01 + rng.normal(0, 0.0007, n))
        stellar_radius = np.clip(rng.normal(1.0, 0.25, n), 0.1, 3)
        stellar_temp = np.clip(rng.normal(5600, 450, n), 3000, 8000)

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
    else:
        st.sidebar.info('Upload CSV or enable example dataset.')
        st.stop()

# -------------------------
# Quick preview and validation
# -------------------------
st.subheader('Dataset preview')
st.write(df.head())

required_cols = {'orbital_period', 'transit_duration', 'planet_radius', 'transit_depth', 'stellar_radius', 'stellar_temp', 'label'}
missing = required_cols - set(df.columns)
if missing:
    st.error(f"Missing required columns: {sorted(list(missing))}")
    st.stop()

# -------------------------
# Feature selection
# -------------------------
all_features = [c for c in df.columns if c != 'label']
st.sidebar.header('Features')
selected_features = st.sidebar.multiselect('Select features for training', all_features,
                                           default=['orbital_period','transit_duration','planet_radius','transit_depth','stellar_radius','stellar_temp'])
if len(selected_features) == 0:
    st.error('Select at least one feature.')
    st.stop()

# -------------------------
# Engineered features
# -------------------------
def add_engineered_features(df_in):
    df = df_in.copy()
    # safe clip for orbital period and denominators
    if 'orbital_period' in df.columns:
        df['orbital_period_clipped'] = df['orbital_period'].clip(lower=1e-6)
        df['log_orbital_period'] = np.log1p(df['orbital_period_clipped'])
    if 'planet_radius' in df.columns and 'stellar_radius' in df.columns:
        df['planet_to_star_radius_ratio'] = df['planet_radius'] / (df['stellar_radius'] + 1e-9)
    if 'transit_depth' in df.columns and 'planet_radius' in df.columns:
        df['transit_depth_ratio'] = df['transit_depth'] / (df['planet_radius'] + 1e-9)
    if 'stellar_temp' in df.columns and 'stellar_radius' in df.columns:
        df['temp_ratio'] = df['stellar_temp'] / (df['stellar_radius'] + 1e-9)
    return df

df = add_engineered_features(df)
engineered_candidates = ['log_orbital_period','planet_to_star_radius_ratio','transit_depth_ratio','temp_ratio']
available_engineered = [c for c in engineered_candidates if c in df.columns]

# expand selected features by engineered ones (checkbox)
st.sidebar.header('Engineered features')
use_engineered = st.sidebar.checkbox('Include engineered features (log, ratios)', value=True)
if use_engineered:
    for feat in available_engineered:
        if feat not in selected_features:
            selected_features.append(feat)

# -------------------------
# Label encoding (dynamic)
# -------------------------
df['label'] = df['label'].astype(str)
unique_labels = sorted(df['label'].unique())
label_mapping = {lab: i for i, lab in enumerate(unique_labels)}
inv_label = {v: k for k, v in label_mapping.items()}
df['label_num'] = df['label'].map(label_mapping)

st.sidebar.write('Detected labels:')
for lab in unique_labels:
    st.sidebar.write(f"- {lab}")

# -------------------------
# Training parameters
# -------------------------
st.sidebar.header('Training parameters')
test_size = st.sidebar.slider('Test set fraction', 0.05, 0.5, 0.2, step=0.05)
seed = int(st.sidebar.number_input('Random seed', value=42, step=1))
classifier_choice = st.sidebar.selectbox('Classifier', ['RandomForest'] + (['XGBoost'] if XGB_AVAILABLE else []))

# Hyperparameter options
st.sidebar.subheader('Hyperparameter search')
n_iter = st.sidebar.slider('RandomizedSearchCV iterations', 5, 50, 20, step=1)
cv_folds = int(st.sidebar.slider('CV folds', 2, 5, 3))

# -------------------------
# Prepare X, y
# -------------------------
X = df[selected_features].copy()
y = df['label_num'].copy()

# Fill NA with median for display and safe transforms later in pipeline
X_display = X.fillna(X.median())

st.subheader('Dataset info')
st.write('Number of examples:', len(df))
st.write('Class distribution:')
st.write(df['label'].value_counts())

# -------------------------
# Train / Test split
# -------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify=y, random_state=seed)

# -------------------------
# Build pipeline with SMOTE (applied only on training folds inside pipeline)
# -------------------------
# adjust k_neighbors to minority class
min_class_count = int(y_train.value_counts().min())
k_neighbors = 3
if min_class_count <= k_neighbors:
    k_neighbors = max(1, min_class_count - 1)

smote = SMOTE(random_state=seed, k_neighbors=k_neighbors)
imputer = SimpleImputer(strategy='median')
scaler = StandardScaler()

if classifier_choice == 'RandomForest':
    base_clf = RandomForestClassifier(random_state=seed, n_jobs=-1)
    param_dist = {
        'clf__n_estimators': [100, 200, 400],
        'clf__max_depth': [None, 8, 16],
        'clf__min_samples_split': [2, 5, 10],
        'clf__min_samples_leaf': [1, 2, 4],
        'clf__class_weight': [None, 'balanced']
    }
else:
    # XGBoost chosen (CPU-safe config)
    if not XGB_AVAILABLE:
        st.error("XGBoost is not installed or incompatible in this environment. Install/upgrade xgboost to use it.")
        st.stop()
    base_clf = XGBClassifier(
        objective='multi:softprob',
        use_label_encoder=False,
        tree_method='hist',      # CPU-safe
        predictor='auto',
        random_state=seed,
        n_jobs=-1,
        verbosity=0
    )
    param_dist = {
        'clf__n_estimators': [100, 200, 400],
        'clf__max_depth': [4, 6, 8],
        'clf__learning_rate': [0.01, 0.03, 0.05],
        'clf__subsample': [0.7, 0.8, 0.9],
        'clf__colsample_bytree': [0.6, 0.7, 0.8],
        'clf__reg_lambda': [0.5, 1.0, 1.5]
    }

pipe = ImbPipeline([
    ('imputer', imputer),
    ('scaler', scaler),
    ('smote', smote),
    ('clf', base_clf)
])

# -------------------------
# Hyperparameter search with StratifiedKFold (defensive)
# -------------------------
st.write("Starting RandomizedSearchCV training. This may take time depending on settings.")
cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)

# Safer default: on Windows or problematic environments use n_jobs=1 for RandomizedSearchCV
search_n_jobs = 1
search = RandomizedSearchCV(
    estimator=pipe,
    param_distributions=param_dist,
    n_iter=n_iter,
    scoring='f1_macro',
    cv=cv,
    random_state=seed,
    n_jobs=search_n_jobs,
    verbose=1,
    refit=True,
    error_score='raise'
)

with st.spinner('Running hyperparameter search...'):
    try:
        search.fit(X_train, y_train)
    except Exception as e:
        tb = traceback.format_exc()
        st.error("Hyperparameter search FAILED. Full traceback (copy this and paste if you want me to inspect):")
        st.text(tb)
        st.info("Falling back to a safe RandomForest search (short).")

        # fallback pipeline with RF only (short search)
        try:
            base_clf = RandomForestClassifier(random_state=seed, n_jobs=-1)
            pipe = ImbPipeline([('imputer', imputer), ('scaler', scaler), ('smote', smote), ('clf', base_clf)])
            rf_param_dist = {
                'clf__n_estimators': [100, 200],
                'clf__max_depth': [None, 10],
            }
            fallback = RandomizedSearchCV(pipe, rf_param_dist, n_iter=4, cv=cv, scoring='f1_macro',
                                          n_jobs=1, random_state=seed, refit=True, error_score='raise')
            fallback.fit(X_train, y_train)
            best_pipe = fallback.best_estimator_
            st.success("Fallback RandomForest training complete")
            st.write('Fallback best params:', fallback.best_params_)
        except Exception as e2:
            st.error("Fallback also failed. See traceback below:")
            st.text(traceback.format_exc())
            st.stop()
    else:
        best_pipe = search.best_estimator_
        st.success('Training complete')
        st.write('Best params:', search.best_params_)

# -------------------------
# Evaluation on test set (safe)
# -------------------------
y_pred = best_pipe.predict(X_test)

# determine labels actually present in test or predicted to avoid mismatch
labels_present = np.unique(np.concatenate([y_test.values, np.array(y_pred)]))
labels_present = np.sort(labels_present).astype(int)
target_names_present = [inv_label[int(i)] for i in labels_present]

acc = accuracy_score(y_test, y_pred)
f1_macro = f1_score(y_test, y_pred, average='macro')
recalls = recall_score(y_test, y_pred, average=None, labels=labels_present, zero_division=0)

report = classification_report(
    y_test,
    y_pred,
    labels=labels_present,
    target_names=target_names_present,
    zero_division=0
)

st.subheader('Model performance (test set)')
st.metric('Accuracy', float(acc))
st.metric('Macro F1', float(f1_macro))
st.write('Per-class recall:')
st.write({target_names_present[i]: float(recalls[i]) for i in range(len(recalls))})
st.text('Classification report:')
st.text(report)

# Confusion matrix plot — ensure tick labels match cm shape
cm = confusion_matrix(y_test, y_pred, labels=labels_present)
fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d',
            xticklabels=target_names_present,
            yticklabels=target_names_present,
            cmap='Blues', ax=ax)
ax.set_xlabel('Predicted')
ax.set_ylabel('True')
st.pyplot(fig)

# -------------------------
# Feature importance (if supported)
# -------------------------
st.subheader('Feature importances')
try:
    clf_in_pipe = best_pipe.named_steps['clf']
    if hasattr(clf_in_pipe, 'feature_importances_'):
        importances = clf_in_pipe.feature_importances_
        feat_imp = pd.Series(importances, index=selected_features).sort_values(ascending=False)
        st.bar_chart(feat_imp)
    else:
        st.info('Classifier does not expose feature_importances_.')
except Exception as e:
    st.info('Could not compute feature importances: ' + str(e))

# -------------------------
# Predict single or batch
# -------------------------
st.subheader('Make predictions')

with st.expander('Single prediction'):
    manual = {}
    cols = st.columns(3)
    for i, f in enumerate(selected_features):
        default_val = float(X_display[f].median()) if f in X_display.columns else 0.0
        with cols[i % 3]:
            manual[f] = st.number_input(f, value=default_val, format="%.6f")
    if st.button('Predict single'):
        single_df = pd.DataFrame([manual])
        single_df = add_engineered_features(single_df)
        single_df = single_df[selected_features]
        try:
            pred = best_pipe.predict(single_df)[0]
            proba = best_pipe.predict_proba(single_df)[0] if hasattr(best_pipe, "predict_proba") else None
            st.success(f'Predicted: {inv_label[int(pred)]}')
            if proba is not None:
                st.write({inv_label[i]: float(proba[i]) for i in range(len(proba))})
        except Exception as e:
            st.error(f'Prediction failed: {e}')

with st.expander('Batch prediction'):
    batch_file = st.file_uploader('Upload CSV for batch prediction', type=['csv'], key='batch')
    if batch_file is not None:
        try:
            batch_df = pd.read_csv(batch_file)
            batch_df = add_engineered_features(batch_df)
            missing_cols = [c for c in selected_features if c not in batch_df.columns]
            if missing_cols:
                st.error(f'Missing columns in batch file: {missing_cols}')
            else:
                X_batch = batch_df[selected_features]
                preds = best_pipe.predict(X_batch)
                probs = best_pipe.predict_proba(X_batch) if hasattr(best_pipe, "predict_proba") else None
                batch_df['pred_label'] = [inv_label[int(p)] for p in preds]
                if probs is not None:
                    for i in range(len(inv_label)):
                        batch_df[f'prob_{inv_label[i]}'] = probs[:, i]
                st.write(batch_df.head())
                csv_bytes = batch_df.to_csv(index=False).encode('utf-8')
                st.download_button('Download predictions CSV', csv_bytes, file_name='predictions.csv')
        except Exception as e:
            st.error(f'Batch prediction failed: {e}')

# -------------------------
# Save model
# -------------------------
st.sidebar.header('Model persistence')
model_save_path = os.path.join(os.getcwd(), "exoplanet_model_artifact.pkl")
if st.sidebar.button('Save trained model'):
    try:
        joblib.dump({'pipeline': best_pipe, 'label_mapping': label_mapping, 'selected_features': selected_features}, model_save_path)
        st.sidebar.success(f'Model saved at {model_save_path}')
    except Exception as e:
        st.sidebar.error(f'Could not save model: {e}')

st.sidebar.markdown('---')
st.sidebar.write('Best params preview:')
try:
    st.sidebar.write(search.best_params_)
except Exception:
    # if fallback used, search may not exist
    try:
        st.sidebar.write(fallback.best_params_)
    except Exception:
        st.sidebar.write("No best params available.")
