# train_catboost_safe.py
import warnings
warnings.filterwarnings("ignore")

import os, time
import joblib
import numpy as np
import pandas as pd
from scipy.stats import randint, uniform

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

# imblearn
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE, RandomOverSampler

# model
from catboost import CatBoostClassifier

# -------------------------
# User settings
# -------------------------
DATA_PATH = r"D:\exoplanet\combined_exoplanet_dataset.csv"  # <<-- set your path
OUT_DIR = r"D:\exoplanet"
os.makedirs(OUT_DIR, exist_ok=True)

RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 3
N_ITER = 60               # random search trials (increase to 120-200 for more search)
RANDOM_SEARCH_SCORING = 'f1_weighted'
# -------------------------

def load_and_engineer(path):
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    # required columns check
    required = ['orbital_period','transit_duration','planet_radius','transit_depth','stellar_radius','stellar_temp','label']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    # Fill numeric missing with median
    base = ['orbital_period','transit_duration','planet_radius','transit_depth','stellar_radius','stellar_temp']
    df[base] = df[base].fillna(df[base].median())

    # Engineering
    df['radius_ratio'] = df['planet_radius'] / (df['stellar_radius'] + 1e-8)
    df['depth_per_duration'] = df['transit_depth'] / (df['transit_duration'] + 1e-8)
    df['orbital_depth_interaction'] = df['orbital_period'] * df['transit_depth']
    for c in ['orbital_period','planet_radius','transit_depth']:
        df[f'{c}_log'] = np.log1p(np.maximum(df[c], 0))

    features = [
        'orbital_period','transit_duration','planet_radius','transit_depth',
        'stellar_radius','stellar_temp','radius_ratio','depth_per_duration',
        'orbital_depth_interaction','orbital_period_log','planet_radius_log','transit_depth_log'
    ]
    return df, features

def choose_sampler_for_smote(y_train):
    vc = pd.Series(y_train).value_counts()
    min_count = int(vc.min())
    if min_count <= 1:
        # SMOTE impossible; fall back to RandomOverSampler
        sampler = RandomOverSampler(random_state=RANDOM_STATE)
        sampler_name = f"RandomOverSampler(min_count={min_count})"
    else:
        k = min(5, max(1, min_count - 1))  # safe k_neighbors
        sampler = SMOTE(k_neighbors=k, random_state=RANDOM_STATE)
        sampler_name = f"SMOTE(k_neighbors={k})"
    return sampler, sampler_name

def safe_randomized_search(pipeline, param_dist, X_train, y_train, n_iter=N_ITER, scoring=RANDOM_SEARCH_SCORING):
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    try:
        rs = RandomizedSearchCV(
            estimator=pipeline,
            param_distributions=param_dist,
            n_iter=n_iter,
            scoring=scoring,
            cv=cv,
            n_jobs=-1,
            verbose=2,
            random_state=RANDOM_STATE,
            error_score='raise'
        )
        rs.fit(X_train, y_train)
    except Exception as e:
        # fallback to single-process (helps debug pickling/parallel errors)
        print("Parallel randomized search failed with error:", e)
        print("Retrying with n_jobs=1 (serial). This will be slower but more robust.")
        rs = RandomizedSearchCV(
            estimator=pipeline,
            param_distributions=param_dist,
            n_iter=n_iter,
            scoring=scoring,
            cv=cv,
            n_jobs=1,
            verbose=2,
            random_state=RANDOM_STATE,
            error_score='raise'
        )
        rs.fit(X_train, y_train)
    return rs

def strip_prefix_params(params, prefix='clf__'):
    """Strip 'clf__' from RandomizedSearchCV best_params keys to pass to estimator."""
    return {k[len(prefix):] if k.startswith(prefix) else k: v for k, v in params.items()}

def main():
    start_time = time.time()
    print("Loading and feature-engineering dataset...")
    df, feature_cols = load_and_engineer(DATA_PATH)
    print("Using features:", feature_cols)
    print("Full label distribution:\n", df['label'].value_counts())

    # Encode labels
    le = LabelEncoder()
    df['label_encoded'] = le.fit_transform(df['label'])
    print("Classes:", le.classes_)

    X = df[feature_cols].copy()
    y = df['label_encoded'].copy()

    # Train/test split (stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )
    print("Train shape:", X_train.shape, "Test shape:", X_test.shape)
    print("Train label distribution:", pd.Series(y_train).value_counts().to_dict())

    # choose sampler
    sampler, sampler_name = choose_sampler_for_smote(y_train)
    print("Sampler chosen:", sampler_name)

    # Build imblearn pipeline with CatBoost as final estimator
    cat_init = CatBoostClassifier(
        loss_function='MultiClass',
        verbose=0,
        random_state=RANDOM_STATE,
        # we will tune major params with RandomizedSearchCV
    )

    pipeline = ImbPipeline([
        ('sampler', sampler),
        ('scaler', StandardScaler()),
        ('clf', cat_init)
    ])

    # Param distribution (safe ranges)
    param_dist = {
        'clf__iterations': randint(300, 2000),
        'clf__depth': randint(4, 10),
        'clf__learning_rate': uniform(0.01, 0.2),
        'clf__l2_leaf_reg': uniform(1.0, 20.0),
        'clf__bagging_temperature': uniform(0.0, 1.0),
        'clf__random_strength': uniform(0.0, 20.0)
    }

    print("\nStarting RandomizedSearchCV (CatBoost)...")
    rs = safe_randomized_search(pipeline, param_dist, X_train, y_train, n_iter=N_ITER)

    print("\nBest CV score:", rs.best_score_)
    print("Best params:", rs.best_params_)

    # Prepare final CatBoost using best params, but refit with early stopping on a validation split
    best_params = strip_prefix_params(rs.best_params_, prefix='clf__')
    print("Stripped best params for CatBoost:", best_params)

    # Fit scaler on X_train and resample outside the pipeline for final early-stopping fit
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Resample
    if isinstance(sampler, SMOTE):
        X_res, y_res = sampler.fit_resample(X_train_scaled, y_train)
    else:
        X_res, y_res = sampler.fit_resample(X_train_scaled, y_train)  # RandomOverSampler or SMOTE

    # Make a small validation split from resampled data for early stopping
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_res, y_res, test_size=0.10, stratify=y_res, random_state=RANDOM_STATE
    )

    # Build final CatBoost with best params and early stopping
    final_cat_params = best_params.copy()
    # Ensure iterations is set sufficiently large; if not present, set a default
    final_iterations = int(final_cat_params.pop('iterations', 1000))
    final_cat = CatBoostClassifier(
        iterations=final_iterations,
        depth=int(final_cat_params.get('depth', 6)),
        learning_rate=float(final_cat_params.get('learning_rate', 0.05)),
        l2_leaf_reg=float(final_cat_params.get('l2_leaf_reg', 3.0)),
        bagging_temperature=float(final_cat_params.get('bagging_temperature', 0.0)),
        random_strength=float(final_cat_params.get('random_strength', 1.0)),
        loss_function='MultiClass',
        verbose=100,
        random_state=RANDOM_STATE
    )

    print("\nTraining final CatBoost with early stopping (uses a small validation split)...")
    final_cat.fit(
        X_tr, y_tr,
        eval_set=(X_val, y_val),
        early_stopping_rounds=75,
        use_best_model=True,
        verbose=100
    )

    # Evaluate on test
    print("\nEvaluating final model on test set...")
    y_pred = final_cat.predict(X_test_scaled)
    y_proba = final_cat.predict_proba(X_test_scaled)

    acc = accuracy_score(y_test, y_pred)
    wf1 = f1_score(y_test, y_pred, average='weighted')
    print(f"Test Accuracy: {acc:.4f}")
    print(f"Weighted F1-score: {wf1:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=le.classes_))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # Save predictions with probabilities
    out_df = X_test.reset_index(drop=True).copy()
    out_df['true_label'] = le.inverse_transform(y_test.reset_index(drop=True))
    out_df['pred_label'] = le.inverse_transform(y_pred.astype(int))
    for i, cls in enumerate(le.classes_):
        out_df[f'prob_{cls}'] = y_proba[:, i]
    out_csv = os.path.join(OUT_DIR, "catboost_single_model_predictions.csv")
    out_df.to_csv(out_csv, index=False)
    print("Saved predictions to:", out_csv)

    # Save the final model, scaler, label encoder and feature list
    artifacts = {
        'model': final_cat,
        'scaler': scaler,
        'label_encoder': le,
        'feature_columns': feature_cols,
        'sampler_name': sampler.__class__.__name__
    }
    joblib.dump(artifacts, os.path.join(OUT_DIR, "catboost_single_model_artifacts.joblib"))
    print("Saved model artifacts.")

    elapsed = time.time() - start_time
    print(f"\nTotal time: {elapsed/60:.2f} minutes")

if __name__ == "__main__":
    main()
