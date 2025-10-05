import pandas as pd

# ----------------- Functions -----------------
def clean_dataset(df, rename_map, label_col):
    """
    Clean and rename dataset columns, drop invalid rows.
    """
    # Rename columns
    df = df.rename(columns=rename_map)

    # Keep only relevant columns
    keep_cols = [
        'orbital_period',
        'transit_duration',
        'planet_radius',
        'transit_depth',
        'stellar_radius',
        'stellar_temp',
        'label'
    ]
    # Keep only columns that exist in df
    keep_cols_existing = [col for col in keep_cols if col in df.columns]
    df = df[keep_cols_existing]

    # Drop missing values
    df = df.dropna()

    # Remove physically invalid values
    if 'orbital_period' in df.columns:
        df = df[df['orbital_period'] > 0]
    if 'planet_radius' in df.columns:
        df = df[df['planet_radius'] > 0]
    if 'stellar_radius' in df.columns:
        df = df[df['stellar_radius'] > 0]
    if 'stellar_temp' in df.columns:
        df = df[df['stellar_temp'] > 0]

    return df

def normalize_labels(df, label_col):
    """
    Normalize labels to: confirmed, candidate, false_positive
    """
    df[label_col] = df[label_col].astype(str).str.lower().replace({
        'confirmed': 'confirmed',
        'cp': 'confirmed',
        'candidate': 'candidate',
        'pc': 'candidate',
        'false positive': 'false_positive',
        'fp': 'false_positive'
    })
    return df

# ----------------- Load and Clean KOI -----------------
print("📥 Loading KOI dataset...")
koi = pd.read_csv("koi_dataset.csv", comment='#', on_bad_lines='skip', quotechar='"')
koi_map = {
    'koi_period': 'orbital_period',
    'koi_duration': 'transit_duration',
    'koi_prad': 'planet_radius',
    'koi_depth': 'transit_depth',
    'koi_srad': 'stellar_radius',
    'koi_steff': 'stellar_temp',
    'koi_disposition': 'label'
}
koi_clean = clean_dataset(koi, koi_map, 'koi_disposition')
koi_clean = normalize_labels(koi_clean, 'label')
print("✅ KOI cleaned")

# ----------------- Load and Clean K2 -----------------
print("📥 Loading K2 dataset...")
k2 = pd.read_csv("k2_dataset.csv", comment='#', on_bad_lines='skip', quotechar='"')
k2_map = {
    'pl_orbper': 'orbital_period',
    'pl_trandurh': 'transit_duration',  # or 'pl_trandur' if available
    'pl_rade': 'planet_radius',
    'pl_trandep': 'transit_depth',
    'st_rad': 'stellar_radius',
    'st_teff': 'stellar_temp',
    'disposition': 'label'
}
k2_clean = clean_dataset(k2, k2_map, 'disposition')
k2_clean = normalize_labels(k2_clean, 'label')
print("✅ K2 cleaned")

# ----------------- Load and Clean TOI -----------------
print("📥 Loading TOI dataset...")
toi = pd.read_csv("toi_dataset.csv", comment='#', on_bad_lines='skip', quotechar='"')
toi_map = {
    'pl_orbper': 'orbital_period',
    'pl_trandurh': 'transit_duration',
    'pl_rade': 'planet_radius',
    'pl_trandep': 'transit_depth',
    'st_rad': 'stellar_radius',
    'st_teff': 'stellar_temp',
    'tfopwg_disp': 'label'
}
toi_clean = clean_dataset(toi, toi_map, 'tfopwg_disp')
toi_clean = normalize_labels(toi_clean, 'label')
print("✅ TOI cleaned")

# ----------------- Combine Datasets -----------------
print("🔗 Combining datasets...")
combined = pd.concat([koi_clean, k2_clean, toi_clean], ignore_index=True)

# Shuffle the combined dataset
combined = combined.sample(frac=1, random_state=42).reset_index(drop=True)

# Save to CSV
combined.to_csv("combined_exoplanet_dataset.csv", index=False)
print("🎉 Combined dataset saved as combined_exoplanet_dataset.csv")

# ----------------- Class Balance -----------------
print("📝 Label distribution:")
print(combined['label'].value_counts())
