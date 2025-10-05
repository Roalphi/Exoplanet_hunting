import pandas as pd

def clean_dataset(df, rename_map, label_col):
    """
    Cleans a dataset: renames columns, keeps relevant columns, drops invalids, normalizes labels.
    Handles missing columns gracefully.
    """
    # Rename columns
    df = df.rename(columns=rename_map)

    # Define all possible columns
    keep_cols = [
        'orbital_period',
        'transit_duration',
        'planet_radius',
        'transit_depth',
        'stellar_radius',
        'stellar_temp',
        'label'
    ]

    # Only keep columns that actually exist in this dataset
    keep_cols = [col for col in keep_cols if col in df.columns]
    df = df[keep_cols]

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

    # Normalize label names
    df['label'] = df['label'].astype(str).str.lower().replace({
        'confirmed': 'confirmed',
        'candidate': 'candidate',
        'false positive': 'false_positive',
        'fp': 'false_positive',
        'p': 'candidate'
    })

    return df


# ----------------- KOI -----------------
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
koi_clean.to_csv("clean_koi.csv", index=False)
print("✅ Saved clean_koi.csv")


# ----------------- K2 -----------------
print("📥 Loading K2 dataset...")
k2 = pd.read_csv("k2_dataset.csv", comment='#', on_bad_lines='skip', quotechar='"')
k2_map = {
    'pl_orbper': 'orbital_period',
    'pl_rade': 'planet_radius',
    'st_rad': 'stellar_radius',
    'st_teff': 'stellar_temp',
    'disposition': 'label'
    # Note: K2 dataset does not have transit_duration or transit_depth
}
k2_clean = clean_dataset(k2, k2_map, 'disposition')
k2_clean.to_csv("clean_k2.csv", index=False)
print("✅ Saved clean_k2.csv")


# ----------------- TOI -----------------
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
toi_clean.to_csv("clean_toi.csv", index=False)
print("✅ Saved clean_toi.csv")


print("🎉 All datasets loaded, cleaned, and saved successfully!")
