"""
Train the Moral AI choice predictor and save it to disk.

Reproduces the pipeline from AAD_MoralAI_NewApproach.ipynb (Sections 1, 3, 5-7, 9, 11)
and persists the fitted scikit-learn pipeline so the Streamlit UI can load it
instantly without retraining.

Run:
    python train_model.py
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from datasets import load_dataset
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
ROOT = Path(__file__).parent
MM_PATH = ROOT / "SharedResponsesSurvey.csv"
BRIDGE_PATH = ROOT / "bridge.csv"
HF_DATASET_ID = "ecorbari/IPIP120-SCORES"

MODEL_OUT = ROOT / "model.joblib"
META_OUT = ROOT / "model_meta.json"

SAMPLE_ROWS = 100_000
CORR_SAMPLE = 30_000
TARGET = "saved"

MM_USECOLS = [
    "UserID", "ScenarioOrder", "Intervention", "PedPed", "Barrier",
    "CrossingSignal", "ScenarioType", "DefaultChoiceIsOmission",
    "NumberOfCharacters", "DiffNumberOFCharacters", "Saved",
    "UserCountry3", "Review_age", "Review_gender", "Review_income",
    "Review_education", "Review_political", "Review_religious",
]

TRAITS = ["openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"]


# ---------------------------------------------------------------------------
# Helpers (lifted from the notebook)
# ---------------------------------------------------------------------------
def load_personality_from_hf(dataset_id: str = HF_DATASET_ID) -> pd.DataFrame:
    ds = load_dataset(dataset_id)
    split = next(iter(ds.keys()))
    return ds[split].to_pandas()


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = (
        df.columns.str.strip().str.lower().str.replace(" ", "_", regex=False)
    )
    if "userid" in df.columns:
        df = df.rename(columns={"userid": "user_id"})
    return df


def normalize_big5_columns(df: pd.DataFrame) -> pd.DataFrame:
    aliases = {
        "openness": ["openness", "big5_openness", "o", "open"],
        "conscientiousness": ["conscientiousness", "big5_conscientiousness", "c", "consc"],
        "extraversion": ["extraversion", "big5_extraversion", "e", "extra"],
        "agreeableness": ["agreeableness", "big5_agreeableness", "a", "agree"],
        "neuroticism": ["neuroticism", "big5_neuroticism", "n", "neuro"],
    }
    available = {c.lower(): c for c in df.columns}
    resolved = {}
    for target, alias_list in aliases.items():
        for alias in alias_list:
            if alias.lower() in available:
                resolved[target] = pd.to_numeric(df[available[alias.lower()]], errors="coerce")
                break
    missing = sorted(set(aliases) - set(resolved))
    if missing:
        raise ValueError("Missing Big Five columns: " + ", ".join(missing))

    result = pd.DataFrame(resolved).dropna().reset_index(drop=True)
    if "sex" in available:
        result["sex"] = df[available["sex"]].iloc[result.index].values
    if "age" in available:
        result["age"] = pd.to_numeric(df[available["age"]].iloc[result.index], errors="coerce").values
    if "country" in available:
        result["country"] = df[available["country"]].iloc[result.index].astype(str).str.strip().values
    return result


def create_age_buckets(df: pd.DataFrame, age_col: str = "age", bucket_size: int = 5) -> pd.Series:
    s = pd.to_numeric(df[age_col], errors="coerce")
    return (s // bucket_size) * bucket_size


def link_profiles_with_personality(profiles, personality_df, age_tolerance=5, use_country=True):
    prof = profiles.copy()
    big5 = normalize_big5_columns(personality_df)

    if "gender" in prof.columns:
        prof["gender_norm"] = prof["gender"].astype(str).str.strip().str.lower()
    if "country" in prof.columns:
        prof["country_norm"] = prof["country"].astype(str).str.strip().str.lower()

    big5["gender_norm"] = big5["sex"].map({1: "female", 2: "male"}).fillna(big5["sex"].astype(str).str.lower())
    big5["country_norm"] = big5["country"].astype(str).str.strip().str.lower()

    prof["age_bucket"] = create_age_buckets(prof, "age", age_tolerance)
    big5["age_bucket"] = create_age_buckets(big5, "age", age_tolerance)

    available_traits = [t for t in TRAITS if t in big5.columns]
    merge_keys = ["gender_norm", "age_bucket"]
    if use_country and "country_norm" in prof.columns and "country_norm" in big5.columns:
        merge_keys.append("country_norm")

    big5_agg = big5.groupby(merge_keys, dropna=False)[available_traits].mean().reset_index()
    rename_map = {t: f"ipip_{t}" for t in available_traits}
    big5_agg = big5_agg.rename(columns=rename_map)

    linked = pd.merge(prof, big5_agg, on=merge_keys, how="left")

    unmatched_mask = linked[f"ipip_{available_traits[0]}"].isna()
    if unmatched_mask.any():
        fallback_keys = ["gender_norm", "age_bucket"]
        big5_fb = big5.groupby(fallback_keys, dropna=False)[available_traits].mean().reset_index()
        big5_fb = big5_fb.rename(columns=rename_map)
        unmatched_df = linked[unmatched_mask].drop(columns=[f"ipip_{t}" for t in available_traits])
        rematched = pd.merge(unmatched_df, big5_fb, on=fallback_keys, how="left")
        linked.loc[unmatched_mask, [f"ipip_{t}" for t in available_traits]] = rematched[
            [f"ipip_{t}" for t in available_traits]
        ].values

        still_missing = linked[f"ipip_{available_traits[0]}"].isna()
        for t in available_traits:
            linked.loc[still_missing, f"ipip_{t}"] = big5[t].mean()

    linked = linked.drop(columns=["gender_norm", "country_norm", "age_bucket"], errors="ignore")
    return linked


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------
def build_dataset() -> pd.DataFrame:
    print("Loading Moral Machine dataset...")
    if not MM_PATH.exists():
        raise FileNotFoundError(f"Missing file: {MM_PATH}")
    mm_sample = pd.read_csv(MM_PATH, usecols=MM_USECOLS, low_memory=False)
    if len(mm_sample) > SAMPLE_ROWS:
        mm_sample = mm_sample.sample(n=SAMPLE_ROWS, random_state=42).reset_index(drop=True)

    print("Loading IPIP-120 personality dataset from Hugging Face...")
    df_ipip = load_personality_from_hf()

    mm_clean = standardize_columns(mm_sample).rename(columns={
        "review_gender": "gender", "review_age": "age", "review_income": "income",
        "review_education": "education", "review_political": "political",
        "review_religious": "religious", "usercountry3": "country",
    })

    for col in ["age", "income", "political", "religious"]:
        if col in mm_clean.columns:
            mm_clean[col] = pd.to_numeric(mm_clean[col], errors="coerce")
    if "gender" in mm_clean.columns:
        mm_clean["gender"] = mm_clean["gender"].astype(str).str.strip().str.lower()
    if "country" in mm_clean.columns:
        mm_clean["country"] = mm_clean["country"].astype(str).str.strip().str.lower()
    if "user_id" in mm_clean.columns:
        mm_clean["user_id"] = mm_clean["user_id"].astype(str)

    scenario_cols = ["scenarioorder", "intervention", "pedped", "barrier",
                     "crossingsignal", "scenariotype", "defaultchoiceisomission",
                     "numberofcharacters", "diffnumberofcharacters", "saved"]
    demo_cols = [c for c in ["age", "gender", "country", "income", "education",
                              "political", "religious"] if c in mm_clean.columns]
    cols = [c for c in ["user_id"] + scenario_cols + demo_cols if c in mm_clean.columns]

    mm_scenario = mm_clean[cols].dropna(subset=["age", "gender"])
    if len(mm_scenario) > CORR_SAMPLE:
        mm_scenario = mm_scenario.sample(n=CORR_SAMPLE, random_state=42)

    print("Linking personality traits to scenarios via demographic cohorts...")
    linked = link_profiles_with_personality(mm_scenario, df_ipip, age_tolerance=5, use_country=True)
    df_final = linked.dropna()
    print(f"Final dataset shape: {df_final.shape}")
    return df_final


def train(df_final: pd.DataFrame):
    cols_to_drop = [TARGET, "user_id", "scenarioorder"]
    X = df_final.drop(columns=cols_to_drop, errors="ignore")
    y = df_final[TARGET].astype(int)

    categorical_cols = [c for c in ["scenariotype", "country", "gender", "education", "income"]
                        if c in X.columns]
    numeric_cols = [c for c in X.columns if c not in categorical_cols]

    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), numeric_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
    ])

    pipe = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(
            n_estimators=150, max_depth=12, random_state=42, n_jobs=-1,
        )),
    ])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print("Training Random Forest...")
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    print(f"Test accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(classification_report(y_test, y_pred))

    # Defaults for fields the UI does not collect: use most common / median
    defaults = {}
    for c in X.columns:
        if c in numeric_cols:
            defaults[c] = float(pd.to_numeric(X[c], errors="coerce").median())
        else:
            mode = X[c].mode(dropna=True)
            defaults[c] = (mode.iloc[0] if not mode.empty else "")

    meta = {
        "feature_columns": list(X.columns),
        "numeric_columns": numeric_cols,
        "categorical_columns": categorical_cols,
        "defaults": defaults,
        "target": TARGET,
    }
    return pipe, meta


def main():
    df_final = build_dataset()
    pipe, meta = train(df_final)
    joblib.dump(pipe, MODEL_OUT)
    META_OUT.write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
    print(f"\nSaved model to {MODEL_OUT}")
    print(f"Saved metadata to {META_OUT}")


if __name__ == "__main__":
    main()
