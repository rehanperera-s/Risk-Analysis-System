"""
Prepares the customer features dataset for ML model training.
Handles encoding, scaling, and train/test splitting.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder


NUMERIC_FEATURES = [
    "tenure_days", "age", "total_bets", "total_staked", "avg_stake",
    "max_stake", "total_pnl", "win_rate", "days_since_last_bet",
    "active_days_span", "bet_frequency", "support_tickets",
    "promo_bets_pct", "unique_sports", "avg_odds",
    "final_balance", "min_balance", "max_balance",
    "used_signup_bonus", "responsible_gambling_flags",
]

CATEGORICAL_FEATURES = [
    "gender", "state", "preferred_sport",
    "bet_type_preference", "deposit_method",
]

TARGET = "churned"


def load_and_clean(path="data/processed/customer_features.csv"):
    df = pd.read_csv(path)

    date_cols = [c for c in df.columns if "date" in c.lower()]
    df = df.drop(columns=date_cols + ["customer_id"], errors="ignore")

    for col in NUMERIC_FEATURES:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df[col] = df[col].fillna(df[col].median())

    return df


def encode_categoricals(df):
    encoders = {}
    for col in CATEGORICAL_FEATURES:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            encoders[col] = le
    return df, encoders


def prepare_features(df):
    df, encoders = encode_categoricals(df)

    feature_cols = [c for c in NUMERIC_FEATURES + CATEGORICAL_FEATURES if c in df.columns]
    X = df[feature_cols].copy()
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median())
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train), columns=feature_cols, index=X_train.index
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test), columns=feature_cols, index=X_test.index
    )

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, encoders, feature_cols


if __name__ == "__main__":
    df = load_and_clean()
    X_train, X_test, y_train, y_test, scaler, encoders, features = prepare_features(df)
    print(f"Train: {X_train.shape} | Test: {X_test.shape}")
    print(f"Churn rate — Train: {y_train.mean():.1%} | Test: {y_test.mean():.1%}")
    print(f"Features ({len(features)}): {features}")
