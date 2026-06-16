"""
Fetches real sports odds from The Odds API and simulates a customer base
of bettors placing wagers on those events to create churn prediction data.
"""

import json
import os
import random
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import requests

from src.config import ODDS_API_KEY, ODDS_API_BASE, SPORTS, NUM_CUSTOMERS, SIMULATION_DAYS


def fetch_odds(sport, regions="us", markets="h2h"):
    """Fetch current odds for a sport from The Odds API."""
    url = f"{ODDS_API_BASE}/sports/{sport}/odds/"
    params = {
        "apiKey": ODDS_API_KEY,
        "regions": regions,
        "markets": markets,
        "oddsFormat": "decimal",
    }
    resp = requests.get(url, params=params)
    resp.raise_for_status()

    remaining = resp.headers.get("x-requests-remaining", "?")
    print(f"  [{sport}] Fetched {len(resp.json())} events | Credits remaining: {remaining}")
    return resp.json()


def fetch_all_odds():
    """Fetch odds across all configured sports."""
    all_events = []
    for sport in SPORTS:
        try:
            events = fetch_odds(sport)
            for event in events:
                event["sport"] = sport
            all_events.extend(events)
        except requests.exceptions.HTTPError as e:
            print(f"  [{sport}] Skipped: {e}")
    return all_events


def save_raw_odds(events, path="data/raw/odds_data.json"):
    """Save raw API response to JSON."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(events, f, indent=2)
    print(f"Saved {len(events)} events to {path}")


def parse_events_to_df(events):
    """Convert raw odds events into a flat DataFrame."""
    rows = []
    for event in events:
        bookmakers = event.get("bookmakers", [])
        if not bookmakers:
            continue
        bk = bookmakers[0]
        market = bk.get("markets", [{}])[0]
        outcomes = {o["name"]: o["price"] for o in market.get("outcomes", [])}
        rows.append({
            "event_id": event["id"],
            "sport": event.get("sport", event.get("sport_key", "")),
            "home_team": event["home_team"],
            "away_team": event["away_team"],
            "commence_time": event["commence_time"],
            "bookmaker": bk["key"],
            "home_odds": outcomes.get(event["home_team"], 0),
            "away_odds": outcomes.get(event["away_team"], 0),
            "draw_odds": outcomes.get("Draw", 0),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Customer simulation
# ---------------------------------------------------------------------------

def generate_customers(n=NUM_CUSTOMERS):
    """Generate a synthetic customer base for a betting platform."""
    rng = np.random.default_rng(42)

    signup_days_ago = rng.integers(30, 365 * 3, size=n)
    signup_dates = [datetime.now() - timedelta(days=int(d)) for d in signup_days_ago]

    customers = pd.DataFrame({
        "customer_id": range(1, n + 1),
        "signup_date": signup_dates,
        "tenure_days": signup_days_ago,
        "age": rng.integers(21, 65, size=n),
        "gender": rng.choice(["M", "F"], size=n, p=[0.72, 0.28]),
        "state": rng.choice(
            ["NY", "NJ", "PA", "IL", "CO", "MI", "AZ", "VA", "OH", "MA"],
            size=n,
        ),
        "preferred_sport": rng.choice(
            ["NFL", "NBA", "Soccer", "MLB", "NHL"],
            size=n,
            p=[0.35, 0.25, 0.15, 0.15, 0.10],
        ),
        "bet_type_preference": rng.choice(
            ["moneyline", "spread", "totals", "parlay", "live"],
            size=n,
            p=[0.30, 0.25, 0.20, 0.15, 0.10],
        ),
        "deposit_method": rng.choice(
            ["card", "bank_transfer", "paypal", "crypto"],
            size=n,
            p=[0.45, 0.25, 0.20, 0.10],
        ),
        "used_signup_bonus": rng.choice([0, 1], size=n, p=[0.35, 0.65]),
    })
    return customers


def simulate_betting_activity(customers, events_df, days=SIMULATION_DAYS):
    """Simulate betting transactions for each customer over a time window."""
    rng = np.random.default_rng(42)
    records = []

    has_events = len(events_df) > 0

    for _, cust in customers.iterrows():
        cid = cust["customer_id"]
        tenure = cust["tenure_days"]

        activity_rate = rng.uniform(0.05, 0.95)
        avg_bet = rng.lognormal(mean=3.0, sigma=0.8)
        avg_bet = np.clip(avg_bet, 5, 500)

        num_bets = int(activity_rate * days * rng.uniform(0.3, 1.5))
        num_bets = max(num_bets, 1)

        bet_days = sorted(rng.choice(days, size=min(num_bets, days), replace=False))
        balance = float(rng.uniform(50, 2000))
        total_deposits = balance
        total_withdrawals = 0.0

        for day_offset in bet_days:
            bet_date = datetime.now() - timedelta(days=int(days - day_offset))
            stake = float(rng.lognormal(mean=np.log(avg_bet), sigma=0.5))
            stake = np.clip(stake, 2, balance * 0.5) if balance > 10 else 5.0

            if has_events:
                evt = events_df.sample(1, random_state=int(cid + day_offset)).iloc[0]
                odds = float(rng.choice([evt["home_odds"], evt["away_odds"]]))
            else:
                odds = float(rng.uniform(1.2, 5.0))

            odds = max(odds, 1.01)
            implied_prob = 1.0 / odds
            won = rng.random() < implied_prob
            pnl = stake * (odds - 1) if won else -stake
            balance += pnl

            if balance < 20:
                deposit = float(rng.choice([50, 100, 200, 500], p=[0.4, 0.3, 0.2, 0.1]))
                balance += deposit
                total_deposits += deposit

            if balance > 1000 and rng.random() < 0.15:
                withdraw = float(rng.uniform(100, balance * 0.4))
                balance -= withdraw
                total_withdrawals += withdraw

            records.append({
                "customer_id": cid,
                "bet_date": bet_date,
                "day_offset": int(day_offset),
                "sport": evt["sport"] if has_events else cust["preferred_sport"],
                "stake": round(float(stake), 2),
                "odds": round(odds, 2),
                "won": int(won),
                "pnl": round(float(pnl), 2),
                "balance_after": round(float(balance), 2),
            })

    txns = pd.DataFrame(records)
    return txns


def build_customer_features(customers, txns):
    """Aggregate transaction data into customer-level features for modeling."""
    agg = txns.groupby("customer_id").agg(
        total_bets=("stake", "count"),
        total_staked=("stake", "sum"),
        avg_stake=("stake", "mean"),
        max_stake=("stake", "max"),
        total_pnl=("pnl", "sum"),
        win_count=("won", "sum"),
        unique_sports=("sport", "nunique"),
        last_bet_date=("bet_date", "max"),
        first_bet_date=("bet_date", "min"),
        avg_odds=("odds", "mean"),
        final_balance=("balance_after", "last"),
    ).reset_index()

    agg["win_rate"] = agg["win_count"] / agg["total_bets"]
    agg["days_since_last_bet"] = (datetime.now() - agg["last_bet_date"]).dt.days
    agg["active_days_span"] = (agg["last_bet_date"] - agg["first_bet_date"]).dt.days
    agg["bet_frequency"] = agg["total_bets"] / agg["active_days_span"].replace(0, 1)

    deposits = txns.groupby("customer_id")["balance_after"].agg(["min", "max"])
    deposits.columns = ["min_balance", "max_balance"]
    agg = agg.merge(deposits, on="customer_id", how="left")

    rng = np.random.default_rng(99)
    n = len(agg)
    agg["support_tickets"] = rng.poisson(lam=1.5, size=n)
    agg["promo_bets_pct"] = rng.uniform(0, 0.4, size=n).round(3)
    agg["responsible_gambling_flags"] = rng.choice([0, 1], size=n, p=[0.85, 0.15])

    df = customers.merge(agg, on="customer_id", how="left")

    # --- Churn label (logistic model for realistic signal) ---
    from scipy.special import expit

    def _z(col):
        return (df[col] - df[col].mean()) / df[col].std().clip(1)

    z = (
        -1.2
        + 0.8 * _z("days_since_last_bet")
        - 0.7 * _z("total_pnl")
        - 0.5 * _z("total_bets")
        - 0.5 * _z("win_rate")
        + 0.4 * _z("support_tickets")
        - 0.3 * _z("tenure_days")
        - 0.3 * _z("bet_frequency")
        + 0.2 * _z("promo_bets_pct")
        - 0.2 * _z("final_balance")
        + rng.normal(0, 0.4, size=len(df))
    )
    churn_prob = expit(z)
    df["churned"] = (rng.random(len(df)) < churn_prob).astype(int)

    return df


def run_data_pipeline():
    """End-to-end: fetch odds, simulate customers, build features, save."""
    print("=== Fetching odds from The Odds API ===")
    raw_events = fetch_all_odds()
    save_raw_odds(raw_events)
    events_df = parse_events_to_df(raw_events)

    print(f"\n=== Parsed {len(events_df)} events with odds ===")
    if len(events_df) > 0:
        events_df.to_csv("data/raw/events.csv", index=False)

    print(f"\n=== Generating {NUM_CUSTOMERS} customers ===")
    customers = generate_customers()
    customers.to_csv("data/raw/customers.csv", index=False)

    print(f"\n=== Simulating betting activity over {SIMULATION_DAYS} days ===")
    txns = simulate_betting_activity(customers, events_df)
    txns.to_csv("data/raw/transactions.csv", index=False)
    print(f"  Generated {len(txns)} transactions")

    print("\n=== Building customer features ===")
    df = build_customer_features(customers, txns)
    df.to_csv("data/processed/customer_features.csv", index=False)
    print(f"  Final dataset: {len(df)} customers, {len(df.columns)} features")
    print(f"  Churn rate: {df['churned'].mean():.1%}")

    return df


if __name__ == "__main__":
    run_data_pipeline()
