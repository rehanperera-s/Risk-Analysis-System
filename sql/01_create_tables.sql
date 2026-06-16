-- Schema for betting platform customer churn analysis

CREATE TABLE IF NOT EXISTS customers (
    customer_id       INTEGER PRIMARY KEY,
    signup_date       DATE NOT NULL,
    tenure_days       INTEGER,
    age               INTEGER,
    gender            VARCHAR(1),
    state             VARCHAR(2),
    preferred_sport   VARCHAR(20),
    bet_type_preference VARCHAR(20),
    deposit_method    VARCHAR(20),
    used_signup_bonus INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS transactions (
    transaction_id    INTEGER PRIMARY KEY AUTOINCREMENT,
    customer_id       INTEGER NOT NULL,
    bet_date          DATE NOT NULL,
    sport             VARCHAR(20),
    stake             DECIMAL(10, 2),
    odds              DECIMAL(6, 2),
    won               INTEGER,
    pnl               DECIMAL(10, 2),
    balance_after     DECIMAL(10, 2),
    FOREIGN KEY (customer_id) REFERENCES customers(customer_id)
);

CREATE TABLE IF NOT EXISTS events (
    event_id          VARCHAR(64) PRIMARY KEY,
    sport             VARCHAR(50),
    home_team         VARCHAR(100),
    away_team         VARCHAR(100),
    commence_time     DATETIME,
    bookmaker         VARCHAR(50),
    home_odds         DECIMAL(6, 2),
    away_odds         DECIMAL(6, 2),
    draw_odds         DECIMAL(6, 2)
);
