-- Customer churn feature extraction queries

-- 1. Customer activity summary
SELECT
    c.customer_id,
    c.signup_date,
    c.tenure_days,
    c.age,
    c.gender,
    c.state,
    c.preferred_sport,
    c.bet_type_preference,
    c.deposit_method,
    c.used_signup_bonus,
    COUNT(t.transaction_id)              AS total_bets,
    SUM(t.stake)                         AS total_staked,
    AVG(t.stake)                         AS avg_stake,
    MAX(t.stake)                         AS max_stake,
    SUM(t.pnl)                           AS total_pnl,
    SUM(t.won)                           AS win_count,
    ROUND(SUM(t.won) * 1.0 / COUNT(*), 3) AS win_rate,
    COUNT(DISTINCT t.sport)              AS unique_sports,
    MAX(t.bet_date)                      AS last_bet_date,
    MIN(t.bet_date)                      AS first_bet_date,
    CAST(julianday('now') - julianday(MAX(t.bet_date)) AS INTEGER)
                                         AS days_since_last_bet,
    AVG(t.odds)                          AS avg_odds
FROM customers c
LEFT JOIN transactions t ON c.customer_id = t.customer_id
GROUP BY c.customer_id;


-- 2. High-risk customers (inactive > 45 days AND negative P&L)
SELECT
    c.customer_id,
    c.preferred_sport,
    SUM(t.pnl) AS total_pnl,
    CAST(julianday('now') - julianday(MAX(t.bet_date)) AS INTEGER)
        AS days_inactive,
    COUNT(t.transaction_id) AS total_bets
FROM customers c
JOIN transactions t ON c.customer_id = t.customer_id
GROUP BY c.customer_id
HAVING days_inactive > 45 AND total_pnl < 0
ORDER BY total_pnl ASC;


-- 3. Monthly churn trend
SELECT
    strftime('%Y-%m', t.bet_date) AS month,
    COUNT(DISTINCT t.customer_id) AS active_customers,
    COUNT(t.transaction_id)       AS total_bets,
    SUM(t.stake)                  AS total_staked,
    SUM(t.pnl)                    AS total_pnl
FROM transactions t
GROUP BY month
ORDER BY month;


-- 4. Churn rate by sport
SELECT
    c.preferred_sport,
    COUNT(*)                                          AS total_customers,
    SUM(CASE WHEN days_inactive > 45 THEN 1 ELSE 0 END) AS at_risk,
    ROUND(SUM(CASE WHEN days_inactive > 45 THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 1)
        AS churn_pct
FROM customers c
JOIN (
    SELECT customer_id,
           CAST(julianday('now') - julianday(MAX(bet_date)) AS INTEGER) AS days_inactive
    FROM transactions
    GROUP BY customer_id
) t ON c.customer_id = t.customer_id
GROUP BY c.preferred_sport
ORDER BY churn_pct DESC;


-- 5. Revenue at risk from high-risk customers
SELECT
    SUM(avg_monthly_stake) AS monthly_revenue_at_risk
FROM (
    SELECT
        c.customer_id,
        SUM(t.stake) / NULLIF(COUNT(DISTINCT strftime('%Y-%m', t.bet_date)), 0)
            AS avg_monthly_stake
    FROM customers c
    JOIN transactions t ON c.customer_id = t.customer_id
    GROUP BY c.customer_id
    HAVING CAST(julianday('now') - julianday(MAX(t.bet_date)) AS INTEGER) > 45
       AND SUM(t.pnl) < -200
);
