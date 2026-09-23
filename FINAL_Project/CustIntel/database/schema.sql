PRAGMA foreign_keys = ON;
DROP TABLE IF EXISTS fact_browsing;
DROP TABLE IF EXISTS fact_orders;
DROP TABLE IF EXISTS dim_reviews;
DROP TABLE IF EXISTS dim_products;
DROP TABLE IF EXISTS dim_customers;

CREATE TABLE dim_customers (
    customer_id TEXT PRIMARY KEY,
    customer_city TEXT NOT NULL,
    customer_state TEXT NOT NULL,
    signup_date TEXT NOT NULL
);
CREATE TABLE dim_products (
    product_id TEXT PRIMARY KEY,
    product_category TEXT NOT NULL,
    product_weight_g REAL,
    product_length_cm REAL,
    product_height_cm REAL,
    product_width_cm REAL
);
CREATE TABLE dim_reviews (
    review_id TEXT PRIMARY KEY,
    order_id TEXT NOT NULL UNIQUE,
    review_score INTEGER CHECK(review_score BETWEEN 1 AND 5),
    review_comment_message TEXT,
    review_date TEXT NOT NULL
);
CREATE TABLE fact_orders (
    order_id TEXT PRIMARY KEY,
    customer_id TEXT NOT NULL,
    product_id TEXT NOT NULL,
    order_status TEXT NOT NULL CHECK(order_status IN ('delivered','shipped','canceled')),
    order_purchase_date TEXT NOT NULL,
    price REAL NOT NULL CHECK(price >= 0),
    freight_value REAL NOT NULL CHECK(freight_value >= 0),
    quantity INTEGER NOT NULL CHECK(quantity > 0),
    FOREIGN KEY(customer_id) REFERENCES dim_customers(customer_id),
    FOREIGN KEY(product_id) REFERENCES dim_products(product_id)
);
CREATE TABLE fact_browsing (
    event_id TEXT PRIMARY KEY,
    customer_id TEXT NOT NULL,
    event_time TEXT NOT NULL,
    product_category TEXT NOT NULL,
    FOREIGN KEY(customer_id) REFERENCES dim_customers(customer_id)
);
CREATE INDEX idx_orders_customer ON fact_orders(customer_id);
CREATE INDEX idx_orders_product ON fact_orders(product_id);
CREATE INDEX idx_orders_date ON fact_orders(order_purchase_date);
CREATE INDEX idx_reviews_order ON dim_reviews(order_id);
CREATE INDEX idx_browsing_customer_time ON fact_browsing(customer_id, event_time);
