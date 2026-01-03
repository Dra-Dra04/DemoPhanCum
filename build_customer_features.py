import pandas as pd
import numpy as np

# ===== 1) Read raw CSVs =====
customer  = pd.read_csv("Logistics/Customer.csv")
shipment  = pd.read_csv("Logistics/Shipment_Details.csv")
status    = pd.read_csv("Logistics/Status.csv")
payment   = pd.read_csv("Logistics/Payment_Details.csv")

# ===== 2) Parse dates =====
status["Sent_date"] = pd.to_datetime(status["Sent_date"], errors="coerce")
status["Delivery_date"] = pd.to_datetime(status["Delivery_date"], errors="coerce")

# ===== 3) Join shipment + status + payment =====
df = shipment.merge(status, on="SH_ID", how="left")
df = df.merge(
    payment[["C_ID", "SH_ID", "Payment_Status", "Payment_Mode"]],
    on=["C_ID", "SH_ID"],
    how="left"
)

# ===== 4) Numeric conversion =====
df["SH_WEIGHT"]  = pd.to_numeric(df["SH_WEIGHT"], errors="coerce")
df["SH_CHARGES"] = pd.to_numeric(df["SH_CHARGES"], errors="coerce")

# ===== 5) Create behavioral features =====
df["shipment_weight"] = df["SH_WEIGHT"]
df["shipping_cost"]   = df["SH_CHARGES"]

df["delivery_days"] = (df["Delivery_date"] - df["Sent_date"]).dt.days

df["is_express"] = (df["SER_TYPE"].astype(str).str.upper() == "EXPRESS").astype(int)
df["is_international"] = (df["SH_DOMAIN"].astype(str).str.upper() == "INTERNATIONAL").astype(int)

df["is_cod"]  = (df["Payment_Mode"].astype(str).str.upper() == "COD").astype(int)
df["is_paid"] = (df["Payment_Status"].astype(str).str.upper() == "PAID").astype(int)

# ===== 6) Aggregate to customer-level =====
# (Dù hiện tại 1 khách = 1 shipment, bước này giúp đúng học thuật & mở rộng)
features = df.groupby("C_ID").agg(
    shipment_weight=("shipment_weight", "mean"),
    shipping_cost=("shipping_cost", "mean"),
    delivery_days=("delivery_days", "mean"),
    is_express=("is_express", "max"),
    is_international=("is_international", "max"),
    is_cod=("is_cod", "max"),
    is_paid=("is_paid", "max"),
).reset_index()

# ===== 7) Clean NaN =====
for col in ["shipment_weight", "shipping_cost", "delivery_days"]:
    features[col] = features[col].fillna(features[col].median())

for col in ["is_express", "is_international", "is_cod", "is_paid"]:
    features[col] = features[col].fillna(0)

# ===== 8) Save final feature table =====
features.to_csv("customer_features_final.csv", index=False)
print("✅ Saved: customer_features_final.csv")
