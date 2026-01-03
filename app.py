import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import streamlit as st
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


# ---------------------------
# Utils
# ---------------------------
def infer_id_column(df: pd.DataFrame) -> str | None:
    """Try to find a reasonable customer id column."""
    candidates = ["customer_id", "C_ID", "c_id", "CustomerID", "Customer_Id", "id", "ID"]
    for c in candidates:
        if c in df.columns:
            return c
    return None


def safe_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Convert selected columns to numeric, coercing errors to NaN."""
    out = df.copy()
    for c in cols:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def plot_elbow(inertias: list[float], ks: list[int]):
    fig = plt.figure()
    plt.plot(ks, inertias, marker="o")
    plt.xlabel("K (number of clusters)")
    plt.ylabel("Inertia (WCSS)")
    plt.title("Elbow Method")
    plt.grid(True)
    st.pyplot(fig)
    plt.close(fig)


def plot_silhouette(scores: list[float], ks: list[int]):
    fig = plt.figure()
    plt.plot(ks, scores, marker="o")
    plt.xlabel("K (number of clusters)")
    plt.ylabel("Silhouette score")
    plt.title("Silhouette Scores")
    plt.grid(True)
    st.pyplot(fig)
    plt.close(fig)


def plot_cluster_sizes(labels: np.ndarray):
    unique, counts = np.unique(labels, return_counts=True)
    fig = plt.figure()
    plt.bar(unique.astype(str), counts)
    plt.xlabel("Cluster")
    plt.ylabel("Number of customers")
    plt.title("Cluster size distribution")
    st.pyplot(fig)
    plt.close(fig)


# ---------------------------
# Streamlit UI
# ---------------------------
st.set_page_config(page_title="Logistics Customer Segmentation (K-Means)", layout="wide")

st.title("Phân cụm khách hàng Logistics (K-Means)")

uploaded = st.file_uploader("Upload CSV", type=["csv"])

if not uploaded:
    st.info("Hãy upload file CSV để bắt đầu.")
    st.stop()

# Read CSV
try:
    df = pd.read_csv(uploaded)
except Exception:
    uploaded.seek(0)
    df = pd.read_csv(uploaded, encoding="utf-8", engine="python")

st.subheader("1) Xem nhanh dữ liệu")
st.write(f"Số dòng: **{len(df)}** | Số cột: **{len(df.columns)}**")
st.dataframe(df.head(20), use_container_width=True)

# Identify ID column
id_col_guess = infer_id_column(df)

with st.sidebar:
    st.header("Cấu hình phân cụm")

    id_col = st.selectbox(
        "Cột định danh khách hàng",
        options=["(không có)"] + list(df.columns),
        index=(1 + list(df.columns).index(id_col_guess)) if id_col_guess in df.columns else 0,
    )
    if id_col == "(không có)":
        id_col = None

    # Feature selection
    numeric_cols = [c for c in df.columns if c != id_col]
    default_features = []
    for c in ["total_orders", "avg_weight", "avg_cost", "total_cost", "delivery_time", "return_rate", "cod_rate"]:
        if c in df.columns and c != id_col:
            default_features.append(c)

    features = st.multiselect(
        "Chọn các thuộc tính để phân cụm (ưu tiên numeric)",
        options=numeric_cols,
        default=default_features if default_features else (numeric_cols[:6] if len(numeric_cols) >= 6 else numeric_cols),
    )

    missing_strategy = st.selectbox(
        "Xử lý giá trị thiếu (NaN)",
        options=["Drop rows with NaN", "Fill with median"],
        index=0,
    )

    st.divider()
    st.subheader("Chọn K")
    k_min = st.number_input("K min", min_value=2, max_value=20, value=2, step=1)
    k_max = st.number_input("K max", min_value=2, max_value=30, value=8, step=1)
    if k_max < k_min:
        st.warning("K max phải >= K min.")
    auto_pick = st.checkbox("Tự chọn K theo Silhouette (khuyến nghị)", value=True)
    chosen_k = st.number_input("Hoặc chọn K thủ công", min_value=2, max_value=50, value=4, step=1)

    st.divider()
    run_btn = st.button("Chạy phân cụm", type="primary")


if not features:
    st.error("Bạn chưa chọn thuộc tính để phân cụm.")
    st.stop()

# Prepare X
work = df.copy()
work = safe_numeric(work, features)

# Handle missing
if missing_strategy == "Drop rows with NaN":
    before = len(work)
    work = work.dropna(subset=features)
    after = len(work)
    if after < before:
        st.warning(f"Đã loại {before - after} dòng do thiếu dữ liệu ở các cột đã chọn.")
else:
    # Fill with median
    for c in features:
        med = work[c].median()
        work[c] = work[c].fillna(med)

if len(work) < 5:
    st.error("Dữ liệu sau tiền xử lý quá ít để phân cụm. Hãy chọn lại thuộc tính hoặc cách xử lý NaN.")
    st.stop()

X = work[features].values

# Scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Run on click
if not run_btn:
    st.info("Chọn cấu hình bên trái và bấm **Chạy phân cụm**.")
    st.stop()

st.subheader("2) Chọn K (Elbow / Silhouette)")

ks = list(range(int(k_min), int(k_max) + 1))
inertias = []
sil_scores = []

for k in ks:
    km = KMeans(n_clusters=k, random_state=42, n_init="auto")
    labels = km.fit_predict(X_scaled)
    inertias.append(km.inertia_)
    # Silhouette needs at least 2 clusters and < n_samples
    if 1 < k < len(X_scaled):
        sil_scores.append(silhouette_score(X_scaled, labels))
    else:
        sil_scores.append(np.nan)

colA, colB = st.columns(2)
with colA:
    plot_elbow(inertias, ks)
with colB:
    plot_silhouette(sil_scores, ks)

# Pick K
best_k = None
if auto_pick:
    valid = [(k, s) for k, s in zip(ks, sil_scores) if np.isfinite(s)]
    if valid:
        best_k = max(valid, key=lambda t: t[1])[0]
        st.success(f"Tự chọn K theo Silhouette: **K = {best_k}**")
    else:
        st.warning("Không tính được Silhouette (dữ liệu quá ít hoặc K không phù hợp). Sẽ dùng K thủ công.")
else:
    st.info("Bạn đang chọn K thủ công.")

k_final = best_k if (auto_pick and best_k is not None) else int(chosen_k)

st.subheader("3) Kết quả phân cụm")

kmeans = KMeans(n_clusters=k_final, random_state=42, n_init="auto")
labels = kmeans.fit_predict(X_scaled)

work_result = work.copy()
work_result["cluster"] = labels

# Merge cluster back to original df (best-effort)
# If we dropped rows, join by index
df_out = df.copy()
df_out["__row_index__"] = np.arange(len(df_out))
work_result["__row_index__"] = work_result.index.values
df_out = df_out.merge(work_result[["__row_index__", "cluster"]], on="__row_index__", how="left").drop(columns=["__row_index__"])

# Show size chart
plot_cluster_sizes(labels)

# Cluster profile
profile = work_result.groupby("cluster")[features].mean().round(4)
counts = work_result["cluster"].value_counts().sort_index()
profile.insert(0, "n_customers", counts.values)

st.write("**Bảng đặc trưng trung bình theo cụm")
st.dataframe(profile, use_container_width=True)

# Show sample rows each cluster
st.write("**Mẫu khách hàng theo từng cụm:**")
show_cols = ([id_col] if id_col and id_col in df_out.columns else []) + features + ["cluster"]
show_cols = [c for c in show_cols if c in df_out.columns]

tabs = st.tabs([f"Cluster {i}" for i in sorted(work_result["cluster"].unique())])
for i, t in enumerate(tabs):
    with t:
        sample = df_out[df_out["cluster"] == i][show_cols].head(20)
        st.dataframe(sample, use_container_width=True)


