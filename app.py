import pandas as pd
import streamlit as st
import urllib
from sqlalchemy import create_engine
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import plotly.express as px

# ==========================================
# 1. CẤU HÌNH KẾT NỐI (Đã cập nhật User/Pass của bạn)
# ==========================================
DB_SERVER = 'LONGNGUYEN\MSSQLSERVER02'
DB_DATABASE = 'Logistics'
DB_USER = 'sa'
DB_PASS = '123'
DB_TABLE = 'FactCustomerBehavior'


# ==========================================
# 2. HÀM KẾT NỐI SQL SERVER
# ==========================================
@st.cache_data(ttl=600)
def load_data_from_sql():
    """Kết nối và lấy dữ liệu từ SQL Server."""
    try:
        # Tạo chuỗi kết nối an toàn
        params = urllib.parse.quote_plus(
            f"DRIVER={{ODBC Driver 17 for SQL Server}};"
            f"SERVER={DB_SERVER};"
            f"DATABASE={DB_DATABASE};"
            f"UID={DB_USER};"
            f"PWD={DB_PASS}"
        )
        engine = create_engine(f"mssql+pyodbc:///?odbc_connect={params}")

        # Query lấy toàn bộ dữ liệu từ bảng Fact
        query = f"SELECT * FROM {DB_TABLE}"
        df = pd.read_sql(query, engine)
        return df, None
    except Exception as e:
        return None, str(e)


def preprocess_df(df):
    """Tiền xử lý chung cho cả CSV và SQL"""
    # 1. Tìm cột ID (nếu có) để không dùng phân cụm
    id_col = None
    candidates = ["C_ID", "customer_id", "id", "ID", "CustomerID"]
    for c in candidates:
        if c in df.columns:
            id_col = c
            break

    # 2. Lọc các cột số
    numeric_cols = [c for c in df.columns if c != id_col and pd.api.types.is_numeric_dtype(df[c])]

    # 3. Xử lý NaN
    df_clean = df.copy()
    for col in numeric_cols:
        if df_clean[col].isnull().sum() > 0:
            df_clean[col] = df_clean[col].fillna(df_clean[col].median())

    return df_clean, id_col, numeric_cols


# ==========================================
# 3. GIAO DIỆN STREAMLIT
# ==========================================
st.set_page_config(page_title="Customer Segmentation Tool", layout="wide")
st.title(" Định hướng Marketing dựa trên lịch sử vận chuyển bằng phương pháp phân cụm")

# --- SIDEBAR: ĐIỀU KHIỂN ---
with st.sidebar:
    st.header("Nguồn dữ liệu")

    # TÙY CHỌN: SQL HAY CSV?
    data_source = st.radio("Chọn nguồn dữ liệu:", (" SQL Server", " Upload File CSV"))

    # LOGIC LOAD DỮ LIỆU
    if 'df_loaded' not in st.session_state:
        st.session_state.df_loaded = None

    # === TRƯỜNG HỢP 1: SQL SERVER ===
    if data_source == " SQL Server":
        if st.button("Kết nối & Tải dữ liệu", type="primary"):
            with st.spinner("Đang kết nối SQL Server..."):
                df_result, error = load_data_from_sql()
                if error:
                    st.error(f"Lỗi SQL: {error}")
                else:
                    st.session_state.df_loaded = df_result
                    st.success(f"Đã tải {len(df_result)} dòng từ SQL.")

    # === TRƯỜNG HỢP 2: CSV FILE ===
    else:
        uploaded_file = st.file_uploader("Chọn file CSV", type=["csv"])
        if uploaded_file is not None:
            # Đọc file ngay khi upload
            try:
                df_csv = pd.read_csv(uploaded_file)
                st.session_state.df_loaded = df_csv
                st.success(f"Đã đọc file CSV: {len(df_csv)} dòng.")
            except Exception as e:
                st.error(f"Lỗi đọc file: {e}")

    # CHỈ HIỆN CẤU HÌNH KHI ĐÃ CÓ DATA
    if st.session_state.df_loaded is not None:
        st.divider()
        st.header(" Cấu hình K-Means")

        # Tiền xử lý sơ bộ để lấy danh sách cột
        df_raw = st.session_state.df_loaded
        _, id_col_detect, all_numeric = preprocess_df(df_raw)

        # Chọn K
        k_num = st.slider("Số lượng cụm (K)", 2, 10, 4)

        # Chọn Features
        st.write("**Chọn thuộc tính phân tích:**")
        selected_features = st.multiselect(
            "Features",
            options=all_numeric,
            default=all_numeric[:5] if len(all_numeric) > 5 else all_numeric,
            label_visibility="collapsed"
        )

        st.divider()
        run_btn = st.button(" Tiến hành Phân cụm", type="primary")

# --- MÀN HÌNH CHÍNH ---

# Kiểm tra dữ liệu
if st.session_state.df_loaded is None:
    if data_source == " SQL Server":
        st.info(" Bấm nút **'Kết nối & Tải dữ liệu'** bên trái để bắt đầu.")
    else:
        st.info(" Vui lòng **Upload file CSV** bên trái để bắt đầu.")
    st.stop()

# Xử lý Logic Phân Cụm
if 'run_btn' in locals() and run_btn:
    if not selected_features:
        st.error("Vui lòng chọn ít nhất 1 thuộc tính!")
    else:
        with st.spinner("Đang xử lý dữ liệu & chạy AI..."):
            # 1. Lấy dữ liệu & Xử lý NaN
            df_work, id_col, _ = preprocess_df(st.session_state.df_loaded)

            # 2. Chuẩn hóa
            X = df_work[selected_features].values
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            # 3. K-Means
            kmeans = KMeans(n_clusters=k_num, random_state=42, n_init="auto")
            labels = kmeans.fit_predict(X_scaled)

            # 4. Lưu kết quả
            df_out = df_work.copy()
            df_out['Cluster'] = labels.astype(str)

            st.session_state.clustered_df = df_out
            st.session_state.features = selected_features
            st.session_state.id_col = id_col
            st.rerun()

# Hiển thị kết quả (Dashboard)
if 'clustered_df' in st.session_state:
    df_out = st.session_state.clustered_df
    feats = st.session_state.features
    id_c = st.session_state.id_col

    # Cột ID để hiển thị (Nếu không tìm thấy ID thì dùng index)
    id_show = [id_c] if id_c else []

    # --- PHẦN I: DASHBOARD ---
    st.subheader(" Tổng quan phân cụm")
    col1, col2 = st.columns([2, 1])

    with col1:
        st.write("**Bản đồ phân bố (PCA 2D)**")
        # PCA
        pca = PCA(n_components=2)
        X_pca = StandardScaler().fit_transform(df_out[feats])
        components = pca.fit_transform(X_pca)
        df_out['PCA1'] = components[:, 0]
        df_out['PCA2'] = components[:, 1]

        fig = px.scatter(
            df_out, x='PCA1', y='PCA2', color='Cluster',
            hover_data=id_show + feats[:3],
            title=f"Phân bố {len(df_out['Cluster'].unique())} cụm khách hàng"
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.write("**Tỷ lệ khách hàng**")
        counts = df_out['Cluster'].value_counts()
        fig_pie = px.pie(values=counts.values, names=counts.index, hole=0.4)
        st.plotly_chart(fig_pie, use_container_width=True)

    # Cluster Profile
    st.write("**Đặc điểm hành vi trung bình (Cluster Profile)**")
    profile = df_out.groupby("Cluster")[feats].mean()
    st.dataframe(
        profile.style.format("{:.2f}").background_gradient(cmap="Blues", axis=0),
        use_container_width=True
    )

    st.divider()

    # --- PHẦN II: CHI TIẾT ---
    st.subheader("Danh sách chi tiết từng nhóm")

    c1, c2 = st.columns([1, 3])
    with c1:
        unique_clusters = sorted(df_out['Cluster'].unique())
        cluster_select = st.selectbox("Chọn cụm (Cluster):", ["Tất cả"] + list(unique_clusters))

    if cluster_select != "Tất cả":
        display_df = df_out[df_out['Cluster'] == cluster_select]
    else:
        display_df = df_out

    show_cols = id_show + ['Cluster'] + feats
    show_cols = [c for c in show_cols if c in display_df.columns]

    st.dataframe(display_df[show_cols], use_container_width=True, height=400)

    csv = display_df[show_cols].to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Tải danh sách (CSV)",
        data=csv,
        file_name=f"List_Cluster_{cluster_select}.csv",
        mime="text/csv",
    )