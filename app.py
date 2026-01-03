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
    """Kết nối và lấy dữ liệu đã làm sạch từ SQL Server."""
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


# ==========================================
# 3. GIAO DIỆN STREAMLIT
# ==========================================
st.set_page_config(page_title="Customer Segmentation Tool", layout="wide")
st.title(" Định hướng Marketing dựa trên lịch sử vận chuyển bằng phương pháp phân cụm với thuật toán K-Means ")

# --- SIDEBAR---
with st.sidebar:
    st.header("1. Dữ liệu nguồn")
    if st.button(" Tải dữ liệu từ SQL", type="primary"):
        st.session_state.load_trigger = True

    # Logic tải dữ liệu
    if 'df_sql' not in st.session_state:
        st.session_state.df_sql = None

    if st.session_state.get('load_trigger'):
        with st.spinner("Đang kết nối SQL Server..."):
            df_result, error = load_data_from_sql()
            if error:
                st.error(f"Lỗi: {error}")
            else:
                st.session_state.df_sql = df_result
                st.success(f"Đã tải {len(df_result)} khách hàng.")
                st.session_state.load_trigger = False

    # Chỉ hiện cấu hình phân cụm khi đã có dữ liệu
    if st.session_state.df_sql is not None:
        st.divider()
        st.header("2. Cấu hình K-Means")

        # Chọn K
        k_num = st.slider("Số lượng cụm (K)", 2, 8, 4)

        # Chọn thuộc tính
        df = st.session_state.df_sql
        all_numeric_cols = [c for c in df.columns if
                            c not in ['C_ID', 'Cluster'] and pd.api.types.is_numeric_dtype(df[c])]

        st.write("**Chọn thuộc tính phân tích:**")
        selected_features = st.multiselect(
            "Features",
            options=all_numeric_cols,
            default=all_numeric_cols,
            label_visibility="collapsed"
        )

        st.divider()
        # Nút chạy nằm cuối Sidebar
        run_btn = st.button(" Tiến hành Phân cụm", type="primary")

# --- MÀN HÌNH CHÍNH ---

# 1. Kiểm tra dữ liệu
if st.session_state.df_sql is None:
    st.stop()

# 2. Xử lý thuật toán khi bấm nút
if 'run_btn' in locals() and run_btn:
    if not selected_features:
        st.error("Vui lòng chọn ít nhất 1 thuộc tính!")
    else:
        with st.spinner("Đang chạy thuật toán AI..."):
            df = st.session_state.df_sql
            # Prepare Data
            X = df[selected_features].values
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            # Run KMeans
            kmeans = KMeans(n_clusters=k_num, random_state=42, n_init="auto")
            labels = kmeans.fit_predict(X_scaled)

            # Save Result to Session State
            df_out = df.copy()
            df_out['Cluster'] = labels.astype(str)

            st.session_state.clustered_df = df_out
            st.session_state.features = selected_features
            st.rerun()  # Load lại trang để hiển thị kết quả

# 3. Hiển thị kết quả
if 'clustered_df' in st.session_state:
    df_out = st.session_state.clustered_df
    feats = st.session_state.features

    # --- PHẦN A: DASHBOARD TỔNG QUAN ---
    st.subheader(" Tổng quan phân cụm")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.write("**Bản đồ phân bố (PCA 2D)**")
        pca = PCA(n_components=2)
        components = pca.fit_transform(StandardScaler().fit_transform(df_out[feats]))
        df_out['PCA1'] = components[:, 0]
        df_out['PCA2'] = components[:, 1]

        fig = px.scatter(
            df_out, x='PCA1', y='PCA2', color='Cluster',
            hover_data=['C_ID'] + feats[:3],
            title=f"Phân bố {len(df_out['Cluster'].unique())} nhóm khách hàng"
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

    # ---  DANH SÁCH CHI TIẾT ---
    st.subheader("Danh sách chi tiết từng nhóm")

    c1, c2, c3 = st.columns([1, 2, 1])
    with c1:
        # Bộ lọc
        unique_clusters = sorted(df_out['Cluster'].unique())
        cluster_select = st.selectbox("Chọn cụm (Cluster) để xem:", ["Tất cả"] + list(unique_clusters))

    # Logic lọc
    if cluster_select != "Tất cả":
        display_df = df_out[df_out['Cluster'] == cluster_select]
    else:
        display_df = df_out

    # Hiển thị bảng
    show_cols = ['C_ID', 'Cluster'] + feats
    st.dataframe(display_df[show_cols], use_container_width=True, height=400)

    # Nút Download
    csv = display_df[show_cols].to_csv(index=False).encode('utf-8')
    st.download_button(
        label=f"Tải danh sách ({len(display_df)} khách hàng)",
        data=csv,
        file_name=f"Danh_sach_Cluster_{cluster_select}.csv",
        mime="text/csv",
    )