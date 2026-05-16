import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from datetime import datetime
import streamlit.components.v1 as components

# Streamlit 페이지 설정
st.set_page_config(page_title="Professional Olist Analysis Dashboard", layout="wide")

# --- 경로 설정 유틸리티 ---
def get_path(relative_path):
    # Project1 폴더 내부에서 실행될 때와 외부에서 실행될 때를 모두 대응
    if os.path.exists(relative_path):
        return relative_path
    # 'Project1/'을 제거한 경로 시도
    alt_path = relative_path.replace('Project1/', '')
    if os.path.exists(alt_path):
        return alt_path
    return relative_path

# --- 데이터 로딩 및 캐싱 ---
@st.cache_data
def load_all_data():
    orders = pd.read_csv(get_path('Project1/data/olist_orders_dataset.csv'))
    customers = pd.read_csv(get_path('Project1/data/olist_customers_dataset.csv'))
    payments = pd.read_csv(get_path('Project1/data/olist_order_payments_dataset.csv'))
    
    orders['order_purchase_timestamp'] = pd.to_datetime(orders['order_purchase_timestamp'])
    df = pd.merge(orders, customers[['customer_id', 'customer_unique_id']], on='customer_id')
    
    order_payments = payments.groupby('order_id')['payment_value'].sum().reset_index()
    df = pd.merge(df, order_payments, on='order_id', how='left')
    df['payment_value'] = df['payment_value'].fillna(0)
    
    return df

def main():
    st.title("🚀 Professional Olist Analysis Dashboard")
    st.markdown("HTML 기반 자기완성형 리포트와 실시간 비즈니스 지표를 통합한 대시보드입니다.")

    with st.spinner('데이터를 분석 중입니다...'):
        df = load_all_data()

    # --- 탭 구성 ---
    tab_report, tab_biz, tab_cohort = st.tabs([
        "📜 전문 분석 리포트 (HTML)", "📈 실시간 핵심 지표", "👥 코호트 리텐션"
    ])

    # --- TAB 1: 전문 분석 리포트 (HTML 렌더링) ---
    with tab_report:
        st.header("Olist 심층 분석 리포트")
        report_path = get_path('Project1/report/eda_report.html')
        try:
            with open(report_path, 'r', encoding='utf-8') as f:
                html_data = f.read()
            # HTML 컴포넌트를 사용하여 렌더링 (scrolling=True로 전체 리포트 확인 가능)
            components.html(html_data, height=800, scrolling=True)
            
            # 다운로드 버튼 제공
            st.download_button(
                label="전체 HTML 리포트 다운로드",
                data=html_data,
                file_name="olist_eda_report.html",
                mime="text/html"
            )
        except FileNotFoundError:
            st.error("리포트 파일을 찾을 수 없습니다. 분석 스크립트를 먼저 실행해 주세요.")

    # --- TAB 2: 핵심 비즈니스 지표 ---
    with tab_biz:
        st.header("실시간 비즈니스 지표 (Filtered)")
        df['date'] = df['order_purchase_timestamp'].dt.date
        df['month_ts'] = df['order_purchase_timestamp'].dt.to_period('M').dt.to_timestamp()
        
        dau = df.groupby('date')['customer_unique_id'].nunique().reset_index(name='DAU')
        mau = df.groupby('month_ts')['customer_unique_id'].nunique().reset_index(name='MAU')
        
        col1, col2 = st.columns(2)
        col1.plotly_chart(px.line(dau, x='date', y='DAU', title="일별 구매자 수 (DAU)"), use_container_width=True)
        col2.plotly_chart(px.line(mau, x='month_ts', y='MAU', title="월별 구매자 수 (MAU)", color_discrete_sequence=['orange']), use_container_width=True)
        
        monthly_rev = df.groupby('month_ts')['payment_value'].sum().reset_index(name='Revenue')
        arppu_df = pd.merge(monthly_rev, mau, on='month_ts')
        arppu_df['ARPPU'] = arppu_df['Revenue'] / arppu_df['MAU']
        st.plotly_chart(px.bar(arppu_df, x='month_ts', y='ARPPU', title="월별 ARPPU (인당 평균 결제액)", text_auto='.1f'), use_container_width=True)

    # --- TAB 3: 코호트 리텐션 ---
    with tab_cohort:
        st.header("월간 코호트 리텐션")
        cohort_data = df.copy()
        cohort_data['order_month'] = cohort_data['order_purchase_timestamp'].dt.to_period('M')
        cohort_data['first_month'] = cohort_data.groupby('customer_unique_id')['order_month'].transform('min')
        cohort_data['cohort_idx'] = (cohort_data['order_month'].dt.year - cohort_data['first_month'].dt.year) * 12 + \
                                    (cohort_data['order_month'].dt.month - cohort_data['first_month'].dt.month)
        
        mode = st.radio("분석 기준", ["고객 수 유지율", "매출액 유지율"], horizontal=True)
        if mode == "고객 수 유지율":
            pivot = cohort_data.groupby(['first_month', 'cohort_idx'])['customer_unique_id'].nunique().reset_index().pivot(index='first_month', columns='cohort_idx', values='customer_unique_id')
            color = 'Blues'
        else:
            pivot = cohort_data.groupby(['first_month', 'cohort_idx'])['payment_value'].sum().reset_index().pivot(index='first_month', columns='cohort_idx', values='payment_value')
            color = 'Reds'
            
        retention = pivot.divide(pivot.iloc[:, 0], axis=0)
        st.plotly_chart(px.imshow(retention, text_auto='.1%', color_continuous_scale=color, y=retention.index.astype(str)), use_container_width=True)

if __name__ == "__main__":
    main()
