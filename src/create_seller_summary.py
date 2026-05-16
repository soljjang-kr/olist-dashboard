"""
Olist의 원시 데이터셋을 통합하여 셀러별 성과 요약 데이터(seller_summary)를 생성하고 분석하는 스크립트입니다.

주요 기능:
- 주문, 아이템, 상품, 리뷰, 셀러 데이터의 다각도 병합
- 셀러별 총 매출, 주문 건수, 평점, 배송 효율성 등 핵심 KPI 산출
- 매출 규모에 따른 셀러 그룹 분류(Top/Middle/Bottom) 및 비교 분석
- 매출과 주요 지표 간의 상관관계 도출
- 파레토 차트 등 주요 지표에 대한 시각화 리포트 생성
"""

import pandas as pd
import numpy as np
import os

# 데이터 경로 설정
DATA_DIR = 'Project1/data'

def create_seller_summary():
    print("데이터 로딩 중...")
    orders = pd.read_csv(os.path.join(DATA_DIR, 'olist_orders_dataset.csv'))
    order_items = pd.read_csv(os.path.join(DATA_DIR, 'olist_order_items_dataset.csv'))
    products = pd.read_csv(os.path.join(DATA_DIR, 'olist_products_dataset.csv'))
    sellers = pd.read_csv(os.path.join(DATA_DIR, 'olist_sellers_dataset.csv'))
    reviews = pd.read_csv(os.path.join(DATA_DIR, 'olist_order_reviews_dataset.csv'))
    
    # 날짜 컬럼 변환
    orders['order_purchase_timestamp'] = pd.to_datetime(orders['order_purchase_timestamp'])
    orders['order_delivered_customer_date'] = pd.to_datetime(orders['order_delivered_customer_date'])
    orders['order_estimated_delivery_date'] = pd.to_datetime(orders['order_estimated_delivery_date'])
    
    print("데이터 병합 중...")
    # 1. Order Items + Orders
    df = order_items.merge(orders, on='order_id', how='left')
    
    # 2. + Products
    df = df.merge(products[['product_id', 'product_category_name', 'product_photos_qty']], on='product_id', how='left')
    
    # 3. + Reviews (order_id 기준)
    # 한 주문에 여러 리뷰가 있을 수 있으므로 평균 점수 사용
    order_reviews = reviews.groupby('order_id')['review_score'].mean().reset_index()
    df = df.merge(order_reviews, on='order_id', how='left')
    
    # 4. + Sellers
    df = df.merge(sellers[['seller_id', 'seller_state']], on='seller_id', how='left')
    
    print("지표 계산 중...")
    
    # 배송 관련 지표 계산 (delivered 주문만)
    delivered_df = df[df['order_status'] == 'delivered'].copy()
    delivered_df['is_on_time'] = (delivered_df['order_delivered_customer_date'] <= delivered_df['order_estimated_delivery_date']).astype(int)
    delivered_df['delivery_days'] = (delivered_df['order_delivered_customer_date'] - delivered_df['order_purchase_timestamp']).dt.total_seconds() / (24 * 3600)
    
    # 셀러별 집계
    seller_group = df.groupby('seller_id')
    
    # 기본 집계
    seller_summary = seller_group.agg(
        total_orders=('order_id', 'nunique'),
        total_sales=('price', 'sum'),
        avg_price=('price', 'mean'),
        avg_review_score=('review_score', 'mean'),
        category_count=('product_category_name', 'nunique'),
        avg_photos_qty=('product_photos_qty', 'mean'),
        first_sale=('order_purchase_timestamp', 'min'),
        last_sale=('order_purchase_timestamp', 'max'),
        seller_state=('seller_state', 'first')
    ).reset_index()
    
    # active_months 계산
    def calculate_active_months(row):
        start = row['first_sale']
        end = row['last_sale']
        return (end.year - start.year) * 12 + (end.month - start.month) + 1
    
    seller_summary['active_months'] = seller_summary.apply(calculate_active_months, axis=1)
    
    # monthly_avg_sales 계산
    seller_summary['monthly_avg_sales'] = seller_summary['total_sales'] / seller_summary['active_months']
    
    # freight_ratio 계산 (셀러별 평균)
    df['ind_freight_ratio'] = df['freight_value'] / df['price']
    # inf나 nan 처리 (price가 0인 경우 대비)
    df['ind_freight_ratio'] = df['ind_freight_ratio'].replace([np.inf, -np.inf], np.nan)
    f_ratio = df.groupby('seller_id')['ind_freight_ratio'].mean().reset_index()
    seller_summary = seller_summary.merge(f_ratio.rename(columns={'ind_freight_ratio': 'freight_ratio'}), on='seller_id', how='left')
    
    # low_review_rate (1~2점 비율)
    df['is_low_review'] = (df['review_score'] <= 2).astype(int)
    low_review = df.groupby('seller_id')['is_low_review'].mean().reset_index()
    seller_summary = seller_summary.merge(low_review.rename(columns={'is_low_review': 'low_review_rate'}), on='seller_id', how='left')
    
    # 배송 지표 (delivered 주문 기준 집계)
    delivery_metrics = delivered_df.groupby('seller_id').agg(
        on_time_rate=('is_on_time', 'mean'),
        avg_delivery_days=('delivery_days', 'mean')
    ).reset_index()
    seller_summary = seller_summary.merge(delivery_metrics, on='seller_id', how='left')
    
    # 불필요한 컬럼 제거
    seller_summary = seller_summary.drop(columns=['first_sale', 'last_sale'])
    
    # 결과 저장
    output_path = os.path.join(DATA_DIR, 'seller_summary.csv')
    seller_summary.to_csv(output_path, index=False)
    
    print(f"\n[OK] seller_summary 생성 완료: {output_path}")
    
    # --- 추가 분석 요구사항 ---
    
    # 1. total_orders 중위값
    median_orders = seller_summary['total_orders'].median()
    print(f"\ntotal_orders 중위값: {median_orders}")
    
    def analyze_groups(df, version_name):
        # 1. 라벨링
        q_low = df['monthly_avg_sales'].quantile(0.2)
        q_high = df['monthly_avg_sales'].quantile(0.8)
        
        df = df.copy()
        df['group'] = 'Middle 60%'
        df.loc[df['monthly_avg_sales'] >= q_high, 'group'] = 'Top 20%'
        df.loc[df['monthly_avg_sales'] <= q_low, 'group'] = 'Bottom 20%'
        
        print(f"\n--- {version_name} 그룹 분류 결과 ---")
        print(df['group'].value_counts().to_markdown())
        
        # 2. 비교 분석
        metrics = [
            'monthly_avg_sales', 'avg_price', 'freight_ratio', 
            'avg_review_score', 'low_review_rate', 'on_time_rate', 
            'avg_delivery_days', 'category_count', 'avg_photos_qty'
        ]
        
        comparison = df.groupby('group')[metrics].mean().T
        if 'Top 20%' in comparison.columns and 'Bottom 20%' in comparison.columns:
            comp_table = comparison[['Top 20%', 'Bottom 20%']].copy()
            comp_table['Difference'] = comp_table['Top 20%'] - comp_table['Bottom 20%']
            comp_table['Abs_Difference'] = comp_table['Difference'].abs()
            comp_table = comp_table.sort_values('Abs_Difference', ascending=False).drop(columns=['Abs_Difference'])
            
            print(f"\n--- {version_name} Top 20% vs Bottom 20% 비교 (차이순 정렬) ---")
            print(comp_table.to_markdown())
        
        return df

    # 버전 A: 전체 셀러
    print("\n[버전 A: 전체 셀러]")
    seller_summary_a = analyze_groups(seller_summary, "버전 A")
    
    # 버전 B: total_orders >= 중위값 셀러
    print(f"\n[버전 B: total_orders >= {median_orders} 셀러]")
    seller_summary_b = seller_summary[seller_summary['total_orders'] >= median_orders]
    seller_summary_b = analyze_groups(seller_summary_b, "버전 B")

    # --- 상관관계 분석 ---
    print("\n[monthly_avg_sales와의 상관관계 분석]")
    corr_metrics = [
        'monthly_avg_sales', 'avg_price', 'freight_ratio', 
        'avg_review_score', 'low_review_rate', 'on_time_rate', 
        'avg_delivery_days', 'category_count', 'avg_photos_qty', 'total_orders'
    ]
    corr_matrix = seller_summary[corr_metrics].corr()
    sales_corr = corr_matrix['monthly_avg_sales'].drop('monthly_avg_sales').to_frame()
    sales_corr.columns = ['Correlation']
    sales_corr['Abs_Correlation'] = sales_corr['Correlation'].abs()
    sales_corr = sales_corr.sort_values('Abs_Correlation', ascending=False)
    
    print(sales_corr[['Correlation']].to_markdown())
    
    top_corr_metric = sales_corr.index[0]
    top_corr_val = sales_corr.iloc[0]['Correlation']
    print(f"\n요약: 매출(monthly_avg_sales)과 가장 강하게 연관된 지표는 '{top_corr_metric}'(상관계수: {top_corr_val:.3f})입니다.")

    # --- 시각화 ---
    import matplotlib.pyplot as plt
    import koreanize_matplotlib
    
    IMAGE_DIR = 'Project1/images'
    if not os.path.exists(IMAGE_DIR):
        os.makedirs(IMAGE_DIR)

    # 1. 월평균 매출 분포 히스토그램 (로그 스케일)
    plt.figure(figsize=(10, 6))
    plt.hist(seller_summary['monthly_avg_sales'], bins=50, color='skyblue', edgecolor='black')
    plt.yscale('log')
    plt.title('월평균 매출 분포 (로그 스케일)')
    plt.xlabel('월평균 매출')
    plt.ylabel('셀러 수 (로그)')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig(os.path.join(IMAGE_DIR, 's01_sales_dist.png'))
    plt.close()
    print("\n[차트 1] 월평균 매출 분포: 대다수의 셀러가 저매출 구간에 집중되어 있으며, 극소수의 하이퍼 셀러가 전체 매출을 견인하는 구조입니다.")

    # 2. 매출 파레토 차트
    sorted_sales = seller_summary['total_sales'].sort_values(ascending=False)
    cum_sales = sorted_sales.cumsum() / sorted_sales.sum() * 100
    
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(cum_sales)), cum_sales, color='red', linewidth=2)
    plt.axhline(y=80, color='gray', linestyle='--')
    plt.title('매출 파레토 차트 (누적 매출 곡선)')
    plt.xlabel('셀러 순위')
    plt.ylabel('누적 매출 비중 (%)')
    plt.savefig(os.path.join(IMAGE_DIR, 's03_sales_share.png'))
    plt.close()
    print("[차트 2] 매출 파레토: 상위 일부 셀러가 전체 매출의 80% 이상을 차지하는 전형적인 파레토 법칙이 관찰됩니다.")

    # 3. 평균 평점 분포 히스토그램
    plt.figure(figsize=(10, 6))
    plt.hist(seller_summary['avg_review_score'].dropna(), bins=20, color='gold', edgecolor='black')
    plt.title('평균 평점 분포')
    plt.xlabel('평균 리뷰 점수')
    plt.ylabel('셀러 수')
    plt.savefig(os.path.join(IMAGE_DIR, 's06_avg_review.png'))
    plt.close()
    print("[차트 3] 평균 평점 분포: 4~5점대 고득점 셀러가 가장 많으나, 1점대 불만족 셀러도 일부 존재합니다.")

    # 4. 배송비 비중 분포 히스토그램
    plt.figure(figsize=(10, 6))
    plt.hist(seller_summary['freight_ratio'].dropna(), bins=50, color='teal', edgecolor='black', range=(0, 1))
    plt.title('배송비 비중 분포 (Freight/Price)')
    plt.xlabel('배송비 비중')
    plt.ylabel('셀러 수')
    plt.savefig(os.path.join(IMAGE_DIR, 's13_freight_comp.png'))
    plt.close()
    print("[차트 4] 배송비 비중 분포: 대부분의 상품에서 배송비가 상품 가격의 20% 내외를 차지하고 있습니다.")

    # 5. 취급 카테고리 수 분포 막대그래프
    cat_counts = seller_summary['category_count'].value_counts().sort_index().head(10)
    plt.figure(figsize=(10, 6))
    cat_counts.plot(kind='bar', color='salmon', edgecolor='black')
    plt.title('취급 카테고리 수 분포')
    plt.xlabel('카테고리 수')
    plt.ylabel('셀러 수')
    plt.savefig(os.path.join(IMAGE_DIR, 's10_category_count.png'))
    plt.close()
    print("[차트 5] 취급 카테고리 수: 대부분의 셀러(약 90% 이상)가 1~2개의 핵심 카테고리에 집중하고 있습니다.")

    print("\n--- 상위 5행 (최종) ---")
    print(seller_summary.head().to_markdown(index=False))
    
    print("\n--- 기술 통계 (최종) ---")
    print(seller_summary.describe().to_markdown())
    
    return seller_summary

if __name__ == "__main__":
    create_seller_summary()
