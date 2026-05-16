"""
Olist 셀러의 성과를 분석하기 위해 여러 데이터셋을 결합하고 고도화된 집계 지표를 생성하는 스크립트입니다.

주요 기능:
- 주문, 아이템, 상품, 리뷰, 셀러 데이터 통합
- 셀러별 월평균 매출, 리뷰 점수, 배송 지표 등 산출
- 매출 상위 20%와 하위 20% 셀러 간의 지표 비교 분석
- 주요 지표와 매출 간의 상관관계 분석 및 지역 분포 확인
"""

import pandas as pd
import numpy as np
import os

DATA_DIR = 'Project1/data'

def main():
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
    orders['order_delivered_carrier_date'] = pd.to_datetime(orders['order_delivered_carrier_date'])
    
    # 1. 베이스 합치기: order_items + orders
    df = order_items.merge(orders, on='order_id', how='left')
    
    # 2. + products (사진 수, 설명 길이 추가)
    prod_cols = ['product_id', 'product_category_name', 'product_photos_qty', 'product_description_lenght']
    df = df.merge(products[prod_cols], on='product_id', how='left')
    
    # 3. + reviews
    # 주문 단위로 리뷰 정보 요약
    order_reviews = reviews.groupby('order_id').agg(
        review_score=('review_score', 'mean'),
        min_review_score=('review_score', 'min'),
        review_count=('review_id', 'count')
    ).reset_index()
    # 1~2점 리뷰 여부 (가장 낮은 점수가 1,2점인 경우)
    order_reviews['is_low_review'] = (order_reviews['min_review_score'] <= 2).astype(int)
    
    df = df.merge(order_reviews, on='order_id', how='left')
    
    # 4. + sellers
    df = df.merge(sellers[['seller_id', 'seller_state']], on='seller_id', how='left')
    
    print("지표 집계 중...")
    # Delivered 주문에 대한 처리시간 계산
    delivered_df = df[df['order_status'] == 'delivered'].copy()
    delivered_df['processing_days'] = (delivered_df['order_delivered_carrier_date'] - delivered_df['order_purchase_timestamp']).dt.total_seconds() / (24 * 3600)
    delivered_df['delivery_days'] = (delivered_df['order_delivered_customer_date'] - delivered_df['order_purchase_timestamp']).dt.total_seconds() / (24 * 3600)
    delivered_df['is_on_time'] = (delivered_df['order_delivered_customer_date'] <= delivered_df['order_estimated_delivery_date']).astype(int)
    
    # Seller 레벨 집계
    seller_summary = df.groupby('seller_id').agg(
        total_orders=('order_id', 'nunique'),
        total_sales=('price', 'sum'),
        avg_price=('price', 'mean'),
        category_count=('product_category_name', 'nunique'),
        avg_photos_qty=('product_photos_qty', 'mean'),
        avg_description_length=('product_description_lenght', 'mean'),
        first_sale=('order_purchase_timestamp', 'min'),
        last_sale=('order_purchase_timestamp', 'max'),
        seller_state=('seller_state', 'first'),
        total_reviews=('review_count', 'sum'),
        low_reviews=('is_low_review', 'sum')
    ).reset_index()
    
    # 활성 개월 수 및 월평균 매출
    def calc_months(row):
        start = row['first_sale']
        end = row['last_sale']
        return max(1, (end.year - start.year) * 12 + (end.month - start.month) + 1)
    
    seller_summary['active_months'] = seller_summary.apply(calc_months, axis=1)
    seller_summary['monthly_avg_sales'] = seller_summary['total_sales'] / seller_summary['active_months']
    
    # low_review_rate
    seller_summary['low_review_rate'] = seller_summary['low_reviews'] / seller_summary['total_reviews'].replace(0, np.nan)
    
    # 배송비 비율
    df['ind_freight_ratio'] = df['freight_value'] / df['price']
    df['ind_freight_ratio'] = df['ind_freight_ratio'].replace([np.inf, -np.inf], np.nan)
    f_ratio = df.groupby('seller_id')['ind_freight_ratio'].mean().reset_index()
    seller_summary = seller_summary.merge(f_ratio.rename(columns={'ind_freight_ratio': 'freight_ratio'}), on='seller_id', how='left')
    
    # 리뷰 관련 추가 지표 (개별 리뷰 데이터 직접 조인하여 std 등 정확히 계산)
    seller_item_reviews = order_items[['seller_id', 'order_id']].merge(reviews[['order_id', 'review_score']], on='order_id', how='inner')
    review_stats = seller_item_reviews.groupby('seller_id').agg(
        avg_review_score=('review_score', 'mean'),
        review_std=('review_score', 'std')
    ).reset_index()
    # std는 리뷰가 1개인 경우 NaN이 됨 -> fillna(0) 처리
    review_stats['review_std'] = review_stats['review_std'].fillna(0)
    seller_summary = seller_summary.merge(review_stats, on='seller_id', how='left')
    
    # 배송 관련 지표
    delivery_metrics = delivered_df.groupby('seller_id').agg(
        on_time_rate=('is_on_time', 'mean'),
        avg_delivery_days=('delivery_days', 'mean'),
        avg_seller_processing_days=('processing_days', 'mean')
    ).reset_index()
    seller_summary = seller_summary.merge(delivery_metrics, on='seller_id', how='left')
    
    print("\n--- 분석 결과 (전체 셀러 기준) ---")
    
    # Top / Bottom 분류
    q_low = seller_summary['monthly_avg_sales'].quantile(0.2)
    q_high = seller_summary['monthly_avg_sales'].quantile(0.8)
    
    top_df = seller_summary[seller_summary['monthly_avg_sales'] >= q_high]
    bot_df = seller_summary[seller_summary['monthly_avg_sales'] <= q_low]
    
    # 비교 지표 리스트
    metrics = [
        'monthly_avg_sales', 'avg_price', 'category_count', 'freight_ratio', 
        'avg_photos_qty', 'avg_description_length', 
        'avg_review_score', 'review_std', 'low_review_rate', 
        'avg_seller_processing_days', 'avg_delivery_days', 'on_time_rate'
    ]
    
    top_mean = top_df[metrics].mean()
    bot_mean = bot_df[metrics].mean()
    
    comp_df = pd.DataFrame({'Top 20%': top_mean, 'Bottom 20%': bot_mean})
    # 차이 = Top - Bottom
    comp_df['차이'] = comp_df['Top 20%'] - comp_df['Bottom 20%']
    comp_df['절대차이'] = comp_df['차이'].abs()
    comp_df = comp_df.sort_values('절대차이', ascending=False).drop(columns=['절대차이'])
    
    print("\n[지표 비교]")
    print(comp_df.to_markdown())
    
    # 지역 분포
    print("\n[상위 20% 셀러 지역 분포 Top 5]")
    print(top_df['seller_state'].value_counts().head(5).to_markdown())
    
    print("\n[하위 20% 셀러 지역 분포 Top 5]")
    print(bot_df['seller_state'].value_counts().head(5).to_markdown())
    
    # 상관관계
    corr_matrix = seller_summary[metrics].corr()
    sales_corr = corr_matrix['monthly_avg_sales'].drop('monthly_avg_sales').to_frame('Correlation')
    sales_corr['Abs_Corr'] = sales_corr['Correlation'].abs()
    sales_corr = sales_corr.sort_values('Abs_Corr', ascending=False).drop(columns=['Abs_Corr'])
    
    print("\n[월평균 매출과의 상관계수 랭킹]")
    print(sales_corr.to_markdown())

if __name__ == "__main__":
    main()
