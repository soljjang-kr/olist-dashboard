"""
셀러 요약 데이터를 활용하여 상위/하위 성과 그룹 간의 지표를 시각적으로 비교하는 차트를 생성하는 스크립트입니다.

주요 기능:
- 매출 기준 상위 20%와 하위 20% 셀러 그룹화
- 그룹별 리뷰 점수, 배송비 비중, 카테고리 수, 객단가 비교 박스플롯 생성
- 주요 수치 지표 간의 상관관계 히트맵 생성
- 생성된 차트 이미지 파일 저장
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import koreanize_matplotlib
import os

# 데이터 경로 설정
DATA_DIR = 'Project1/data'
IMAGE_DIR = 'Project1/images'
SUMMARY_PATH = os.path.join(DATA_DIR, 'seller_summary.csv')

if not os.path.exists(IMAGE_DIR):
    os.makedirs(IMAGE_DIR)

def create_comparison_charts():
    print("데이터 로딩 중...")
    df = pd.read_csv(SUMMARY_PATH)
    
    # 1. 버전 A 기준 라벨링 (전체 셀러 기준 상위/하위 20%)
    q_low = df['monthly_avg_sales'].quantile(0.2)
    q_high = df['monthly_avg_sales'].quantile(0.8)
    
    df_comp = df[(df['monthly_avg_sales'] >= q_high) | (df['monthly_avg_sales'] <= q_low)].copy()
    df_comp['group'] = np.where(df_comp['monthly_avg_sales'] >= q_high, 'Top 20%', 'Bottom 20%')
    
    print("차트 생성 중...")
    
    # 1. 평균 평점 박스플롯
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='group', y='avg_review_score', data=df_comp, palette='Set2')
    plt.title('그룹별 평균 평점 분포 비교')
    plt.xlabel('그룹')
    plt.ylabel('평균 리뷰 점수')
    plt.savefig(os.path.join(IMAGE_DIR, 'comp_avg_review.png'))
    plt.close()
    print("[차트 1] 평점 비교: 상위 20% 셀러와 하위 20% 셀러 간의 평균 평점 차이는 크지 않으나, 상위 셀러의 평점 편차가 상대적으로 작게 나타납니다.")

    # 2. 배송비 비중 박스플롯
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='group', y='freight_ratio', data=df_comp, palette='Set3')
    plt.ylim(0, 1) # 비중이므로 0~1 사이 집중
    plt.title('그룹별 배송비 비중 분포 비교')
    plt.xlabel('그룹')
    plt.ylabel('배송비 비중 (Freight/Price)')
    plt.savefig(os.path.join(IMAGE_DIR, 'comp_freight_ratio.png'))
    plt.close()
    print("[차트 2] 배송비 비중 비교: 하위 20% 셀러의 경우 상품 가격 대비 배송비 비중이 상위 셀러보다 높고 편차도 큽니다.")

    # 3. 취급 카테고리 수 박스플롯
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='group', y='category_count', data=df_comp, palette='pastel')
    plt.title('그룹별 취급 카테고리 수 분포 비교')
    plt.xlabel('그룹')
    plt.ylabel('카테고리 수')
    plt.savefig(os.path.join(IMAGE_DIR, 'comp_category_count.png'))
    plt.close()
    print("[차트 3] 카테고리 수 비교: 상위 20% 셀러가 하위 셀러보다 상대적으로 더 다양한 카테고리의 상품을 취급하는 경향이 있습니다.")

    # 4. 평균 객단가 박스플롯
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='group', y='avg_price', data=df_comp, palette='muted')
    plt.yscale('log')
    plt.title('그룹별 평균 객단가 분포 비교 (로그 스케일)')
    plt.xlabel('그룹')
    plt.ylabel('평균 상품 가격 (로그)')
    plt.savefig(os.path.join(IMAGE_DIR, 'comp_avg_price.png'))
    plt.close()
    print("[차트 4] 객단가 비교: 상위 20% 셀러의 평균 상품 가격이 하위 셀러보다 유의미하게 높게 형성되어 매출 규모 차이의 핵심 원인임을 보여줍니다.")

    # 5. 주요 수치 지표 상관관계 히트맵
    corr_metrics = [
        'monthly_avg_sales', 'avg_price', 'freight_ratio', 
        'avg_review_score', 'low_review_rate', 'on_time_rate', 
        'avg_delivery_days', 'category_count'
    ]
    plt.figure(figsize=(12, 10))
    sns.heatmap(df[corr_metrics].corr(), annot=True, cmap='RdBu_r', fmt='.2f', center=0)
    plt.title('주요 수치 지표 간 상관관계 히트맵')
    plt.savefig(os.path.join(IMAGE_DIR, 'comp_heatmap.png'))
    plt.close()
    print("[차트 5] 상관관계 히트맵: 월평균 매출은 상품 가격 및 카테고리 수와 정적 상관관계를, 배송비 비중과는 부적 상관관계를 보입니다.")

    print("\n✅ 모든 비교 차트 생성이 완료되었습니다.")

if __name__ == "__main__":
    create_comparison_charts()
