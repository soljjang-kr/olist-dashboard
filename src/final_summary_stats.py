"""
셀러 요약 데이터를 바탕으로 상위/하위 그룹 간의 성과 지표 차이를 요약 통계로 산출하는 스크립트입니다.

주요 기능:
- 매출 상위 20%와 하위 20% 그룹 간의 평균 지표 비교
- 전체 셀러 그룹(버전 A)과 최소 주문량 기준 필터링 그룹(버전 B)별 분석 수행
- 그룹 간 비율(Ratio) 및 증감률(Diff_Pct) 계산
"""

import pandas as pd

df = pd.read_csv('Project1/data/seller_summary.csv')

def get_comp(data):
    q_low = data['monthly_avg_sales'].quantile(0.2)
    q_high = data['monthly_avg_sales'].quantile(0.8)
    top = data[data['monthly_avg_sales'] >= q_high]
    bot = data[data['monthly_avg_sales'] <= q_low]
    metrics = ['monthly_avg_sales', 'avg_price', 'category_count', 'freight_ratio', 'avg_review_score', 'on_time_rate', 'avg_delivery_days']
    res = pd.concat([top[metrics].mean(), bot[metrics].mean()], axis=1, keys=['Top', 'Bottom'])
    res['Ratio'] = res['Top'] / res['Bottom']
    res['Diff_Pct'] = (res['Top'] - res['Bottom']) / res['Bottom'] * 100
    return res

print("--- Version A (All Sellers) ---")
print(get_comp(df))

print("\n--- Version B (Orders >= 6) ---")
print(get_comp(df[df['total_orders'] >= 6]))
