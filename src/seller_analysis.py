"""
Olist 셀러 데이터를 성과 중심(매출 기준)으로 그룹화하여 다각도로 분석하고 시각화 리포트를 생성하는 스크립트입니다.

주요 기능:
- 셀러별 매출, 리뷰, 배송, 카테고리 지표 통합 및 요약(seller_summary) 생성
- 매출 기준 상위 20%, 중위 60%, 하위 20% 셀러 그룹 정의
- 그룹별 매출 기여도, 주문 수, 평점, 배송 효율성 비교 시각화(16종 차트)
- 주력 카테고리 분석 및 가격/배송비 구조 비교
- 정제된 집계 데이터를 텍스트 파일로 저장
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import koreanize_matplotlib
import os
from tabulate import tabulate

# 설정
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
IMAGE_DIR = os.path.join(BASE_DIR, "images")
REPORT_DIR = os.path.join(BASE_DIR, "report")

os.makedirs(IMAGE_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

def save_fig(name):
    plt.savefig(os.path.join(IMAGE_DIR, name), bbox_inches='tight')
    plt.close()

# 1. 데이터 로드
print("데이터 로딩 중...")
customers = pd.read_csv(os.path.join(DATA_DIR, "olist_customers_dataset.csv"))
items = pd.read_csv(os.path.join(DATA_DIR, "olist_order_items_dataset.csv"))
payments = pd.read_csv(os.path.join(DATA_DIR, "olist_order_payments_dataset.csv"))
reviews = pd.read_csv(os.path.join(DATA_DIR, "olist_order_reviews_dataset.csv"))
orders = pd.read_csv(os.path.join(DATA_DIR, "olist_orders_dataset.csv"))
products = pd.read_csv(os.path.join(DATA_DIR, "olist_products_dataset.csv"))
translation = pd.read_csv(os.path.join(DATA_DIR, "product_category_name_translation.csv"))
sellers = pd.read_csv(os.path.join(DATA_DIR, "olist_sellers_dataset.csv"))

# 2. 데이터 병합 (Seller 중심)
print("데이터 병합 중...")
# items + orders
df = items.merge(orders, on="order_id", how="left")
# + reviews (중복 방지를 위해 리뷰 점수 평균 사용 혹은 첫 번째 리뷰 사용)
# Olist는 보통 주문당 하나지만, 여러 아이템이 있을 때 리뷰가 중복될 수 있음
# 여기서는 order_id별 평균 점수를 구해서 병합
order_reviews = reviews.groupby('order_id')['review_score'].mean().reset_index()
df = df.merge(order_reviews, on="order_id", how="left")
# + products
df = df.merge(products, on="product_id", how="left")
# + translation
df = df.merge(translation, on="product_category_name", how="left")

# 날짜 변환
date_cols = ['order_purchase_timestamp', 'order_delivered_customer_date', 'order_estimated_delivery_date']
for col in date_cols:
    df[col] = pd.to_datetime(df[col])

# 배송 지표 계산용
df_delivered = df[(df['order_status'] == 'delivered') & (df['order_delivered_customer_date'].notnull())].copy()
df_delivered['is_on_time'] = df_delivered['order_delivered_customer_date'] <= df_delivered['order_estimated_delivery_date']
df_delivered['delivery_days'] = (df_delivered['order_delivered_customer_date'] - df_delivered['order_purchase_timestamp']).dt.days
df_delivered['delay_days'] = (df_delivered['order_delivered_customer_date'] - df_delivered['order_estimated_delivery_date']).dt.days
df_delivered['delay_days'] = df_delivered['delay_days'].apply(lambda x: max(0, x))

# 3. seller_summary 생성
print("seller_summary 생성 중...")

# 매출/규모
sales_agg = df.groupby('seller_id').agg(
    total_sales=('price', 'sum'),
    order_count=('order_id', 'nunique'),
    item_count=('order_id', 'count')
).reset_index()
sales_agg['avg_order_value'] = sales_agg['total_sales'] / sales_agg['order_count']

# 리뷰
review_agg = df.groupby('seller_id').agg(
    avg_review_score=('review_score', 'mean'),
    review_count=('review_score', 'count')
).reset_index()

# 저평점/고평점 비율
df['is_low_review'] = df['review_score'].apply(lambda x: 1 if x <= 2 else 0)
df['is_high_review'] = df['review_score'].apply(lambda x: 1 if x >= 4 else 0)
review_rates = df.groupby('seller_id').agg(
    low_review_rate=('is_low_review', 'mean'),
    high_review_rate=('is_high_review', 'mean')
).reset_index()

# 배송
delivery_agg = df_delivered.groupby('seller_id').agg(
    on_time_rate=('is_on_time', 'mean'),
    avg_delivery_days=('delivery_days', 'mean'),
    avg_delay_days=('delay_days', 'mean'),
    delayed_order_count=('is_on_time', lambda x: (x == False).sum()),
    delivered_count=('is_on_time', 'count')
).reset_index()
delivery_agg['delayed_order_rate'] = delivery_agg['delayed_order_count'] / delivery_agg['delivered_count']

# 카테고리
cat_agg = df.groupby('seller_id').agg(
    category_count=('product_category_name_english', 'nunique')
).reset_index()

# 메인 카테고리
def get_main_cat(x):
    counts = x['product_category_name_english'].value_counts()
    if counts.empty: return None
    return counts.index[0]

def get_main_cat_share(x):
    counts = x['product_category_name_english'].value_counts()
    if counts.empty: return 0
    return counts.iloc[0] / len(x)

main_cat = df.groupby('seller_id').apply(lambda x: pd.Series({
    'main_category': get_main_cat(x),
    'main_category_share': get_main_cat_share(x)
})).reset_index()

# 가격/배송비
price_agg = df.groupby('seller_id').agg(
    avg_price=('price', 'mean'),
    median_price=('price', 'median'),
    avg_freight=('freight_value', 'mean')
).reset_index()

df['freight_ratio'] = df['freight_value'] / df['price']
df.loc[df['price'] == 0, 'freight_ratio'] = np.nan
freight_ratio_agg = df.groupby('seller_id').agg(
    avg_freight_ratio=('freight_ratio', 'mean')
).reset_index()

# 병합
seller_summary = sales_agg.merge(review_agg, on='seller_id', how='left')\
                          .merge(review_rates, on='seller_id', how='left')\
                          .merge(delivery_agg, on='seller_id', how='left')\
                          .merge(cat_agg, on='seller_id', how='left')\
                          .merge(main_cat, on='seller_id', how='left')\
                          .merge(price_agg, on='seller_id', how='left')\
                          .merge(freight_ratio_agg, on='seller_id', how='left')

# 4. 셀러 그룹 정의
print("셀러 그룹 정의 중...")
seller_summary = seller_summary.sort_values('total_sales', ascending=False)
n_sellers = len(seller_summary)
top_20_idx = int(n_sellers * 0.2)
bottom_20_idx = n_sellers - int(n_sellers * 0.2)

seller_summary['seller_group'] = 'Middle'
seller_summary.iloc[:top_20_idx, seller_summary.columns.get_loc('seller_group')] = 'Top 20%'
seller_summary.iloc[bottom_20_idx:, seller_summary.columns.get_loc('seller_group')] = 'Bottom 20%'

# 보조 분석용 (order_count >= 5)
seller_summary_filtered = seller_summary[seller_summary['order_count'] >= 5].copy()
seller_summary_filtered = seller_summary_filtered.sort_values('total_sales', ascending=False)
n_filtered = len(seller_summary_filtered)
t20_f = int(n_filtered * 0.2)
b20_f = n_filtered - int(n_filtered * 0.2)
seller_summary_filtered['seller_group_filtered'] = 'Middle'
seller_summary_filtered.iloc[:t20_f, seller_summary_filtered.columns.get_loc('seller_group_filtered')] = 'Top 20%'
seller_summary_filtered.iloc[b20_f:, seller_summary_filtered.columns.get_loc('seller_group_filtered')] = 'Bottom 20%'

# 5. 시각화
print("시각화 생성 중...")

def plot_group_comparison(data, column, title, filename, ylabel=None):
    plt.figure(figsize=(10, 6))
    order = ['Top 20%', 'Middle', 'Bottom 20%']
    sns.barplot(data=data, x='seller_group', y=column, order=order, palette='viridis')
    plt.title(title)
    if ylabel: plt.ylabel(ylabel)
    save_fig(filename)

# 1) seller_id별 total_sales 분포
plt.figure(figsize=(10, 6))
sns.histplot(seller_summary['total_sales'], bins=100, kde=True)
plt.xscale('log')
plt.title("seller_id별 총 매출 분포 (Log Scale)")
save_fig("s01_sales_dist.png")

# 2) seller_group별 셀러 수
plt.figure(figsize=(8, 6))
sns.countplot(data=seller_summary, x='seller_group', order=['Top 20%', 'Middle', 'Bottom 20%'])
plt.title("셀러 그룹별 셀러 수")
save_fig("s02_group_count.png")

# 3) 매출 기여도
group_sales = seller_summary.groupby('seller_group')['total_sales'].sum().reset_index()
plt.figure(figsize=(8, 8))
plt.pie(group_sales['total_sales'], labels=group_sales['seller_group'], autopct='%1.1f%%', startangle=140)
plt.title("셀러 그룹별 매출 기여도")
save_fig("s03_sales_share.png")

# 4~5) 주문 및 아이템 수
plot_group_comparison(seller_summary, 'order_count', "그룹별 평균 주문 수", "s04_avg_order_count.png")
plot_group_comparison(seller_summary, 'item_count', "그룹별 평균 판매 아이템 수", "s05_avg_item_count.png")

# 6~7) 리뷰
plot_group_comparison(seller_summary, 'avg_review_score', "그룹별 평균 리뷰 점수", "s06_avg_review.png")
# low/high review rate
df_review_rates = seller_summary.groupby('seller_group')[['low_review_rate', 'high_review_rate']].mean().reset_index()
df_review_rates_melt = df_review_rates.melt(id_vars='seller_group')
plt.figure(figsize=(10, 6))
sns.barplot(data=df_review_rates_melt, x='seller_group', y='value', hue='variable', order=['Top 20%', 'Middle', 'Bottom 20%'])
plt.title("그룹별 저평점/고평점 비율 비교")
save_fig("s07_review_rates.png")

# 8~9) 배송
plot_group_comparison(seller_summary, 'on_time_rate', "그룹별 평균 정시 배송률", "s08_on_time_rate.png")
# avg_delivery_days / avg_delay_days
df_delivery = seller_summary.groupby('seller_group')[['avg_delivery_days', 'avg_delay_days']].mean().reset_index()
df_delivery_melt = df_delivery.melt(id_vars='seller_group')
plt.figure(figsize=(10, 6))
sns.barplot(data=df_delivery_melt, x='seller_group', y='value', hue='variable', order=['Top 20%', 'Middle', 'Bottom 20%'])
plt.title("그룹별 평균 배송/지연 일수")
save_fig("s09_delivery_days.png")

# 10) 카테고리 수
plot_group_comparison(seller_summary, 'category_count', "그룹별 평균 취급 카테고리 수", "s10_category_count.png")

# 11) Top 10 카테고리 비교
def get_top_10_cats(group_name):
    return seller_summary[seller_summary['seller_group'] == group_name]['main_category'].value_counts().head(10)

top_cats_top = get_top_10_cats('Top 20%')
top_cats_bottom = get_top_10_cats('Bottom 20%')

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
sns.barplot(x=top_cats_top.values, y=top_cats_top.index, ax=ax1)
ax1.set_title("Top 20% 셀러 주력 카테고리 TOP 10")
sns.barplot(x=top_cats_bottom.values, y=top_cats_bottom.index, ax=ax2)
ax2.set_title("Bottom 20% 셀러 주력 카테고리 TOP 10")
save_fig("s11_top_categories_comp.png")

# 12) 가격 비교
df_price = seller_summary.groupby('seller_group')[['avg_price', 'median_price']].mean().reset_index()
df_price_melt = df_price.melt(id_vars='seller_group')
plt.figure(figsize=(10, 6))
sns.barplot(data=df_price_melt, x='seller_group', y='value', hue='variable', order=['Top 20%', 'Middle', 'Bottom 20%'])
plt.title("그룹별 평균/중앙 가격 비교")
save_fig("s12_price_comp.png")

# 13) 배송비 비교
df_freight = seller_summary.groupby('seller_group')[['avg_freight', 'avg_freight_ratio']].mean().reset_index()
fig, ax1 = plt.subplots(figsize=(10, 6))
ax2 = ax1.twinx()
sns.barplot(data=df_freight, x='seller_group', y='avg_freight', order=['Top 20%', 'Middle', 'Bottom 20%'], ax=ax1, color='lightblue', label='Avg Freight')
sns.lineplot(data=df_freight, x='seller_group', y='avg_freight_ratio', sort=False, ax=ax2, color='red', marker='o', label='Freight Ratio')
ax1.set_ylabel("Avg Freight Value")
ax2.set_ylabel("Avg Freight Ratio")
plt.title("그룹별 평균 배송비 및 배송비 비율")
save_fig("s13_freight_comp.png")

# 14~16) 산점도
plt.figure(figsize=(10, 6))
sns.scatterplot(data=seller_summary.sample(min(2000, len(seller_summary))), x='on_time_rate', y='avg_review_score', alpha=0.3)
plt.title("정시 배송률 vs 리뷰 점수")
save_fig("s14_ontime_vs_review.png")

plt.figure(figsize=(10, 6))
sns.scatterplot(data=seller_summary.sample(min(2000, len(seller_summary))), x='avg_freight_ratio', y='avg_review_score', alpha=0.3)
plt.title("배송비 비율 vs 리뷰 점수")
save_fig("s15_freight_vs_review.png")

# 6. 리포트용 집계 데이터 저장
print("집계 데이터 저장 중...")

# 핵심 요약표
def get_summary_metrics(df_group):
    return pd.Series({
        '셀러 수': len(df_group),
        '총매출': df_group['total_sales'].sum(),
        '매출 기여도': df_group['total_sales'].sum() / seller_summary['total_sales'].sum(),
        '평균 total_sales': df_group['total_sales'].mean(),
        '평균 order_count': df_group['order_count'].mean(),
        '평균 item_count': df_group['item_count'].mean(),
        '평균 avg_review_score': df_group['avg_review_score'].mean(),
        '평균 low_review_rate': df_group['low_review_rate'].mean(),
        '평균 high_review_rate': df_group['high_review_rate'].mean(),
        '평균 on_time_rate': df_group['on_time_rate'].mean(),
        '평균 avg_delivery_days': df_group['avg_delivery_days'].mean(),
        '평균 avg_delay_days': df_group['avg_delay_days'].mean(),
        '평균 category_count': df_group['category_count'].mean(),
        '평균 avg_price': df_group['avg_price'].mean(),
        '평균 median_price': df_group['median_price'].mean(),
        '평균 avg_freight': df_group['avg_freight'].mean(),
        '평균 avg_freight_ratio': df_group['avg_freight_ratio'].mean()
    })

comparison_table = seller_summary.groupby('seller_group').apply(get_summary_metrics).reindex(['Top 20%', 'Middle', 'Bottom 20%'])

# 보조 분석 (Filtered)
comparison_table_filtered = seller_summary_filtered.groupby('seller_group_filtered').apply(get_summary_metrics).reindex(['Top 20%', 'Middle', 'Bottom 20%'])

with open(os.path.join(REPORT_DIR, "seller_analysis_results.txt"), "w", encoding="utf-8") as f:
    f.write("=== 전체 셀러 기준 그룹 비교 ===\n")
    f.write(tabulate(comparison_table, headers='keys', tablefmt='pipe') + "\n\n")
    f.write("=== 주문 5건 이상 셀러 기준 그룹 비교 ===\n")
    f.write(tabulate(comparison_table_filtered, headers='keys', tablefmt='pipe') + "\n\n")

print("분석 완료!")
