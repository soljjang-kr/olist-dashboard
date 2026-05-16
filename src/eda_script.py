"""
Olist 이커머스 데이터의 전반적인 탐색적 분석(EDA)을 수행하고 시각화 차트 및 통계 리포트를 생성하는 스크립트입니다.

주요 기능:
- 고객, 주문, 아이템, 결제 등 여러 원시 데이터셋 로드 및 병합
- 데이터 무결성 검증 및 수치형/범주형 변수 기술통계 산출
- 주문 상태, 결제 수단, 카테고리 분포 등 10종 이상의 시각화 이미지 생성
- 시계열 주문 추이 및 지역별 주문 분포 분석
- 리뷰 텍스트에 대한 TF-IDF 기반 키워드 랭킹 산출 및 결과 저장
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import koreanize_matplotlib
import os
from sklearn.feature_extraction.text import TfidfVectorizer
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

# 2. 데이터 병합
print("데이터 병합 중...")
df = orders.merge(items, on="order_id", how="left")
df = df.merge(customers, on="customer_id", how="left")
df = df.merge(payments, on="order_id", how="left")
df = df.merge(products, on="product_id", how="left")
df = df.merge(translation, on="product_category_name", how="left")

# 날짜 변환
date_cols = ['order_purchase_timestamp', 'order_approved_at', 
             'order_delivered_carrier_date', 'order_delivered_customer_date', 
             'order_estimated_delivery_date']
for col in date_cols:
    df[col] = pd.to_datetime(df[col])

# 3. 기본 검증 및 기술통계
print("기본 검증 및 기술통계 산출 중...")
info_str = []
df.info(buf=None) # 콘솔 출력용

# 상위/하위 5개행
head_df = df.head(5)
tail_df = df.tail(5)

# 중복 데이터
duplicates = df.duplicated().sum()

# 기술통계
desc_num = df.describe()
desc_obj = df.describe(include=['object', 'datetime'])

# 4. 시각화 및 분석
print("시각화 생성 중...")

# 1) 주문 상태 분포 (범주형 빈도)
plt.figure(figsize=(10, 6))
sns.countplot(data=df, y='order_status', order=df['order_status'].value_counts().index)
plt.title("주문 상태 분포")
save_fig("01_order_status.png")

# 2) 결제 수단 분포 (범주형 빈도)
plt.figure(figsize=(10, 6))
sns.countplot(data=df, x='payment_type', order=df['payment_type'].value_counts().index)
plt.title("결제 수단 분포")
save_fig("02_payment_type.png")

# 3) 상위 30개 상품 카테고리 (범주형 빈도)
plt.figure(figsize=(12, 8))
top_30_cat = df['product_category_name_english'].value_counts().head(30)
sns.barplot(x=top_30_cat.values, y=top_30_cat.index)
plt.title("상위 30개 상품 카테고리")
save_fig("03_top_categories.png")

# 4) 가격 분포 (일변량 수치형)
plt.figure(figsize=(10, 6))
sns.histplot(df['price'].dropna(), bins=50, kde=True)
plt.title("상품 가격 분포")
save_fig("04_price_dist.png")

# 5) 결제 금액과 배송비 관계 (이변량)
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df.sample(1000), x='price', y='freight_value', alpha=0.5)
plt.title("가격 vs 배송비 (샘플 1000건)")
save_fig("05_price_vs_freight.png")

# 6) 월별 주문 추이 (시계열)
df['order_month'] = df['order_purchase_timestamp'].dt.to_period('M')
monthly_orders = df.groupby('order_month').size()
plt.figure(figsize=(12, 6))
monthly_orders.plot(kind='line', marker='o')
plt.title("월별 주문 추이")
plt.xticks(rotation=45)
save_fig("06_monthly_orders.png")

# 7) 요일별 주문 분포
df['day_of_week'] = df['order_purchase_timestamp'].dt.day_name()
plt.figure(figsize=(10, 6))
sns.countplot(data=df, x='day_of_week', order=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
plt.title("요일별 주문 분포")
save_fig("07_day_of_week_orders.png")

# 8) 결제 수단별 평균 결제 금액 (이변량)
plt.figure(figsize=(10, 6))
sns.barplot(data=df, x='payment_type', y='payment_value')
plt.title("결제 수단별 평균 결제 금액")
save_fig("08_payment_avg_value.png")

# 9) 고객 거주 주(State)별 주문 수 (상위 20개)
plt.figure(figsize=(12, 8))
top_states = df['customer_state'].value_counts().head(20)
sns.barplot(x=top_states.values, y=top_states.index)
plt.title("고객 거주 주별 주문 수 (상위 20개)")
save_fig("09_top_states.png")

# 10) 결제 할부 횟수 분포
plt.figure(figsize=(10, 6))
sns.histplot(df['payment_installments'].dropna(), bins=20)
plt.title("결제 할부 횟수 분포")
save_fig("10_payment_installments.png")

# 5. 텍스트 분석 (리뷰 데이터)
print("텍스트 분석 중...")
reviews_text = reviews['review_comment_message'].dropna().head(5000) # 성능상 상위 5000건만
if not reviews_text.empty:
    vectorizer = TfidfVectorizer(max_features=30, stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(reviews_text)
    keywords = vectorizer.get_feature_names_out()
    sums = tfidf_matrix.sum(axis=0)
    data = []
    for col, idx in enumerate(vectorizer.vocabulary_):
        data.append((idx, sums[0, col]))
    
    ranking = pd.DataFrame(list(vectorizer.vocabulary_.items()), columns=['keyword', 'index'])
    ranking['score'] = ranking['index'].apply(lambda x: sums[0, x])
    ranking = ranking.sort_values('score', ascending=False).head(30)
    
    plt.figure(figsize=(12, 8))
    sns.barplot(data=ranking, x='score', y='keyword')
    plt.title("리뷰 텍스트 TF-IDF 상위 30 키워드")
    save_fig("11_review_keywords.png")
    
    keyword_table = ranking[['keyword', 'score']].reset_index(drop=True)
else:
    keyword_table = pd.DataFrame()

# 6. 리포트용 데이터 정리 (JSON 등으로 저장하거나 직접 리포트 작성)
# 여기서는 간단히 텍스트 파일로 중간 결과 저장
with open(os.path.join(REPORT_DIR, "analysis_results.txt"), "w", encoding="utf-8") as f:
    f.write(f"전체 행 수: {df.shape[0]}, 열 수: {df.shape[1]}\n")
    f.write(f"중복 데이터 수: {duplicates}\n\n")
    
    f.write("--- 상위 5개행 ---\n")
    f.write(tabulate(head_df, headers='keys', tablefmt='pipe') + "\n\n")
    
    f.write("--- 수치형 기술통계 ---\n")
    f.write(tabulate(desc_num, headers='keys', tablefmt='pipe') + "\n\n")
    
    f.write("--- 범주형 기술통계 ---\n")
    f.write(tabulate(desc_obj, headers='keys', tablefmt='pipe') + "\n\n")
    
    # 각 시각화에 대응하는 통계표 추가
    f.write("--- 01. 주문 상태 빈도표 ---\n")
    f.write(tabulate(df['order_status'].value_counts().reset_index(), headers=['Status', 'Count'], tablefmt='pipe') + "\n\n")
    
    f.write("--- 02. 결제 수단 빈도표 ---\n")
    f.write(tabulate(df['payment_type'].value_counts().reset_index(), headers=['Payment Type', 'Count'], tablefmt='pipe') + "\n\n")
    
    f.write("--- 03. 상위 30개 상품 카테고리 ---\n")
    f.write(tabulate(top_30_cat.reset_index(), headers=['Category', 'Count'], tablefmt='pipe') + "\n\n")
    
    f.write("--- 04. 상품 가격 기술통계 ---\n")
    f.write(tabulate(df['price'].describe().reset_index(), headers=['Metric', 'Value'], tablefmt='pipe') + "\n\n")
    
    f.write("--- 05. 가격 vs 배송비 상관계수 ---\n")
    corr = df[['price', 'freight_value']].corr()
    f.write(tabulate(corr, headers='keys', tablefmt='pipe') + "\n\n")
    
    f.write("--- 06. 월별 주문 수 ---\n")
    f.write(tabulate(monthly_orders.reset_index(), headers=['Month', 'Orders'], tablefmt='pipe') + "\n\n")
    
    f.write("--- 07. 요일별 주문 수 ---\n")
    f.write(tabulate(df['day_of_week'].value_counts().reindex(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']).reset_index(), headers=['Day', 'Orders'], tablefmt='pipe') + "\n\n")
    
    f.write("--- 08. 결제 수단별 평균 결제 금액 ---\n")
    pay_avg = df.groupby('payment_type')['payment_value'].mean().reset_index()
    f.write(tabulate(pay_avg, headers=['Payment Type', 'Avg Value'], tablefmt='pipe') + "\n\n")
    
    f.write("--- 09. 고객 거주 주별 주문 수 (상위 20) ---\n")
    f.write(tabulate(top_states.reset_index(), headers=['State', 'Orders'], tablefmt='pipe') + "\n\n")
    
    f.write("--- 10. 결제 할부 횟수 빈도 ---\n")
    installments = df['payment_installments'].value_counts().sort_index().head(20).reset_index()
    f.write(tabulate(installments, headers=['Installments', 'Count'], tablefmt='pipe') + "\n\n")

    if not keyword_table.empty:
        f.write("--- 11. TF-IDF 키워드 상위 30 ---\n")
        f.write(tabulate(keyword_table, headers='keys', tablefmt='pipe') + "\n\n")

print("분석 완료!")
