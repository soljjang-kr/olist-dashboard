"""
Project1 데이터 디렉토리 내의 모든 CSV 파일을 순회하며 구조와 샘플 데이터를 점검하는 스크립트입니다.

주요 기능:
- 지정된 디렉토리 내의 CSV 파일 리스트 확보
- 각 파일의 컬럼명 출력
- 각 파일의 상위 2개 행 샘플 데이터 출력
"""

import pandas as pd
import glob
import os

data_dir = 'Project1/data'
files = glob.glob(os.path.join(data_dir, '*.csv'))

for f in files:
    try:
        df = pd.read_csv(f, nrows=5)
        print(f"--- {os.path.basename(f)} ---")
        print(f"Columns: {df.columns.tolist()}")
        print(f"Sample:\n{df.head(2)}")
        print("\n")
    except Exception as e:
        print(f"Error reading {f}: {e}")
