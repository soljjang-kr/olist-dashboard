# 베이스 이미지 설정
FROM python:3.14-slim

# 작업 디렉토리 설정
WORKDIR /app

# 시스템 라이브러리 업데이트 및 필수 패키지 설치
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

# 의존성 파일 복사 및 설치
COPY Project1/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 프로젝트 코드 및 데이터 복사
# 배포 시 데이터 경로 구조를 유지하기 위해 Project1 폴더 자체를 복사하거나 구조를 조정합니다.
COPY Project1 /app/Project1

# 포트 설정 (Streamlit 기본 포트)
EXPOSE 8501

# 실행 환경 변수 설정
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_HEADLESS=true

# 앱 실행 (src/dashboard_app.py 위치에 맞춰 실행)
ENTRYPOINT ["streamlit", "run", "Project1/src/dashboard_app.py"]
