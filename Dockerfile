# 1. 베이스 이미지 설정
FROM python:3.11-slim

# 2. 폰트 설치 명령어
RUN apt-get update && apt-get install -y fonts-nanum*

# 3. 작업 디렉토리 설정
WORKDIR /app

# 4. 라이브러리 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5. 프로젝트 파일 복사
COPY . .

# 6. 서버 실행
CMD ["python", "api.py"]