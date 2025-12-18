FROM python:3.11-slim

WORKDIR /app

# 시스템 패키지 설치
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# 의존성 파일 복사 및 설치
COPY pyproject.toml .
RUN pip install --no-cache-dir .

# 애플리케이션 코드 복사
COPY javis/ ./javis/
COPY static/ ./static/
COPY configs/ ./configs/

# 포트 설정
EXPOSE 8000

# 환경변수
ENV PYTHONUNBUFFERED=1

# 실행
CMD ["uvicorn", "javis.interfaces.api:app", "--host", "0.0.0.0", "--port", "8000"]
