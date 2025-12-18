# JAVIS 사용 가이드

## 시작하기

### 서버 실행
```bash
cd C:\Develop\workspace\12.JAVIS
.venv\Scripts\activate
python -m uvicorn javis.interfaces.api:app --host 0.0.0.0 --port 8000 --reload
```

### 웹 접속
브라우저에서 http://localhost:8000 접속

---

## 기능 사용법

### 1. 텍스트 대화
- 입력창에 메시지 입력
- Enter 또는 "전송" 버튼 클릭

### 2. 음성 입력
- 🎤 버튼 클릭
- 마이크 권한 허용
- 말하기 (자동으로 텍스트 변환)

### 3. 음성 출력 (TTS)
- 우상단 "TTS OFF" 버튼 클릭하여 "TTS ON"으로 변경
- AI 응답이 음성으로 읽어짐

### 4. 파일 업로드
- 📎 버튼 클릭
- 파일 선택
- 메시지와 함께 전송

### 5. 피드백 남기기 (중요!)
- AI 응답 아래 👍 또는 👎 클릭
- **👍 좋아요**: 학습 데이터로 사용됨
- **👎 별로**: 학습에서 제외됨

### 6. 대화 초기화
- 우상단 "Clear" 버튼 클릭
- 대화가 저장되고 새 세션 시작

---

## 학습 데이터 수집 가이드

### 좋은 학습 데이터 만들기

#### DO (이렇게 하세요)
- 평소처럼 자연스럽게 대화
- 좋은 응답에는 반드시 👍 클릭
- 다양한 주제로 대화 (코딩, 일상, 질문 등)
- 원하는 응답 스타일이 나오면 👍

#### DON'T (이러면 안됨)
- 의미없는 테스트 메시지 반복
- 모든 응답에 무분별하게 👍
- 잘못된 정보가 포함된 응답에 👍

### 데이터 확인
```bash
# 저장된 대화 확인
dir data\conversations\2025-12\

# 특정 대화 내용 보기
type data\conversations\2025-12\session_xxx.json
```

### 학습 데이터 내보내기
```bash
# API로 내보내기
curl -X POST http://localhost:8000/api/export-training

# 또는 브라우저에서
# POST http://localhost:8000/api/export-training
```

결과 파일: `data/training/exported/conversations_YYYYMMDD.jsonl`

---

## API 엔드포인트

| 메서드 | 경로 | 설명 |
|--------|------|------|
| GET | / | 웹 UI |
| GET | /health | 헬스체크 |
| GET | /docs | API 문서 (Swagger) |
| POST | /api/chat | 대화 |
| POST | /api/clear | 세션 초기화 |
| POST | /api/feedback | 피드백 추가 |
| POST | /api/upload | 파일 업로드 |
| POST | /api/export-training | 학습 데이터 내보내기 |
| GET | /api/sessions | 활성 세션 목록 |

---

## 트러블슈팅

### 음성 인식이 안됨
1. Chrome 브라우저 사용 권장
2. 주소창 왼쪽 자물쇠 → 마이크 권한 허용
3. Windows 설정 → 개인정보 → 마이크 → 앱 접근 허용

### 서버가 안됨
```bash
# 포트 확인
netstat -ano | findstr :8000

# 프로세스 종료
taskkill //F //PID [PID번호]

# 다시 시작
python -m uvicorn javis.interfaces.api:app --port 8000
```

### API 키 오류
`.env` 파일 확인:
```
GROQ_API_KEY=gsk_xxx...
```
