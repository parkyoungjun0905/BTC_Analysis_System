# 🚀 Azure Functions 배포 가이드
## 11개 선행지표 비트코인 리스크 모니터링 시스템

---

## 📋 시스템 개요

### 구성요소
- **무료 지표**: 8개 (Binance, Yahoo Finance 등)
- **CryptoQuant**: 3개 (온체인 핵심 지표)
- **Claude AI**: 예측 분석
- **텔레그램**: 알림 발송

### 주요 기능
- 30분마다 자동 실행
- 11개 선행지표 실시간 수집
- Claude AI 가격 예측
- 정확도 기반 알림 필터링
- 예측 성과 자동 학습

---

## 🔧 사전 준비사항

### 1. 필수 계정
- [ ] Azure 계정 (무료 크레딧 가능)
- [ ] CryptoQuant API 키 (구독 필요)
- [ ] Claude API 키 (Anthropic)
- [ ] 텔레그램 봇 토큰 & 채팅 ID

### 2. 로컬 도구 설치
```bash
# Azure Functions Core Tools
brew install azure-functions-core-tools@4

# Azure CLI
brew install azure-cli

# Python 3.11
brew install python@3.11
```

---

## 📦 로컬 테스트

### 1. 환경 설정
```bash
# 가상환경 생성
python3.11 -m venv venv
source venv/bin/activate

# 패키지 설치
pip install -r requirements.txt
```

### 2. 환경변수 설정
`local.settings.json` 파일 수정:
```json
{
  "Values": {
    "CRYPTOQUANT_API_KEY": "실제-API-키",
    "CLAUDE_API_KEY": "실제-API-키",
    "TELEGRAM_BOT_TOKEN": "봇-토큰",
    "TELEGRAM_CHAT_ID": "채팅-ID"
  }
}
```

### 3. 로컬 실행
```bash
# Azure Functions 로컬 실행
func start

# 헬스체크
curl http://localhost:7071/api/health

# 수동 모니터링 실행
curl http://localhost:7071/api/monitor
```

---

## ☁️ Azure 배포

### 1. Azure 리소스 생성
```bash
# 로그인
az login

# 리소스 그룹 생성
az group create \
  --name btc-monitor-rg \
  --location koreacentral

# 스토리지 계정 생성
az storage account create \
  --name btcmonitorstorage \
  --resource-group btc-monitor-rg \
  --location koreacentral \
  --sku Standard_LRS

# Function App 생성
az functionapp create \
  --name btc-risk-monitor-11 \
  --resource-group btc-monitor-rg \
  --storage-account btcmonitorstorage \
  --consumption-plan-location koreacentral \
  --runtime python \
  --runtime-version 3.11 \
  --functions-version 4 \
  --os-type Linux
```

### 2. 환경변수 설정
```bash
# API 키 설정
az functionapp config appsettings set \
  --name btc-risk-monitor-11 \
  --resource-group btc-monitor-rg \
  --settings \
    "CRYPTOQUANT_API_KEY=실제-키" \
    "CLAUDE_API_KEY=실제-키" \
    "TELEGRAM_BOT_TOKEN=봇-토큰" \
    "TELEGRAM_CHAT_ID=채팅-ID"
```

### 3. 코드 배포
```bash
# 배포
func azure functionapp publish btc-risk-monitor-11

# 로그 확인
func azure functionapp logstream btc-risk-monitor-11
```

---

## 📊 모니터링 및 관리

### Azure Portal에서 확인
1. [Azure Portal](https://portal.azure.com) 접속
2. Function App > btc-risk-monitor-11 선택
3. 모니터링 > 로그 스트림 확인

### 실행 주기 변경
`function_app.py`에서 스케줄 수정:
```python
# 30분마다 (현재)
@app.timer_trigger(schedule="0 */30 * * * *")

# 1시간마다로 변경
@app.timer_trigger(schedule="0 0 * * * *")

# 15분마다로 변경
@app.timer_trigger(schedule="0 */15 * * * *")
```

### 수동 실행
```bash
# HTTP 엔드포인트 호출
curl https://btc-risk-monitor-11.azurewebsites.net/api/monitor

# 상태 확인
curl https://btc-risk-monitor-11.azurewebsites.net/api/health
```

---

## 💰 비용 관리

### 예상 비용 (월)
- **Consumption Plan**: ~3-5만원
  - 실행: 30분마다 = 1,440회/월
  - 평균 실행시간: 5-10초
  - 메모리: 128MB

### 비용 절감 팁
1. 실행 주기를 1시간으로 변경
2. 불필요한 로그 줄이기
3. 예측 정확도 낮은 시간대 스킵

---

## 🔍 문제 해결

### 1. 배포 실패
```bash
# Python 버전 확인
python --version  # 3.11 필수

# 패키지 충돌 해결
pip install --upgrade pip
pip install -r requirements.txt --force-reinstall
```

### 2. API 키 오류
```bash
# 환경변수 확인
az functionapp config appsettings list \
  --name btc-risk-monitor-11 \
  --resource-group btc-monitor-rg
```

### 3. 텔레그램 알림 안 옴
- 봇이 그룹에 추가되었는지 확인
- 채팅 ID가 올바른지 확인 (음수 포함)
- 봇 권한 확인 (메시지 전송 권한)

### 4. CryptoQuant API 한도
- 분당 요청 제한 확인
- 캐싱 로직 추가 고려
- 필요시 지표 개수 축소

---

## 📈 성능 최적화

### 1. 캐싱 전략
- Redis Cache 추가 (선택)
- 지표별 캐시 TTL 설정

### 2. 병렬 처리
- 8개 무료 지표 동시 수집
- asyncio 활용 최적화

### 3. 알림 최적화
- 중복 알림 방지 로직
- 신뢰도 임계값 조정

---

## 🔄 업데이트 방법

### 코드 수정 후 재배포
```bash
# 1. 로컬 테스트
func start

# 2. 배포
func azure functionapp publish btc-risk-monitor-11

# 3. 로그 확인
func azure functionapp logstream btc-risk-monitor-11
```

### 지표 추가/제거
1. `enhanced_11_indicators.py` 수정
2. 가중치 재조정
3. 테스트 후 배포

---

## 📞 지원 및 문의

### 트러블슈팅 체크리스트
- [ ] 모든 API 키가 올바르게 설정됨
- [ ] Python 3.11 사용 중
- [ ] requirements.txt 모든 패키지 설치됨
- [ ] Azure Functions Core Tools 최신 버전
- [ ] 로컬에서 정상 작동 확인

### 로그 위치
- **Azure Portal**: Monitor > Log stream
- **로컬**: `.func/logs/`
- **SQLite DB**: `predictions.db`

---

## 🎯 다음 단계

### 단기 (1-2주)
- [ ] 예측 정확도 모니터링
- [ ] 알림 임계값 조정
- [ ] 성능 최적화

### 중기 (1-2개월)
- [ ] Glassnode 지표 추가
- [ ] 머신러닝 모델 통합
- [ ] 대시보드 구축

### 장기 (3-6개월)
- [ ] 25개 전체 지표 시스템
- [ ] 자동 매매 연동
- [ ] 멀티 에셋 확장

---

**마지막 업데이트**: 2024년 12월
**버전**: 1.0.0 (11-Indicator System)