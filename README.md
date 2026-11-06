# KODEX 200 주가 예측 — CNN + LSTM (Base Model)

한국투자증권 **KIS OpenAPI**에서 수집한 KODEX 200 ETF(069500)의 **5분봉 OHLCV 데이터**로 다음 5분봉의 **상승 여부(분류)**와 **수익률(회귀)**을 예측하는 프로젝트입니다.

과거 20개 5분봉을 입력으로 사용하는 CNN+LSTM 기반 **Base Model**이며, 스케일링과 추가 피처 엔지니어링은 적용하지 않았습니다. 학습 및 실험 목적의 프로젝트로, 실거래용으로 검증된 모델은 아닙니다.

## 실행 방법

### 1. 라이브러리 설치

프로젝트를 다운로드한 후 프로젝트 폴더에서 실행합니다.

```bash
pip install tensorflow pandas numpy scikit-learn requests python-dotenv
```

### 2. API 키 설정

[한국투자증권 KIS OpenAPI 포털](https://apiportal.koreainvestment.com)
한국투자증권 KIS OpenAPI에서 **APP KEY**와 **APP SECRET**을 발급받습니다.

프로젝트 폴더의 `.env.example`을 복사해 이름을 `.env`로 변경합니다.

Windows PowerShell:

```powershell
Copy-Item .env.example .env
```

macOS / Linux:

```bash
cp .env.example .env
```

`.env` 파일에 발급받은 키를 입력합니다.

```dotenv
APP_KEY=발급받은_APP_KEY
APP_SECRET=발급받은_APP_SECRET
```

**주의:** 실제 키가 들어 있는 `.env`와 `token.txt`는 GitHub에 업로드하지 마세요. `.env.example`만 공개합니다.

### 3. 모델 학습

```bash
python train.py
```

KIS API에서 데이터를 수집하고, Train / Validation / Test 데이터셋을 생성한 뒤 분류·회귀 모델을 학습하고 평가합니다. 학습된 모델은 `models/`에 저장됩니다.

### 4. 실시간 예측

```bash
python predict.py
```

학습된 모델이 있는 상태에서 실행하면 **상승 확률**과 **예상 수익률**을 출력합니다. 현재 코드는 국내 주식시장 거래일 **10:45~15:15(한국 시간)**에 실행하도록 구성되어 있습니다.