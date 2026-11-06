from src.get_raw_data import get_access_token
from src.preprocessing import WINDOW_SIZE, FEATURES
from datetime import datetime, timedelta
import sys
from dotenv import load_dotenv
from pathlib import Path
import os
import requests
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib

# 경로 설정
ROOT_DIR = Path(__file__).resolve().parent
MODEL_DIR = ROOT_DIR / "models"
DATA_DIR = ROOT_DIR / "data"

# API 설정
load_dotenv(ROOT_DIR / ".env")

APP_KEY = os.getenv("APP_KEY")
APP_SECRET = os.getenv("APP_SECRET")
TOKEN = get_access_token()

BASE_URL = "https://openapi.koreainvestment.com:9443"

# 시간 체크 함수
def check_time():
    now = datetime.now()

    h = now.hour
    m = now.minute

    time = 100*h + m
    today = now.strftime("%Y%m%d")

    if time < 1045 or time > 1515:
        print("예측 가능한 시간대가 아닙니다.")
        sys.exit()

    url = f"{BASE_URL}/uapi/domestic-stock/v1/quotations/chk-holiday"
    headers = {
        "content-type": "application/json; charset=utf-8",
        "authorization": f"Bearer {TOKEN}",
        "appkey": APP_KEY,
        "appsecret": APP_SECRET,
        "tr_id": "CTCA0903R",
        "custtype": "P"
    }
    params = {
        "BASS_DT": today,
        "CTX_AREA_NK": "",
        "CTX_AREA_FK": ""
    }

    res = requests.get(url, headers = headers, params = params)

    res.raise_for_status()
    data = res.json()

    if data.get("rt_cd") != "0":
        print("휴장일 조회 실패:", data.get("msg1"))
        sys.exit()

    if data.get("output")[0]['opnd_yn'] != "Y":
        print("오늘은 주식시장 휴장일입니다.")
        sys.exit()

    return 0

def get_realtime_data():
    url = f"{BASE_URL}/uapi/domestic-stock/v1/quotations/inquire-time-itemchartprice"
    headers = {
        "content-type": "application/json; charset=utf-8",
        "authorization" : f"Bearer {TOKEN}",
        "appkey": APP_KEY,
        "appsecret": APP_SECRET,
        "tr_id": "FHKST03010200",
        "custtype": "P"
    }

    current_dt = datetime.now()

    df = []

    for _ in range(4):
        current_time = current_dt.strftime("%H%M%S")

        params = {
            "FID_COND_MRKT_DIV_CODE": "J",
            "FID_INPUT_ISCD": "069500",
            "FID_INPUT_HOUR_1": current_time,
            "FID_PW_DATA_INCU_YN": "N",
            "FID_ETC_CLS_CODE": ""
        }
        
        res = requests.get(url, headers = headers, params = params)
        res.raise_for_status()
        
        data = res.json()
        if data.get("rt_cd") != "0":
            print(f"API 오류 발생: {data.get('msg1')}")
            sys.exit()
        df.append(pd.DataFrame(data.get("output2")))

        current_dt -= timedelta(minutes = 30)

    full_df = pd.concat(df, ignore_index=True)
    full_df = full_df.drop(columns = ['acml_tr_pbmn'])
    full_df = full_df.drop_duplicates(subset = ['stck_bsop_date', 'stck_cntg_hour'], keep = 'first')
    full_df.columns = ['Date', 'Time', 'Close', 'Open', 'High', 'Low', 'Volume']
    full_df = full_df[['Date', 'Time', 'Open', 'Close', 'High', 'Low', 'Volume']]
    full_df = full_df.sort_values("Time").reset_index(drop = True)
    full_df[['Open', 'Close', 'High', 'Low', 'Volume']] = full_df[['Open', 'Close', 'High', 'Low', 'Volume']].apply(pd.to_numeric)
    
    if full_df.isna().any().any():
        print("데이터에 결측값이 존재합니다.")
        sys.exit()

    if (full_df['Volume'] == 0).any():
        print("최근 120분 사이에 사이킷런이 발동했었습니다.")
        sys.exit()

    full_df['DateTime'] = pd.to_datetime(full_df['Date'] + full_df['Time'], format = "%Y%m%d%H%M%S")
    full_df.set_index("DateTime", inplace = True)

    current_dt = pd.Timestamp.now()
    full_df = full_df[full_df.index < current_dt.floor("5min")]

    df_5min = full_df.resample("5min", closed = "left", label = "left").agg({
            "Open": "first",
            "Close": "last",
            "High": "max",
            "Low": "min",
            "Volume": "sum"
    })
    df_5min.reset_index(inplace = True)

    scaler = joblib.load(ROOT_DIR / "data" / "scaler.pkl")

    X = df_5min[FEATURES].tail(WINDOW_SIZE)
    X = scaler.transform(X)
    X = np.array(X).reshape(1, WINDOW_SIZE, len(FEATURES))

    return X

# 파일 직접 실행
if __name__ == "__main__":
    # check_time()

    # X = get_realtime_data()

    test_set = np.load(DATA_DIR / "test.npz")
    X = test_set["X"][0:1]
    
    print(X, X.shape)

    classifier = tf.keras.models.load_model(MODEL_DIR / "classifier.keras", compile = False)
    regressor = tf.keras.models.load_model(MODEL_DIR / "regressor.keras", compile = False)

    output_class = classifier.predict(X, verbose = 0)[0, 0]
    output_reg = regressor.predict(X, verbose = 0)[0, 0]

    print(f"상승 확률: {output_class * 100:.2f}%")
    print(f"예상 수익률: {output_reg * 100:.3f}%")