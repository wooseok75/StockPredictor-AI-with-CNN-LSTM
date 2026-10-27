from src.get_raw_data import get_access_token, get_1min_data
from src import preprocessing
from datetime import datetime, timedelta
import sys
import requests
from dotenv import load_dotenv
import os
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf

load_dotenv()

APPKEY = os.getenv("APP_KEY")
APPSECRET = os.getenv("APP_SECRET")
BASEURL = "https://openapi.koreainvestment.com:9443"
FEATURES = ["Open_return", "Close_return", "High_return", "Low_return", "Volume_return"]

token = get_access_token()

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT_DIR, 'data')
RAW_DATA_PATH = os.path.join(DATA_DIR, 'raw_data.csv')

def check_time():
    now = datetime.now()
    h = now.hour
    m = now.minute
    time = h * 100 + m
    today = now.strftime("%Y%m%d")

    if time < 1045 or time > 1515:
        print("예측 가능한 시간대가 아닙니다.")
        sys.exit()

    url = f"{BASEURL}/uapi/domestic-stock/v1/quotations/chk-holiday"
    headers = {
        "content-type": "application/json; charset=utf-8",
        "authorization": f"Bearer {token}",
        "appkey": APPKEY,
        "appsecret": APPSECRET,
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
    
def get_realtime_data(token):
    url = f"{BASEURL}/uapi/domestic-stock/v1/quotations/inquire-time-itemchartprice"
    headers = {
        "content-type": "application/json; charset=utf-8",
        "authorization" : f"Bearer {token}",
        "appkey": APPKEY,
        "appsecret": APPSECRET,
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

        if res.status_code != 200:
            print("실시간 데이터 조회 실패")
            sys.exit()
        
        df.append(pd.DataFrame(res.json()['output2']))
        current_dt -= timedelta(minutes=30)

    full_df = pd.concat(df, ignore_index=True)
    full_df = full_df.drop(columns = ['acml_tr_pbmn'])
    full_df = full_df.drop_duplicates(subset = ['stck_bsop_date', 'stck_cntg_hour'], keep = 'first')
    full_df.columns = ['Date', 'Time', 'Open', 'Close', 'High', 'Low', 'Volume']
    full_df = full_df.sort_values(["Date", "Time"], ascending=False)
    full_df = full_df.head(105)
    full_df = full_df[::-1].reset_index(drop = True)
    full_df[['Open', 'Close', 'High', 'Low', 'Volume']] = full_df[['Open', 'Close', 'High', 'Low', 'Volume']].apply(pd.to_numeric)

    full_df['DateTime'] = pd.to_datetime(full_df['Date'] + full_df['Time'].str.zfill(6), format = '%Y%m%d%H%M%S')
    full_df.set_index('DateTime', inplace = True)

    df_5min = full_df.resample('5min', closed = 'left', label = 'left').agg({
            'Open': 'first',
            'Close': 'last',
            'High': 'max',
            'Low': 'min',
            'Volume': 'sum'
    })
    
    df_5min.reset_index(inplace = True)
    
    return df_5min

def preprocessing_real(data):
    data = data.copy()

    data["Open_return"] = data["Open"].pct_change()
    data["Close_return"] = data["Close"].pct_change()
    data["High_return"] = data["High"].pct_change()
    data["Low_return"] = data["Low"].pct_change()
    data["Volume_return"] = data["Volume"].pct_change()

    data.replace([np.inf, -np.inf], np.nan, inplace=True)
    data.dropna(subset=FEATURES, inplace=True)
    data.reset_index(drop=True, inplace=True)

    scaler_path = os.path.join(DATA_DIR, "scaler.pkl")
    scaler = joblib.load(scaler_path)

    data.loc[:, FEATURES] = scaler.transform(data[FEATURES])
    return data


if __name__ == '__main__':

    check_time()

    df = get_realtime_data(token)

    if df.isna().any().any():
        print("결측값이 존재합니다.")
        sys.exit()

    processed_df = preprocessing_real(df)

    X = processed_df[FEATURES].to_numpy(dtype = np.float32)
    X = np.expand_dims(X, axis=0)

    classifier = tf.keras.models.load_model(os.path.join(ROOT_DIR, "models", "classifier.keras"), compile=False)

    regressor = tf.keras.models.load_model(os.path.join(ROOT_DIR, "models", "regressor.keras"), compile=False)

    classifer_output = classifier.predict(X, verbose=0)[0, 0]
    regressor_output = regressor.predict(X, verbose=0)[0, 0]

    print(f"상승 확률: {classifer_output * 100:.2f}%")
    print(f"예상 수익률: {regressor_output * 100:.3f}%")
    