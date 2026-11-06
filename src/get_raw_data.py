from pathlib import Path
from dotenv import load_dotenv
import os
import time
import requests
import sys
import pandas as pd

# 경로 설정
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"

# API 설정
load_dotenv(ROOT_DIR / ".env")

APP_KEY = os.getenv("APP_KEY")
APP_SECRET = os.getenv("APP_SECRET")

BASE_URL = "https://openapi.koreainvestment.com:9443"

# Token 발급 함수
def get_access_token():
    token_file = ROOT_DIR / "token.txt"

    if token_file.exists() and time.time() - os.path.getmtime(token_file) <= 12 * 60 * 60:
        with open(token_file, "r") as f:
            token = f.read().strip()
        if token:
            return token

    url = f"{BASE_URL}/oauth2/tokenP"
    bodys = {
        "grant_type": "client_credentials",
        "appkey": APP_KEY,
        "appsecret": APP_SECRET
    }

    res = requests.post(url, json = bodys, timeout = 10)
    res.raise_for_status()

    data = res.json()

    token = data.get("access_token")

    if not token:
        print("토큰 발급 실패")
        sys.exit()

    with open(token_file, "w") as f:
        f.write(token)

    return token

# 1분봉 수집 함수    
def get_1min_data(token, target_date, stock_code):
    all_dfs = []

    url = f"{BASE_URL}/uapi/domestic-stock/v1/quotations/inquire-time-dailychartprice"
    headers = {
        "content-type": "application/json; charset=utf-8",
        "authorization": f"Bearer {token}",
        "appkey": APP_KEY,
        "appsecret": APP_SECRET,
        "tr_id": "FHKST03010230",
        "custtype": "P"
    }
    current_time = "153000"

    while True:
        params = {
            "FID_COND_MRKT_DIV_CODE": "J",
            "FID_INPUT_ISCD": stock_code,
            "FID_INPUT_HOUR_1": current_time,
            "FID_INPUT_DATE_1": target_date,
            "FID_PW_DATA_INCU_YN": "N",
            "FID_FAKE_TICK_INCU_YN": ""
        }

        res = requests.get(url, headers = headers, params = params, timeout = 10)
        res.raise_for_status()

        data = res.json()
        if data.get("rt_cd") != "0":
            raise RuntimeError(f"API 오류 발생: {data.get('msg1')}")

        rows = data.get("output2")

        if not rows:
            break

        df = pd.DataFrame(rows)
        all_dfs.append(df)

        earliest_time = df['stck_cntg_hour'].min()
        
        if int(earliest_time) == 90000 or earliest_time == current_time:
            break
        
        current_time = earliest_time
        time.sleep(0.2)
        
    full_df = pd.concat(all_dfs)
    full_df = full_df.drop(columns = ['acml_tr_pbmn'])
    full_df = full_df.drop_duplicates(subset = ['stck_bsop_date', 'stck_cntg_hour'], keep = 'first')
    full_df.columns = ['Date', 'Time', 'Close', 'Open', 'High', 'Low', 'Volume']
    full_df = full_df[['Date', 'Time', 'Open', 'Close', 'High', 'Low', 'Volume']]
    full_df = full_df[full_df['Time'] != '153000']
    full_df[['Open', 'Close', 'High', 'Low', 'Volume']] = full_df[['Open', 'Close', 'High', 'Low', 'Volume']].apply(pd.to_numeric)
    full_df = full_df[full_df['Volume'] != 0]
    full_df = full_df.sort_values("Time").reset_index(drop = True)

    return full_df

# 1분봉 -> 5분봉 변환 함수
def convert_to_5min(df):
    df['DateTime'] = pd.to_datetime(df['Date'] + df['Time'], format = "%Y%m%d%H%M%S")
    df.set_index("DateTime", inplace = True)

    df_5min = df.resample("5min", closed = "left", label = "left").agg({
        "Open": "first",
        "Close": "last",
        "High": "max",
        "Low": "min",
        "Volume": "sum",

        'Samsung_Open': 'first',
        'Samsung_Close': 'last',
        'Samsung_High': 'max',
        'Samsung_Low': 'min',
        'Samsung_Volume': 'sum',

        'Hynix_Open': 'first',
        'Hynix_Close': 'last',
        'Hynix_High': 'max',
        'Hynix_Low': 'min',
        'Hynix_Volume': 'sum'
    })
    df_5min.reset_index(inplace = True)
    
    return df_5min

# 실행 함수
def get(START, END):
    token = get_access_token()

    collected = []
    failed = []

    kodex_data = {}
    samsung_data = {}
    hynix_data = {}
    
    target_dates = pd.date_range(start = START, end = END, freq = 'B').strftime("%Y%m%d")

    for target_date in target_dates:
        try:
            df = get_1min_data(token, target_date, "069500")
            if df.empty:
                print(f"{target_date}: 거래 데이터 없음")
                continue

            actual_date = df['Date'].iloc[0]

            if target_date != actual_date:
                print(f"{target_date}: (KODEX200) 데이터 수집 실패 - 공휴일입니다.")
                continue

            kodex_data[target_date] = df
            print(f"{target_date}: (KODEX200) 데이터 수집 완료")

        except Exception as e:
            print(f"{target_date}: (KODEX200) 데이터 수집 실패 - {e}")

        finally:    
            time.sleep(0.2)

    for target_date in target_dates:
        try:
            df = get_1min_data(token, target_date, "005930")
            if df.empty:
                print(f"{target_date}: 거래 데이터 없음")
                continue
    
            actual_date = df['Date'].iloc[0]
    
            if target_date != actual_date:
                print(f"{target_date}: (삼성전자) 데이터 수집 실패 - 공휴일입니다.")
                continue
    
            samsung_data[target_date] = df
            print(f"{target_date}: (삼성전자) 데이터 수집 완료")
    
        except Exception as e:
            print(f"{target_date}: (삼성전자) 데이터 수집 실패 - {e}")
    
        finally:    
            time.sleep(0.2)

    for target_date in target_dates:
        try:
            df = get_1min_data(token, target_date, "000660")
            if df.empty:
                print(f"{target_date}: 거래 데이터 없음")
                continue
        
            actual_date = df['Date'].iloc[0]
        
            if target_date != actual_date:
                print(f"{target_date}: (하이닉스) 데이터 수집 실패 - 공휴일입니다.")
                continue
        
            hynix_data[target_date] = df
            print(f"{target_date}: (하이닉스) 데이터 수집 완료")
        
        except Exception as e:
            print(f"{target_date}: (하이닉스) 데이터 수집 실패 - {e}")
        
        finally:    
            time.sleep(0.2)

    for target_date in target_dates:
        try:
            kodex = kodex_data[target_date].set_index(['Date', 'Time'])
            samsung = samsung_data[target_date].set_index(['Date', 'Time']).add_prefix('Samsung_')
            hynix = hynix_data[target_date].set_index(['Date', 'Time']).add_prefix('Hynix_')

            df_1min = pd.concat([kodex, samsung, hynix], axis=1)

            df_1min = df_1min.dropna().sort_index()
            df_1min = df_1min[(df_1min['Volume'] > 0) & (df_1min['Samsung_Volume'] > 0) & (df_1min['Hynix_Volume'] > 0)]
            df_1min = df_1min.reset_index()

            if len(df_1min) != 380:
                raise ValueError(f"1분봉 데이터 불완전: {len(df_1min)}개")
            
            df_5min = convert_to_5min(df_1min)
            collected.append(df_5min)

            print(f'{target_date}: 세 종목 수집 완료')

        except Exception as e:
            failed.append(target_date)
            print(f'{target_date}: 데이터 수집 실패 - {e}')

    if collected:
        DATA_DIR.mkdir(exist_ok = True)

        master_df = pd.concat(collected, ignore_index = True)
        master_df.to_csv(DATA_DIR / "raw_data.csv", index = False, encoding = "utf-8")
        print(f"데이터 저장 완료 (수집 실패 날짜: {failed})")

        return master_df