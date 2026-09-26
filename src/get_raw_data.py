from dotenv import load_dotenv
import os
import time
import requests
import json
import pandas as pd

load_dotenv()

APP_KEY = os.getenv("APP_KEY")
APP_SECRET = os.getenv("APP_SECRET")
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE_URL = 'https://openapi.koreainvestment.com:9443'

def get_access_token(): # 토큰 발급 함수
    token_file = os.path.join(ROOT_DIR, 'token.txt')

    # 기존 토큰 가져오기
    if os.path.exists(token_file): 
        file_age = time.time() - os.path.getmtime(token_file)
        if (file_age <= 12 * 3600):
            with open(token_file, 'r') as f:
                token = f.read().strip()
            if token:
                return token
            
    # 신규 토큰 발급받기
    url = f'{BASE_URL}/oauth2/tokenP' 
    headers = {'content-type': 'application/json; charset=UTF-8'}
    bodys = {
        'grant_type': 'client_credentials',
        'appkey': APP_KEY,
        'appsecret': APP_SECRET
    }

    res = requests.post(url, headers = headers, data = json.dumps(bodys))

    if res.status_code == 200:
        token = res.json()['access_token']
        with open(token_file, 'w') as f:
            f.write(token)
        return token
    else:
        print(res.status_code)
        return None

def get_1min_data(token, target_date): # 1분봉 데이터 수집하는 함수
    all_dfs = []

    url = f"{BASE_URL}/uapi/domestic-stock/v1/quotations/inquire-time-dailychartprice"
    headers = {
        'content-type': 'application/json; charset=utf-8',
        'authorization': f'Bearer {token}',
        'appkey': APP_KEY,
        'appsecret': APP_SECRET,
        'tr_id': 'FHKST03010230',
        'custtype': 'P'
    }
    current_time = '153000'

    while True:
        params = {
            'FID_COND_MRKT_DIV_CODE': 'J',
            'FID_INPUT_ISCD': '069500',
            'FID_INPUT_HOUR_1': f'{current_time}',
            'FID_INPUT_DATE_1': f'{target_date}',
            'FID_PW_DATA_INCU_YN': 'N',
            'FID_FAKE_TICK_INCU_YN': 'N'
        }

        res = requests.get(url, headers = headers, params = params)

        if res.status_code != 200:
            print(f"{res.status_code} 오류 발생!")
            break

        raw_data = res.json()

        if 'output2' not in raw_data or raw_data['output2'] is None:
            print("다른 것 수신받음")
            break

        df = pd.DataFrame(raw_data['output2'])
        all_dfs.append(df)

        earliest_time = df['stck_cntg_hour'].iloc[-1]

        if int(earliest_time) <= 90000:
            break
        if earliest_time == current_time:
            break

        current_time = earliest_time
        time.sleep(0.2)

    # df 가공하기
    full_dfs = pd.concat(all_dfs)
    full_dfs = full_dfs.drop(columns = ['acml_tr_pbmn'])
    full_dfs = full_dfs.drop_duplicates(subset = ['stck_bsop_date', 'stck_cntg_hour'], keep = 'first')
    full_dfs.columns = ['Date', 'Time', 'Open', 'Close', 'High', 'Low', 'Volume']
    full_dfs = full_dfs[::-1].reset_index(drop = True)
    full_dfs = full_dfs[full_dfs['Time'] != '153000']
    full_dfs[['Open', 'Close', 'High', 'Low', 'Volume']] = full_dfs[['Open', 'Close', 'High', 'Low', 'Volume']].apply(pd.to_numeric)

    return full_dfs

def convert_to_5min(df):
    df['DateTime'] = pd.to_datetime(df['Date'] + df['Time'].str.zfill(6), format = '%Y%m%d%H%M%S')
    df.set_index('DateTime', inplace = True)

    df_5min = df.resample('5min', closed = 'right', label = 'right').agg({
        'Open': 'first',
        'Close': 'last',
        'High': 'max',
        'Low': 'min',
        'Volume': 'sum'
    }).dropna()
    df_5min.reset_index(inplace = True)

    return df_5min


if __name__ == '__main__':
    token = get_access_token()

    date_list = pd.date_range(start = '20250922', end = '20260922', freq = 'B').strftime("%Y%m%d").tolist()

    full_df = []

    for target_date in date_list:
        try:
            df_1min = get_1min_data(token, target_date)

            if df_1min is None or df_1min.empty:
                print(f"{target_date}의 값이 없으므로 넘어갑니다.")
                continue

            actual_date = df_1min['Date'].iloc[0]

            if target_date != actual_date:
                print(f"{target_date}는 공휴일이므로 넘어갑니다.")
                continue

            df_5min = convert_to_5min(df_1min)

            if df_5min is not None and not df_5min.empty:
                full_df.append(df_5min)
                print(f"{target_date} 일일 데이터 저장 완료!")

        except Exception as e:
            print(f"{target_date} 오류 발생 -> {e}")

        time.sleep(0.2)

    if full_df:
        master_df = pd.concat(full_df, ignore_index = True)

        save_dir = os.path.join(ROOT_DIR, 'data')
        os.makedirs(save_dir, exist_ok = True)
        file_path = os.path.join(save_dir, 'raw_data.csv')

        master_df.to_csv(file_path, index = False, encoding = 'UTF-8')

        print("1년치 데이터 저장 완료!")