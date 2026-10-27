from src import get_raw_data, preprocessing, model
from datetime import datetime, timedelta
import os
import sys
import pandas as pd

# 실시간 날짜 정보 조회
today = datetime.now().date()

end_date = today - timedelta(days = 1)
start_date = end_date - timedelta(days = 365)

end_date = end_date.strftime("%Y%m%d")
start_date = start_date.strftime("%Y%m%d")

# 원시 데이터 가져오기
if os.path.exists("data/raw_data.csv"):
    raw_data = pd.read_csv("data/raw_data.csv")
    dates = pd.to_datetime(raw_data['DateTime'])

    data_start = dates.min().strftime("%Y%m%d")
    data_end = dates.max().strftime("%Y%m%d")

    print(f"현재 {data_start} ~ {data_end}의 데이터가 저장되어 있습니다.")
    answer = input("최신화 하시겠습니까? [y/n]\n")

    if answer == 'y':
        raw_data = get_raw_data.get(start_date, end_date)

    elif answer == 'n':
        print("기존 데이터를 사용합니다.")

    else:
        print("잘못된 입력입니다. 프로그램을 종료합니다.")
        sys.exit()

else:
    print("데이터 파일이 없습니다. 새로 받아옵니다.")
    raw_data = get_raw_data.get(start_date, end_date)

# 데이터 전처리 하기
train_df, val_df, test_df = preprocessing.preprocess(raw_data)

datasets = preprocessing.create_sequences(train_df, val_df, test_df)

model.train_models(datasets)