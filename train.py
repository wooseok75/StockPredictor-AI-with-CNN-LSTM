from pathlib import Path
from src.get_raw_data import  get
from src.preprocessing import preprocessing
from src.model import train_models
from src.evaluate import evaluate_models
from datetime import datetime, timedelta
import pandas as pd
import sys

# 경로 설정
ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"

# 데이터 로드 함수
def load_raw_data():
    path = DATA_DIR / "raw_data.csv"

    end = datetime.now().date() - timedelta(days = 1)
    start = end - timedelta(days = 365)

    if path.exists():
        df = pd.read_csv(path)

        dates = pd.to_datetime(df['DateTime'])
        print(f"저장된 데이터: {dates.min():%Y%m%d} ~ {dates.max():%Y%m%d}")

        answer = input("최신 데이터로 다시 수집하시겠습니까? [y/n] ").strip().lower()

        if answer == 'n':
            return df
        elif answer == 'y':
            pass
        else:
            print("y 또는 n을 입력해야 합니다.")
            sys.exit()
    else:
        print("기존 CSV파일이 없어 새로 수집합니다.")

    return get(start.strftime("%Y%m%d"), end.strftime("%Y%m%d"))

# 파일 직접 실행
if __name__ == '__main__':
    raw_data = load_raw_data()
    train_set, val_set, test_set = preprocessing(raw_data)
    train_models(train_set, val_set)
    evaluate_models()