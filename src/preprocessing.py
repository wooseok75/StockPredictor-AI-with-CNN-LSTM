from pathlib import Path
from datetime import timedelta
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib

# 경로 설정
DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# 설정변수 선언
WINDOW_SIZE = 20
FEATURES = ['Open', 'Close', 'High', 'Low', 'Volume']

# 데이터셋 분할 함수
def split_data(df):
    print("[데이터 분할을 시작합니다.]")

    df['DateTime'] = pd.to_datetime(df['DateTime'])

    start = df["DateTime"].min().normalize()
    train_end = start + pd.DateOffset(months = 8)
    val_end = train_end + pd.DateOffset(months = 2)

    train_df = df[df['DateTime'] < train_end].copy()
    val_df = df[(df['DateTime'] < val_end) & (df['DateTime'] >= train_end)].copy()
    test_df = df[df['DateTime'] >= val_end].copy()

    print(f"(분할 결과) train셋: {len(train_df)}개, val셋: {len(val_df)}개, test셋: {len(test_df)}개")

    return train_df, val_df, test_df

# 시퀀스 생성 함수
def create_sequence(df, file_name = None):
    X, y_class, y_reg = [], [], []

    for _, group in df.groupby(df['DateTime'].dt.date):
        for i in range(len(group) - WINDOW_SIZE):
            window = group.iloc[i:i+WINDOW_SIZE]
            current_close = window['Close_copy'].iloc[-1]
            next_close = group['Close_copy'].iloc[i+WINDOW_SIZE]

            X.append(window[FEATURES].to_numpy())
            y_class.append(int(current_close < next_close))
            y_reg.append((next_close - current_close) / current_close)

    X = np.array(X)
    y_class = np.array(y_class)
    y_reg = np.array(y_reg)

    if file_name:
        np.savez(DATA_DIR / f"{file_name}.npz", X = X, y_class = y_class, y_reg = y_reg)

    dataset = {
        "X": X,
        "y_class": y_class,
        "y_reg": y_reg
    }

    return dataset

# 정규화 함수
def Scaling(train_df, val_df, test_df):
    Scaler = StandardScaler()

    for df in [train_df, val_df, test_df]:
        df['Close_copy'] = df['Close'].copy()

    train_df[FEATURES] = Scaler.fit_transform(train_df[FEATURES])
    val_df[FEATURES] = Scaler.transform(val_df[FEATURES])
    test_df[FEATURES] = Scaler.transform(test_df[FEATURES])

    joblib.dump(Scaler, DATA_DIR / "scaler.pkl")

    return train_df, val_df, test_df

# 실행 함수
def preprocessing(raw_data):
    train_df, val_df, test_df = split_data(raw_data)

    train_df, val_df, test_df = Scaling(train_df, val_df, test_df)

    train_set = create_sequence(train_df, "train")
    val_set = create_sequence(val_df, "val")
    test_set = create_sequence(test_df, "test")

    return train_set, val_set, test_set