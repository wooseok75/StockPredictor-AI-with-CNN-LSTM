import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib


SRC_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SRC_DIR)
DATA_DIR = os.path.join(ROOT_DIR, 'data')
RAW_DATA_PATH = os.path.join(DATA_DIR, 'raw_data.csv')

FEATURES = ["Open_return", "Close_return", "High_return", "Low_return", "Volume_return"]

WINDOW_SIZE = 20

def remove_incomplete_days(df):
    print("[불완전한 날짜 삭제 중...]\n")

    df = df.copy()

    df["DateTime"] = pd.to_datetime(df["DateTime"])
    df["Date"] = df["DateTime"].dt.date

    df = df.sort_values("DateTime").reset_index(drop=True)

    # 날짜별 5분봉 개수
    row_counts = df.groupby("Date").size()

    # OHLC 중 하나라도 NaN이 존재하는 날짜
    missing_dates = df.loc[
        df[["Open", "Close", "High", "Low"]].isna().any(axis=1),
        "Date"
    ].unique()

    valid_dates = row_counts[
        (row_counts == 76) &
        (~row_counts.index.isin(missing_dates))
    ].index

    complete_days_df = df[df["Date"].isin(valid_dates)].copy()

    removed_dates = row_counts.index.difference(valid_dates)

    for date in removed_dates:
        print(f"삭제된 날짜: {date}")

    complete_days_df.drop(columns="Date", inplace=True)
    complete_days_df.reset_index(drop=True, inplace=True)

    print(f"\n삭제된 거래일: {len(removed_dates)}일")
    print("[불완전한 날짜 삭제 완료!]\n")

    return complete_days_df

def split_train_val_test(df): # 데이터셋 분할 함수
    print("[데이터셋 분할 중..]\n")

    start_date = df["DateTime"].min().normalize()

    train_end = start_date + pd.DateOffset(months = 8)
    val_end = train_end + pd.DateOffset(months = 2)

    train_df = df[df['DateTime'] < train_end].copy()
    val_df = df[(df['DateTime'] >= train_end) & (df['DateTime'] < val_end)].copy()
    test_df = df[df['DateTime'] >= val_end].copy()

    print(f"(train셋 개수: {train_df.shape[0]})")
    print(f"(val셋 개수: {val_df.shape[0]})")
    print(f"(test셋 개수: {test_df.shape[0]})\n")
    print("[데이터셋 분할 완료!]\n")

    return train_df, val_df, test_df

def create_features(train_df, val_df, test_df): # return feature 만드는 함수
    print("[feature 생성 중...]\n")

    for data in [train_df, val_df, test_df]:
        data["Open_return"] = data.groupby(data["DateTime"].dt.date)["Open"].pct_change()
        data["Close_return"] = data.groupby(data["DateTime"].dt.date)["Close"].pct_change()
        data["High_return"] = data.groupby(data["DateTime"].dt.date)["High"].pct_change()
        data["Low_return"] = data.groupby(data["DateTime"].dt.date)["Low"].pct_change()
        data["Volume_return"] = data.groupby(data["DateTime"].dt.date)["Volume"].pct_change()

    for data in [train_df, val_df, test_df]:
        data.replace([np.inf, -np.inf], np.nan, inplace=True)
        data.dropna(subset=FEATURES, inplace=True)
        data.reset_index(drop=True, inplace=True)

    print("[feature 생성 완료!]\n")

    return train_df, val_df, test_df

def scaling(train_df, val_df, test_df): # 표준화 함수
    print("[표준화 중..]\n")

    scaler = StandardScaler()

    # Train으로만 기준
    scaler.fit(train_df[FEATURES])

    # 동일한 scaler 적용
    train_df.loc[:, FEATURES] = scaler.transform(train_df[FEATURES])
    val_df.loc[:, FEATURES] = scaler.transform(val_df[FEATURES])
    test_df.loc[:, FEATURES] = scaler.transform(test_df[FEATURES])

    scaler_path = os.path.join(DATA_DIR, "scaler.pkl")
    joblib.dump(scaler, scaler_path)

    print("[표준화 완료!]\n")

    return train_df, val_df, test_df

def create_sequences(train_df, val_df, test_df):
    print("[시퀀스 데이터 생성 중...]\n")

    datasets = {}

    for name, data in [("train", train_df), ("val", val_df), ("test", test_df)]:
        X = []
        y_class = []
        y_reg = []

        for date, group in data.groupby(data["DateTime"].dt.date):
            group = group.sort_values("DateTime").reset_index(drop=True)

            for i in range(len(group) - WINDOW_SIZE):
                window = group.iloc[i:i + WINDOW_SIZE]

                current_close = window.iloc[-1]["Close"]
                next_close = group.iloc[i + WINDOW_SIZE]["Close"]

                target_class = int(next_close > current_close)

                target_reg = (next_close - current_close) / current_close

                X.append(window[FEATURES].values)
                y_class.append(target_class)
                y_reg.append(target_reg)

        X = np.array(X)
        y_class = np.array(y_class)
        y_reg = np.array(y_reg)

        np.savez(os.path.join(DATA_DIR, f"{name}.npz"), X = X, y_class = y_class, y_reg = y_reg)

        datasets[name] = {"X": X, "y_class": y_class, "y_reg": y_reg}

    print("\n[시퀀스 데이터 생성 및 저장 완료!]\n")

    return datasets

def preprocess(raw_data):
    complete_days_df = remove_incomplete_days(raw_data)

    train_df, val_df, test_df = split_train_val_test(complete_days_df)

    train_df, val_df, test_df = create_features(train_df, val_df, test_df)

    train_df, val_df, test_df = scaling(train_df, val_df, test_df)

    return train_df, val_df, test_df