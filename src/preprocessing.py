import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SRC_DIR)
DATA_DIR = os.path.join(ROOT_DIR, 'data')
RAW_DATA_PATH = os.path.join(DATA_DIR, 'raw_data.csv')

FEATURES = ["Open", "Close", "High", "Low", "Volume"]

WINDOW_SIZE = 10

def get_raw_data(): # 원시데이터 가져오는 함수
    print("[원시 데이터 가져오는 중..]\n")

    raw_data = pd.read_csv(RAW_DATA_PATH)

    if raw_data.empty:
        print("[csv파일이 비어있습니다.]\n")
    else:
        print(f"{raw_data.head()}\n")
        print("[원시 데이터 가져오기 완료!]\n")

    return raw_data

def remove_incomplete_days(df): # 불완전한 데이터 지우는 함수
    print("[불완전한 날짜 삭제 중...]\n")

    df['DateTime'] = pd.to_datetime(df['DateTime'])
    df['Date'] = df['DateTime'].dt.date

    df = df.sort_values('DateTime').reset_index(drop=True)

    row_counts = df.groupby('Date').size()

    vaild_index = row_counts[row_counts == 77].index

    df.set_index('Date', inplace = True)

    complete_days_df = df.loc[vaild_index].copy()
    complete_days_df.reset_index(inplace = True)
    complete_days_df.drop(columns = 'Date', inplace = True)

    incomplete_days = row_counts[row_counts != 77]

    for date, cnt in incomplete_days.items():
        print(f"삭제된 날짜: {date}, 데이터 개수: {cnt}")

    print("\n[불완전한 날짜 삭제 완료!]\n")

    return complete_days_df

def split_train_val_test(df): # 데이터셋 분할 함수
    print("[데이터셋 분할 중..]\n")

    train_df = df[
        (df['DateTime'] >= '2025-09-01') &
        (df['DateTime'] < '2026-05-01')
    ].copy()

    val_df = df[
        (df['DateTime'] >= '2026-05-01') &
        (df['DateTime'] < '2026-07-01')
    ].copy()

    test_df = df[
        (df['DateTime'] >= '2026-07-01') &
        (df['DateTime'] < '2026-09-22')
    ].copy()

    print(f"(train셋 개수: {train_df.shape[0]})")
    print(f"(val셋 개수: {val_df.shape[0]})")
    print(f"(test셋 개수: {test_df.shape[0]})\n")
    print("[데이터셋 분할 완료!]\n")

    return train_df, val_df, test_df

def Scaling(train_df, val_df, test_df): # 표준화 시키는 함수
    print("[표준화 중..]\n")

    scaler = StandardScaler()
    scaler.fit(train_df[FEATURES])
    
    train_df[FEATURES] = train_df[FEATURES].astype(float)
    val_df[FEATURES] = val_df[FEATURES].astype(float)
    test_df[FEATURES] = test_df[FEATURES].astype(float)

    train_df.loc[:, FEATURES] = scaler.transform(train_df[FEATURES])
    val_df.loc[:, FEATURES] = scaler.transform(val_df[FEATURES])
    test_df.loc[:, FEATURES] = scaler.transform(test_df[FEATURES])

    scaler_path = os.path.join(DATA_DIR, 'scaler.pkl')
    joblib.dump(scaler, scaler_path)

    print(f"{train_df.head()}\n")
    print("[표준화 완료!]\n")

    return train_df, val_df, test_df

def create_sequences(df):
    print("[시퀀스 데이터 생성 중...]\n")

    X = []
    y = []

    for date, group in df.groupby(df['DateTime'].dt.date):
        for i in range(len(group) - WINDOW_SIZE):
            window = group.iloc[i : i + WINDOW_SIZE]

            current_close = window.iloc[-1]['Close']
            next_close = group.iloc[i + WINDOW_SIZE]['Close']

            if current_close < next_close:
                target = 1
            else:
                target = 0

            X.append(window[FEATURES].values)
            y.append(target)

    X = np.array(X)
    y = np.array(y)

    print(f"X의 shape: {X.shape}")
    print(f"y의 shape: {y.shape}\n")

    print(f"상승 데이터의 개수: {np.sum(y == 1)}")
    print(f"하락/유지 데이터의 개수: {np.sum(y == 0)}\n")

    print("[시퀀스 데이터 생성 완료!]\n")

    return X, y


if __name__ == '__main__':
    raw_data = get_raw_data()

    complete_days_df = remove_incomplete_days(raw_data)

    train_df, val_df, test_df = split_train_val_test(complete_days_df)

    train_df, val_df, test_df = Scaling(train_df, val_df, test_df)

    X_train, y_train = create_sequences(train_df)
    X_val, y_val = create_sequences(val_df)
    X_test, y_test = create_sequences(test_df)

    np.savez(os.path.join(DATA_DIR, 'train.npz'), X = X_train, y = y_train)
    np.savez(os.path.join(DATA_DIR, 'val.npz'), X = X_val, y = y_val)
    np.savez(os.path.join(DATA_DIR, 'test.npz'), X = X_test, y = y_test)

    print("[데이터셋 저장 완료!]")