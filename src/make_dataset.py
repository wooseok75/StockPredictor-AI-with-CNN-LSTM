import os
import pandas as pd
import numpy as np

# 데이터 가져오기
src_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(src_dir)
data_dir = os.path.join(root_dir, 'data')
df_path = os.path.join(data_dir, 'cleaned_df.csv')

df = pd.read_csv(df_path)
df['DateTime'] = pd.to_datetime(df['DateTime'])

X = []
y = []

WINDOW_SIZE = 10

for date, group in df.groupby(df['DateTime'].dt.date):

    for i in range(len(group) - WINDOW_SIZE):
        window = group.iloc[i : i + WINDOW_SIZE]

        current_close = window.iloc[-1]['Close']
        next_close = group.iloc[i + WINDOW_SIZE]['Close']

        if current_close < next_close:
            target = 1
        else:
            target = 0

        window = window[['Open', 'Close', 'High', 'Low', 'Volume']].values

        X.append(window)
        y.append(target)

X = np.array(X)
y = np.array(y)

# 입출력 데이터 shape 확인
print(f"X의 shape: {X.shape}")
print(f"y의 shape: {y.shape}")
print()

# 상승, 하락 데이터 개수 확인
print(f"상승 데이터: {np.sum(y == 1)}개")
print(f"하락 데이터: {np.sum(y == 0)}개")

# dataset 저장
save_path = os.path.join(data_dir, 'dataset.npz')
np.savez(save_path, X=X, y=y)
print("데이터셋 저장 완료!")