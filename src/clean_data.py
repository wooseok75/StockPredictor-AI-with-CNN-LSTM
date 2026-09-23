import os
import pandas as pd

# df 들고오기
src_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(src_dir)
data_dir = os.path.join(root_dir, 'data')
df_path = os.path.join(data_dir, '5min_df.csv')

df = pd.read_csv(df_path)

# df 전처리
df['DateTime'] = pd.to_datetime(df['DateTime'])
df['Date'] = df['DateTime'].dt.date

row_counts = df.groupby('Date').size()

vaild_index = row_counts[row_counts == 77].index

df.set_index('Date', inplace = True)

cleaned_df = df.loc[vaild_index].copy()
cleaned_df.reset_index(inplace = True)
cleaned_df.drop(columns = 'Date', inplace = True)

# 삭제된 날짜 확인
deleted_day = row_counts[row_counts != 77]

print("====== {삭제된 날짜 및 데이터 개수} ======")
for date, cnt in deleted_day.items():
    print(f"삭제된 날짜: {date}, 데이터 개수: {cnt}")
print("=======================================")

# 전처리한 데이터 저장
saving_path = os.path.join(data_dir, 'cleaned_df.csv')

cleaned_df.to_csv(saving_path, index = False, encoding = 'UTF-8')

print("데이터 저장 완료!")