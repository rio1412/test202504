import requests
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from dotenv import load_dotenv
import os
import json

# .env ファイルを読み込む
load_dotenv()

# .env ファイルからキーを取得
LINE_ACCESS_TOKEN = os.getenv("LINE_ACCESS_TOKEN")
LINE_USER_ID = os.getenv("LINE_USER_ID")
GOOGLE_CREDENTIALS_PATH = os.getenv("GOOGLE_CREDENTIALS_PATH")

# LINE 通知を送信する関数
def send_line_notify(message):
    url = "https://api.line.me/v2/bot/message/push"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {LINE_ACCESS_TOKEN}"
    }
    payload = {
        "to": LINE_USER_ID,
        "messages": [
            {
                "type": "text",
                "text": message
            }
        ]
    }
    response = requests.post(url, headers=headers, json=payload)
    return response.json()

# Google Sheets API 認証とデータ取得
scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
credentials = ServiceAccountCredentials.from_json_keyfile_name(GOOGLE_CREDENTIALS_PATH, scope)
gc = gspread.authorize(credentials)

SPREADSHEET_NAME = "染め物在庫管理"
SHEET_NAME = "在庫データ"
sheet = gc.open(SPREADSHEET_NAME).worksheet(SHEET_NAME)
data = sheet.get_all_records()
df = pd.DataFrame(data)

# 前処理 (カテゴリ変数のエンコーディング)
df = pd.get_dummies(df, columns=["生地タイプ (fabric_type)", "生地名 (fabric_name)"])

# 最終更新日 (last_updated) を日付型に変換
df["最終更新日 (last_updated)"] = pd.to_datetime(df["最終更新日 (last_updated)"])
df["days_since_update"] = (pd.Timestamp.today() - df["最終更新日 (last_updated)"]).dt.days
df.drop(columns=["最終更新日 (last_updated)"], inplace=True)

# XGBoost モデルのトレーニング
X = df.drop(columns=["価格 (price)"])  # 説明変数
y = df["価格 (price)"]  # 目的変数

model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
model.fit(X, y)

# 予測
y_pred = model.predict(X)
mae = mean_absolute_error(y, y_pred)

# 予測結果を Google Sheets に保存 (オプション)
sheet.update("J1", [["実際価格(Actual Price)", "予想価格(Predicted Price)"]])

# 予測結果を一度にセル範囲に更新
predicted_values = []
for actual, pred in zip(y, y_pred):
    predicted_values.append([actual, pred])

# J2 セルから始めて予測結果を一度に更新
cell_range = f'J2:K{len(predicted_values) + 1}'  # 必要なセル範囲を計算
sheet.update(cell_range, predicted_values)

# Streamlit ダッシュボード
st.title("染め物屋 在庫管理システム")

# 予測結果の表示
st.subheader(f"XGBoost モデルによる予測結果 (MAE: {mae:.2f})")
st.write("以下のグラフは、実際の価格と予測価格を比較したものです。")

# 実際の価格と予測価格と誤差を表形式で表示
comparison_df = pd.DataFrame({
    "実際の価格": y.reset_index(drop=True),
    "予測価格": pd.Series(y_pred).round(2)
})

# 誤差（絶対値差）を追加
comparison_df["誤差"] = (comparison_df["実際の価格"] - comparison_df["予測価格"]).abs().round(2)

# 表の表示
st.subheader("実際の価格と予測価格の比較表（誤差付き）")
st.dataframe(comparison_df.style.format({
    "実際の価格": "{:.2f}",
    "予測価格": "{:.2f}",
    "誤差": "{:.2f}"
}))

# 在庫状況の表示
st.subheader("現在の在庫状況")
st.dataframe(df[["生地ID (fabric_id)", "在庫数 (stock)"]])

# 在庫の可視化
st.subheader("在庫の分布")

# 生地IDごとの合計在庫数を計算
inventory_count = df.groupby("生地ID (fabric_id)")["在庫数 (stock)"].sum().reset_index()

# fig, axを使用してグラフを作成
fig, ax = plt.subplots(figsize=(10, 6))

# barplotを使用して在庫数を表示
sns.barplot(x="生地ID (fabric_id)", y="在庫数 (stock)", data=inventory_count, ax=ax)

# グラフのタイトルとラベル
ax.set_title("Fabric Type Wise Inventory Distribution")
ax.set_xlabel("Fabric ID")
ax.set_ylabel("Total Stock")

# Streamlitに表示
st.pyplot(fig)

# 在庫が少ない場合にアラート通知
st.subheader("在庫警告")
threshold = st.slider("在庫数の閾値", 1, 50, 10)
low_stock_df = df[df["在庫数 (stock)"] < threshold]
if len(low_stock_df) > 0:
    st.write("在庫が少ないアイテムがあります！")
    st.dataframe(low_stock_df[["生地ID (fabric_id)", "在庫数 (stock)"]])

    # LINE 通知を送信
    for index, row in low_stock_df.iterrows():
        fabric_type = row["生地ID (fabric_id)"]
        stock = row["在庫数 (stock)"]
        message = f"在庫警告: {fabric_type}の在庫が少なくなりました。残り{stock}個です。"
        send_line_notify(message)
else:
    st.write("在庫は十分にあります。")
