import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# データ前処理関数
def process_df(df):
    df['Age'] = df['Age'].fillna(df['Age'].median())  # 'Age'の欠損値を中央値で補完, fillna(x)でxの値で欠損部分を埋める
    df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})  # 'Sex'を数値化, map('x': y)で文字列'x'を数値y等別の形式に変換
    df['Fare'] = df['Fare'].fillna(df['Fare'].median())  # 'Fare'の欠損値を中央値で補完
    df = df.drop(['Name', 'Ticket', 'Cabin', 'Embarked'], axis=1)  # 不要な列を削除, 引き数[]はリスト, axis=1で列 axis=0で行を指定    
    return df

# データセットの読み込みと前処理
df_train = pd.read_csv(r'E:\repository\kaggle\titanic\data\train.csv')
df_train = process_df(df_train)
# コンペ提出用テストデータであるからsurvived列は含まれていない。つまりモデル精度を事前に評価するには
# df_trainデータをさらにtrain, testに分ける必要がある
df_test = pd.read_csv(r'E:\repository\kaggle\titanic\data\test.csv')
df_test = process_df(df_test)



# 特徴量と目的変数に分割
X = df_train.drop(['Survived'], axis=1)
y = df_train['Survived']

# 訓練データとテストデータに分割
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ロジスティック回帰モデルの学習
model = LogisticRegression(max_iter=500)  # 収束しやすいように最大イテレーションを増やす
model.fit(X_train, y_train)

# 予測
y_pred = model.predict(X_test)

# 精度評価
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")
