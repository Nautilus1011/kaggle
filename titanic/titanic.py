import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# データ前処理関数
def process_df(df):
    df['Age'] = df['Age'].fillna(df['Age'].median())  # 'Age'の欠損値を中央値で補完
    df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})  # 'Sex'を数値化
    df['Fare'] = df['Fare'].fillna(df['Fare'].median())  # 'Fare'の欠損値を中央値で補完
    df = df.drop(['Name', 'Ticket', 'Cabin', 'Embarked'], axis=1)  # 不要な列を削除
    return df

# データセットの読み込みと前処理
df_train = pd.read_csv(r'E:\repository\kaggle\titanic\train.csv')
df_train = process_df(df_train)

df_test = pd.read_csv(r'E:\repository\kaggle\titanic\test.csv')
df_test = process_df(df_test)

# 特徴量と目的変数に分割
X = df_train.drop(['Survived'], axis=1)
y = df_train['Survived']

# データを学習用とテスト用に分割
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# モデルの学習
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# 予測
y_pred_test = model.predict(X_test)

# 精度の評価
accuracy = accuracy_score(y_test, y_pred_test)
print(f"Accuracy: {accuracy:.4f}")
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred_test))

# テストデータに対する予測
y_pred_submission = model.predict(df_test)

# 結果の保存
output = pd.DataFrame({"PassengerId": df_test["PassengerId"], "Survived": y_pred_submission})
output.to_csv('submission.csv', index=False)
