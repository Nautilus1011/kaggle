import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import warnings
from sklearn.model_selection import train_test_split
import gc # メモリ解放のため

# 警告を非表示にする設定 (任意だが、Notebookでの警告表示が煩わしい場合)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

print("--- スクリプト開始 ---")

# --- 1. データ準備 ---
print("\n--- 1. データ準備 ---")

# CSVファイルのパス (VS Codeでtrain.csvとtest.csvが同じディレクトリにある場合)
# もし別の場所にある場合は、適切なパスに修正してください
train_file_path = '/home/jinysd/workspace/datasets/london-house-price-prediction-advanced-techniques/train.csv'
test_file_path = '/home/jinysd/workspace/datasets/london-house-price-prediction-advanced-techniques/test.csv'

try:
    train_df = pd.read_csv(train_file_path)
    test_df = pd.read_csv(test_file_path)
    print(f"データ読み込み完了: {train_file_path}, {test_file_path}")
except FileNotFoundError:
    print(f"エラー: '{train_file_path}' または '{test_file_path}' が見つかりません。")
    print("スクリプトと同じディレクトリにCSVファイルを置くか、パスを修正してください。")
    exit() # ファイルが見つからない場合は終了

# 目的変数と特徴量の定義
target_column = 'price' # 住宅価格
feature_column = 'floorAreaSqM' # 床面積

# 必要な列のみを抽出
# .copy() を使って元のDataFrameへの参照ではなく新しいDataFrameを作成
df_selected = train_df[[feature_column, target_column]].copy()

# 欠損値の処理: 該当する列の欠損値を含む行を削除
print(f"処理前のデータ形状: {df_selected.shape}")
df_selected.dropna(subset=[feature_column, target_column], inplace=True)
print(f"欠損値処理後のデータ形状: {df_selected.shape}")

# NumPy配列に変換
# reshape(-1, 1) でPyTorchが期待する2次元形式 [データ数, 特徴量数] に変換
X_np = df_selected[feature_column].values.astype(np.float32).reshape(-1, 1)
y_np = df_selected[target_column].values.astype(np.float32).reshape(-1, 1)

# 訓練データと検証データに分割 (20%を検証データに)
# random_state を固定することで、毎回同じ分割になるようにし、再現性を保つ
X_train_np, X_val_np, y_train_np, y_val_np = train_test_split(
    X_np, y_np, test_size=0.2, random_state=42
)

# NumPy配列からPyTorchテンソルに変換
X_train_tensor = torch.from_numpy(X_train_np)
y_train_tensor = torch.from_numpy(y_train_np)
X_val_tensor = torch.from_numpy(X_val_np)
y_val_tensor = torch.from_numpy(y_val_np)

print(f"\n訓練データXの形状: {X_train_tensor.shape}")
print(f"訓練データyの形状: {y_train_tensor.shape}")
print(f"検証データXの形状: {X_val_tensor.shape}")
print(f"検証データyの形状: {y_val_tensor.shape}")
print(f"訓練データXのサンプル:\n{X_train_tensor[:5]}")
print(f"訓練データyのサンプル:\n{y_train_tensor[:5]}")

# メモリ解放のため、不要になった大きなPandas DataFrameやNumPy配列を削除
del train_df, test_df, df_selected, X_np, y_np, X_train_np, X_val_np, y_train_np, y_val_np
gc.collect()


# --- 2. モデル定義 ---
print("\n--- 2. モデル定義 ---")

class LinearRegression(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        # 線形層を定義: 入力サイズ1、出力サイズ1
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, x):
        # 順伝播の計算: 入力 x を線形層に通す
        return self.linear(x)

# 単回帰なので、入力も出力も次元は1
input_dim = 1
output_dim = 1
model = LinearRegression(input_dim, output_dim)

print(f"モデル構造:\n{model}")


# --- 3. 損失関数と最適化アルゴリズムの定義 ---
print("\n--- 3. 損失関数と最適化アルゴリズムの定義 ---")

# 平均二乗誤差 (MSE) を損失関数として使用
criterion = nn.MSELoss()

# 確率的勾配降下法 (SGD) を最適化アルゴリズムとして使用
# ★重要: 学習率を非常に小さく設定！価格のスケールが大きいため発散しやすいです。
# 0.0000001 から試し、必要に応じて調整してください
learning_rate = 0.0000001
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

print(f"学習率: {learning_rate}")


# --- 4. モデルの学習（訓練ループ） ---
print("\n--- 4. モデルの学習開始 ---")

train_loss_history = [] # 訓練損失の履歴を記録
val_loss_history = []   # 検証損失の履歴を記録
num_epochs = 50000      # 学習のエポック数

for epoch in range(num_epochs):
    # --- 訓練フェーズ ---
    model.train() # モデルを訓練モードに設定 (Dropoutなどが有効になるが、線形回帰では影響なし)
    optimizer.zero_grad() # 前のエポックの勾配をゼロに初期化
    
    # 順伝播: 訓練データで予測を実行
    outputs_train = model(X_train_tensor)
    
    # 損失計算: 予測結果と正解データから訓練損失を計算
    loss_train = criterion(outputs_train, y_train_tensor)
    
    # 逆伝播: 損失の勾配を計算
    loss_train.backward()
    
    # パラメータ更新: 勾配と学習率を使ってモデルの重みを更新
    optimizer.step() 
    
    # 訓練損失を履歴に追加
    train_loss_history.append(loss_train.item())

    # --- 検証フェーズ ---
    model.eval() # モデルを評価モードに設定 (Dropoutなどが無効になる)
    with torch.no_grad(): # 勾配計算を無効化 (メモリと時間の節約)
        # 順伝播: 検証データで予測を実行
        outputs_val = model(X_val_tensor)
        
        # 損失計算: 検証データでの損失を計算
        loss_val = criterion(outputs_val, y_val_tensor)
        
        # 検証損失を履歴に追加
        val_loss_history.append(loss_val.item())

    # 進捗の表示 (5000エポックごと)
    if (epoch + 1) % 5000 == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}], "
              f"Train Loss: {loss_train.item():.4f}, "
              f"Validation Loss: {loss_val.item():.4f}")

print("--- 学習完了！ ---")


# --- 5. 結果の可視化と評価 ---
print("\n--- 5. 結果の可視化と評価 ---")

# 損失の推移をプロット
plt.figure(figsize=(12, 6))
plt.plot(train_loss_history, label="Train Loss", color='blue', alpha=0.8)
plt.plot(val_loss_history, label="Validation Loss", color='red', alpha=0.8)
plt.xlabel("Epoch")
plt.ylabel("Loss (MSE)")
plt.title("Training and Validation Loss Over Epochs")
plt.legend()
plt.grid(True)
plt.yscale('log') # 損失が急激に変化する場合、対数スケールで見ると分かりやすい
plt.show()

# 訓練データに対する予測値の生成
model.eval() # 評価モードに設定（念のため）
with torch.no_grad(): # 勾配計算をしない
    predicted_train_tensor = model(X_train_tensor)
    predicted_train_np = predicted_train_tensor.numpy()

# 検証データに対する予測値の生成
with torch.no_grad():
    predicted_val_tensor = model(X_val_tensor)
    predicted_val_np = predicted_val_tensor.numpy()

# 元のデータと予測データを散布図でプロット
plt.figure(figsize=(12, 8))

# 訓練データ点を表示
plt.scatter(X_train_tensor.numpy(), y_train_tensor.numpy(),
            label=f"Original Train Data ({feature_column} vs {target_column})",
            color="lightcoral", alpha=0.5, s=10) # 色を薄くして、予測線を見やすく

# 訓練データに対する予測線を引く (線形回帰なので線として表示)
# X_train の最小値から最大値までをカバーする新しいテンソルを作成
x_range = torch.tensor([[X_train_tensor.min().item()], [X_train_tensor.max().item()]], dtype=torch.float32)
with torch.no_grad():
    y_range_pred = model(x_range).numpy()
plt.plot(x_range.numpy(), y_range_pred, color="blue", linewidth=2, label="Predicted Regression Line (Train)")


# 検証データ点を表示
plt.scatter(X_val_tensor.numpy(), y_val_tensor.numpy(),
            label=f"Original Validation Data",
            color="lightgreen", alpha=0.7, s=15) # 色を薄くして、予測線を見やすく

# 検証データに対する予測線を引く
x_val_range = torch.tensor([[X_val_tensor.min().item()], [X_val_tensor.max().item()]], dtype=torch.float32)
with torch.no_grad():
    y_val_range_pred = model(x_val_range).numpy()
# 検証データは訓練データと同じ予測線に沿うはずなので、あえて別の線は引かないことが多い
# しかし、ここでは分かりやすさのため、検証データの範囲で線を引くことも可能

plt.xlabel(feature_column)
plt.ylabel(target_column)
plt.title(f"Original vs Predicted {target_column} by {feature_column} (Train & Val)")
plt.legend()
plt.grid(True)
plt.show()


# 特定の新しい値での予測例 (例: 床面積が100平方メートルの場合の価格予測)
new_floor_area_sqm = torch.tensor([[100.0]], dtype=torch.float32)
with torch.no_grad(): # 勾配計算を無効化
    predicted_price = model(new_floor_area_sqm).item() # .item()でPyTorchテンソルからPythonの数値に変換
print(f"\n予測: 床面積が {new_floor_area_sqm.item()} 平方メートルの場合の価格: £{predicted_price:,.2f}")

# 学習した重みとバイアスを表示
learned_weight = model.linear.weight.item()
learned_bias = model.linear.bias.item()
print(f"学習された重み (Weight): {learned_weight:.4f}")
print(f"学習されたバイアス (Bias): {learned_bias:.4f}")
print(f"予測式: {target_column} = {learned_weight:.4f} * {feature_column} + {learned_bias:.4f}")

print("\n--- スクリプト終了 ---")