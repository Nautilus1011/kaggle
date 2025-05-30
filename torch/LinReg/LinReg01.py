import pandas as pd 
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


train_file_path = '/home/jinysd/workspace/datasets/london-house-price-prediction-advanced-techniques/train.csv'
test_file_path = '/home/jinysd/workspace/datasets/london-house-price-prediction-advanced-techniques/test.csv'


train_df = pd.read_csv(train_file_path)
test_df = pd.read_csv(test_file_path)

target_column = "price"
feature_column = "floorAreaSqM"

df_selected = train_df[[feature_column, target_column]].copy()

df_selected.dropna(subset=[feature_column, target_column], inplace=True)


# Numpy配列に変換
# reshape(-1, 1)でPyTrochが期待する2次元形式[データ数, 特徴量数]に変換
X_np = df_selected[feature_column].values.astype(np.float32).reshape(-1, 1)
y_np = df_selected[target_column].values.astype(np.float32).reshape(-1, 1)

X_train, X_val, y_train, y_val = train_test_split(
    X_np, y_np, test_size=0.2, random_state=42
)

X_train = torch.from_numpy(X_train)
y_train = torch.from_numpy(y_train)
X_val = torch.from_numpy(X_val)
y_val = torch.from_numpy(y_val)



# モデル定義
class LinearRegression(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.linear = nn.Linear(input_size, output_size)
    
    def forward(self, x):
        return self.linear(x)
    
input_dim = 1
output_dim = 1
model = LinearRegression(input_dim, output_dim)


# 損失関数
mse = nn.MSELoss()

# 最適化アルゴリズム
learning_rate = 0.00001
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)


# モデル学習
train_loss_history = []
val_loss_history = []
num_epochs = 50000

for epoch in range(num_epochs):
    
    model.train()
    optimizer.zero_grad()
    outputs_train = model(X_train)
    loss_train = mse(outputs_train, y_train)
    loss_train.backward()
    optimizer.step()
    train_loss_history.append(loss_train.item())

    model.eval()
    with torch.no_grad():
        outputs_val = model(X_val)
        loss_val = mse(outputs_val, y_val)

        val_loss_history.append(loss_val.item())
    
    if (epoch + 1) % 5000 == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}], "
              f"Train Loss: {loss_train.item():.4f},"
              f"Validation Loss: {loss_val.item():.4f}")
        

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
