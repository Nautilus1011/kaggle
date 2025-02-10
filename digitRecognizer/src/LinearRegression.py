import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# データの定義
car_prices_array = np.array([3, 4, 5, 6, 7, 8, 9], dtype=np.float32).reshape(-1, 1)
number_of_car_sell_array = np.array([7.5, 7, 6.5, 6.0, 5.5, 5.0, 4.5], dtype=np.float32).reshape(-1, 1)

# NumPy配列をPyTorchテンソルに変換
car_price_tensor = torch.from_numpy(car_prices_array)
number_of_car_sell_tensor = torch.from_numpy(number_of_car_sell_array)

# データの可視化
plt.scatter(car_prices_array, number_of_car_sell_array, color='red')
plt.xlabel("Car Price $")
plt.ylabel("Number of Car Sell")
plt.title("Car Price$ VS Number of Car Sell")
plt.show()

# 線形回帰モデルの定義
class LinearRegression(nn.Module):
    def __init__(self, input_size, output_size):
        super(LinearRegression, self).__init__()
        self.linear = nn.Linear(input_size, output_size)
    
    def forward(self, x):
        return self.linear(x)

# モデルの作成
input_dim = 1
output_dim = 1
model = LinearRegression(input_dim, output_dim)

# 損失関数と最適化手法
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.02)

# 学習
num_epochs = 1001
loss_list = []

for epoch in range(num_epochs):
    optimizer.zero_grad()
    outputs = model(car_price_tensor)
    loss = criterion(outputs, number_of_car_sell_tensor)
    loss.backward()
    optimizer.step()
    loss_list.append(loss.item())
    
    if epoch % 50 == 0:
        print(f'Epoch {epoch}, Loss: {loss.item()}')

# 損失の推移をプロット
plt.plot(range(num_epochs), loss_list)
plt.xlabel("Number of Iterations")
plt.ylabel("Loss")
plt.title("Loss Curve")
plt.show()

# 予測
with torch.no_grad():
    predicted = model(car_price_tensor).numpy()

plt.scatter(car_prices_array, number_of_car_sell_array, label="Original Data", color="red")
plt.plot(car_prices_array, predicted, label="Fitted Line", color="blue")
plt.xlabel("Car Price $")
plt.ylabel("Number of Car Sell")
plt.title("Original vs Predicted values")
plt.legend()
plt.show()

# 価格が10ドルのときの販売台数予測
with torch.no_grad():
    predicted_10 = model(torch.tensor([[10.0]], dtype=torch.float32)).item()
    print(f'Predicted number of car sells for $10 price: {predicted_10:.2f}')
