import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import warnings
warnings.filterwarnings("ignore")


# 部屋の広さ（平方メートル）のデータ
# 例: 20平米、25平米、30平米、...
room_sizes_array = [20.0, 25.0, 30.0, 35.0, 40.0, 45.0, 50.0, 55.0, 60.0, 65.0]
room_sizes_np = np.array(room_sizes_array, dtype=np.float32)

# 対応する家賃（万円）のデータ
# 広さに比例して家賃も上がるが、多少のばらつきがある
rents_array = [6.5, 7.8, 9.0, 10.5, 11.8, 13.0, 14.5, 15.8, 17.0, 18.5]
rents_np = np.array(rents_array, dtype=np.float32)

# PyTorchのテンソルに変換 (これまで通り reshape(-1, 1) で2次元にするのを忘れずに)
# input_data: 部屋の広さ (特徴量)
room_sizes_tensor = torch.from_numpy(room_sizes_np).reshape(-1, 1)

# target_data: 家賃 (正解ラベル)
rents_tensor = torch.from_numpy(rents_np).reshape(-1, 1)

# print("部屋の広さデータ (テンソル):\n", room_sizes_tensor)
# print("\n家賃データ (テンソル):\n", rents_tensor)



class LinearRegression(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, x):
        return self.linear(x)
    
input_dim = 1
output_dim = 1
model = LinearRegression(input_dim, output_dim)

mse = nn.MSELoss()

learning_rate = 0.00001
optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate)

loss_list = []
iteration_number = 1001
for iteration in range(iteration_number):
    
    optimizer.zero_grad()
    results = model(room_sizes_tensor)
    loss = mse(results, rents_tensor)
    loss.backward()
    optimizer.step()
    loss_list.append(loss.item())

    if(iteration % 50 == 0):
        print("epoch {}, loss {}".format(iteration, loss.item()))

plt.plot(range(iteration_number), loss_list)
plt.xlabel("Number of Iteration")
plt.ylabel("Loss")
plt.show()

# predict our car price 
predicted = model(room_sizes_tensor).data.numpy()
plt.scatter(room_sizes_array,rents_array,label = "original data",color ="red")
plt.scatter(room_sizes_array,predicted,label = "predicted data",color ="blue")

# predict if car price is 10$, what will be the number of car sell
#predicted_10 = model(torch.from_numpy(np.array([10]))).data.numpy()
#plt.scatter(10,predicted_10.data,label = "car price 10$",color ="green")
plt.legend()
plt.xlabel("Car Price $")
plt.ylabel("Number of Car Sell")
plt.title("Original vs Predicted values")
plt.show()