import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import warnings
warnings.filterwarnings("ignore")


import numpy as np
import torch

# 部屋の広さ（平方メートル）のデータ
# 例: 20平米、25平米、30平米、...
room_sizes_np = np.array([20.0, 25.0, 30.0, 35.0, 40.0, 45.0, 50.0, 55.0, 60.0, 65.0], dtype=np.float32)

# 対応する家賃（万円）のデータ
# 広さに比例して家賃も上がるが、多少のばらつきがある
rents_np = np.array([6.5, 7.8, 9.0, 10.5, 11.8, 13.0, 14.5, 15.8, 17.0, 18.5], dtype=np.float32)

# PyTorchのテンソルに変換 (これまで通り reshape(-1, 1) で2次元にするのを忘れずに)
# input_data: 部屋の広さ (特徴量)
room_sizes_tensor = torch.from_numpy(room_sizes_np).reshape(-1, 1)

# target_data: 家賃 (正解ラベル)
rents_tensor = torch.from_numpy(rents_np).reshape(-1, 1)

print("部屋の広さデータ (テンソル):\n", room_sizes_tensor)
print("\n家賃データ (テンソル):\n", rents_tensor)