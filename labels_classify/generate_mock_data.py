import pandas as pd
import numpy as np
import random


def generate_glucose_profile(is_diabetic=False):
    """生成一条模拟的一天血糖数据 (288个点)"""
    # 基础血糖值
    base = 160 if is_diabetic else 100

    # 生成 288 个点 (24小时 * 12个点/小时)
    time_points = np.linspace(0, 24, 288)

    # 模拟三餐波动 (早餐8点, 午餐12点, 晚餐18点)
    meal_effect = np.zeros(288)
    for meal_time in [8, 12, 18]:
        # 制造一个类似于高斯分布的餐后血糖峰值
        peak_height = random.uniform(80, 150) if is_diabetic else random.uniform(30, 60)
        # 峰值持续时间 (sigma)
        sigma = 1.5 if is_diabetic else 0.8
        meal_effect += peak_height * np.exp(-0.5 * ((time_points - meal_time) / sigma) ** 2)

    # 加入随机噪声
    noise = np.random.normal(0, 5, 288)

    # 合成最终血糖曲线
    glucose = base + meal_effect + noise

    # 限制范围在 40-400 之间
    glucose = np.clip(glucose, 40, 400)
    return glucose.astype(int)


# 生成 5 条测试数据
data = []
for i in range(5):
    # 前3个人是 T2D (标签1)，后2个人是 Normal (标签0)
    label = 1 if i < 3 else 0
    glucose_seq = generate_glucose_profile(is_diabetic=(label == 1))

    row = {
        "id": f"Patient_{100 + i}",
        "label": label
    }
    # 添加血糖列 g_0 到 g_287
    for t, val in enumerate(glucose_seq):
        row[f"g_{t}"] = val

    data.append(row)

# 转换为 DataFrame
# 调整列顺序：id 在最前，label 在最后，中间是血糖
cols = ["id"] + [f"g_{i}" for i in range(288)] + ["label"]
df = pd.DataFrame(data, columns=cols)

# 保存为 CSV
df.to_csv("my_cgm_data.csv", index=False)
print("✅ 测试数据已生成: my_cgm_data.csv")
print(f"数据形状: {df.shape} (应为 5 行, 290 列)")
print("包含 3 个模拟糖尿病样本 (Label 1) 和 2 个正常样本 (Label 0)")