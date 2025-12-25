import pandas as pd
import numpy as np
import random
import os


def generate_glucose_profile(is_diabetic=False):
    """生成一条模拟的一天血糖数据 (288个点，5分钟间隔)"""
    # 基础血糖值
    base = 160 if is_diabetic else 100

    # 恢复为 288 个点
    num_points = 288

    time_points = np.linspace(0, 24, num_points)

    # 模拟三餐波动 (早餐8点, 午餐12点, 晚餐18点)
    meal_effect = np.zeros(num_points)
    for meal_time in [8, 12, 18]:
        # 制造一个类似于高斯分布的餐后血糖峰值
        peak_height = random.uniform(80, 150) if is_diabetic else random.uniform(30, 60)
        # 峰值持续时间 (sigma)
        sigma = 1.5 if is_diabetic else 0.8
        meal_effect += peak_height * np.exp(-0.5 * ((time_points - meal_time) / sigma) ** 2)

    # 加入随机噪声
    noise = np.random.normal(0, 5, num_points)

    # 合成最终血糖曲线
    glucose = base + meal_effect + noise

    # 限制范围在 40-400 之间
    glucose = np.clip(glucose, 40, 400)
    return glucose.astype(int)


# ================= 配置生成数量 =================
NUM_SAMPLES = 20  # 生成 20 条数据
NUM_POINTS = 288  # 保持 288 个点
# ===============================================

data = []
for i in range(NUM_SAMPLES):
    # 简单的标签分配：前一半是糖尿病(1)，后一半是正常(0)
    # 也就是 0-9 是 T2D, 10-19 是 Normal
    label = 1 if i < (NUM_SAMPLES / 2) else 0

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
cols = ["id"] + [f"g_{i}" for i in range(NUM_POINTS)] + ["label"]
df = pd.DataFrame(data, columns=cols)

# ================= 保存路径 =================
save_dir = r"C:\Users\haoxiang.chen\PycharmProjects\CGMformer\labels_classify"

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

save_path = os.path.join(save_dir, "my_cgm_data20.csv")
df.to_csv(save_path, index=False)
# ===========================================

print(f"✅ 测试数据已生成: {save_path}")
print(f"数据形状: {df.shape} (应为 {NUM_SAMPLES} 行, {NUM_POINTS + 2} 列)")
print(f"包含 {NUM_SAMPLES // 2} 个糖尿病样本 (Label 1) 和 {NUM_SAMPLES - (NUM_SAMPLES // 2)} 个正常样本 (Label 0)")