import pandas as pd
from datasets import Dataset
import pickle
import os
import numpy as np
import random

# ================= 配置绝对路径 =================
# 项目根目录
PROJECT_ROOT = r"C:\Users\haoxiang.chen\PycharmProjects\CGMformer"

# 1. 字典路径
DICT_PATH = os.path.join(PROJECT_ROOT, "cgm_ckp", "token2id.pkl")

# 2. 输出数据集保存路径
# 保存在 mutil_labels_classify 文件夹下
OUTPUT_PATH = os.path.join(PROJECT_ROOT, "mutil_labels_classify", "my_multilabel_input")
# ===============================================

print(f"📖 正在加载字典: {DICT_PATH}")
if not os.path.exists(DICT_PATH):
    raise FileNotFoundError(f"找不到字典文件: {DICT_PATH}")

with open(DICT_PATH, 'rb') as f:
    token2id = pickle.load(f)

# 简单的字典查找逻辑
def find_key(token_dict, candidates):
    for key in candidates:
        if key in token_dict: return key
    return None

UNK_KEY = find_key(token2id, ['<unk>', '<UNK>', '[UNK]']) or '<UNK>'
unk_id = token2id.get(UNK_KEY, 0)
# 尝试查找 CLS，找不到就用 0
cls_id = token2id.get('<cls>', token2id.get('<CLS>', 0))

def process_glucose(value):
    try:
        val_str = str(int(float(value)))
        return token2id.get(val_str, unk_id)
    except:
        return unk_id

# 2. 生成模拟的多标签原始数据
print("🧪 正在生成模拟的多标签数据...")
data_list = []

# 生成 20 条数据方便观察
for i in range(20):
    # 随机生成 288 个血糖值
    input_ids = [cls_id] + [process_glucose(random.randint(80, 180)) for _ in range(288)]

    # 随机生成 3 个标签 (0或1)
    # 注意：保持 key 中的空格，与训练代码一致
    row = {
        "input_ids": input_ids,
        "macrovascular ": random.randint(0, 1), # 注意这里的空格
        "microvascular": random.randint(0, 1),
        "complication": random.randint(0, 1)
    }
    data_list.append(row)

# 3. 保存 Dataset
dataset = Dataset.from_list(data_list)

# 确保父目录存在
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

dataset.save_to_disk(OUTPUT_PATH)
print(f"✅ 多标签数据已保存至: {OUTPUT_PATH}")
print("包含列名:", dataset.column_names)
print(f"数据量: {len(dataset)}")