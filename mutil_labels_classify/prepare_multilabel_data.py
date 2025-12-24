import pandas as pd
from datasets import Dataset
import pickle
import os
import numpy as np
import random

# 1. 加载字典
DICT_PATH = '../cgm_ckp/token2id.pkl'
with open(DICT_PATH, 'rb') as f:
    token2id = pickle.load(f)


# 简单的字典查找逻辑 (复用之前的优化版)
def find_key(token_dict, candidates):
    for key in candidates:
        if key in token_dict: return key
    return None


UNK_KEY = find_key(token2id, ['<unk>', '<UNK>', '[UNK]']) or '<UNK>'
unk_id = token2id.get(UNK_KEY, 0)
cls_id = token2id.get('<cls>', token2id.get('<CLS>', 0))


def process_glucose(value):
    try:
        val_str = str(int(float(value)))
        return token2id.get(val_str, unk_id)
    except:
        return unk_id


# 2. 生成模拟的多标签原始数据
# 我们直接在这里生成 Dataset，跳过 CSV 步骤
print("🧪 正在生成模拟的多标签数据...")
data_list = []
for i in range(5):
    # 随机生成 288 个血糖值
    input_ids = [cls_id] + [process_glucose(random.randint(80, 180)) for _ in range(288)]

    # 随机生成 3 个标签 (0或1)
    # 注意：代码里有一个列名带有空格 'macrovascular '，必须完全匹配
    row = {
        "input_ids": input_ids,
        "macrovascular ": random.randint(0, 1),
        "microvascular": random.randint(0, 1),
        "complication": random.randint(0, 1)
    }
    data_list.append(row)

# 3. 保存 Dataset
dataset = Dataset.from_list(data_list)
OUTPUT_PATH = "./data/my_multilabel_input"
dataset.save_to_disk(OUTPUT_PATH)
print(f"✅ 多标签数据已保存至: {OUTPUT_PATH}")
print("包含列名:", dataset.column_names)