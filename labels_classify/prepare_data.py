import pandas as pd
from datasets import Dataset
import pickle
import os
import numpy as np

# 1. 加载字典
DICT_PATH = '../cgm_ckp/token2id.pkl'
if not os.path.exists(DICT_PATH):
    raise FileNotFoundError(f"找不到字典文件: {DICT_PATH}")

with open(DICT_PATH, 'rb') as f:
    token2id = pickle.load(f)


# --- 🔍 自动侦测特殊 Token 的 Key ---
# 有些字典用 <UNK>, 有些用 <unk>, 有些用 [UNK]
def find_key(token_dict, candidates):
    for key in candidates:
        if key in token_dict:
            return key
    return None


UNK_KEY = find_key(token2id, ['<unk>', '<UNK>', '[UNK]'])
CLS_KEY = find_key(token2id, ['<cls>', '<CLS>', '[CLS]'])
PAD_KEY = find_key(token2id, ['<pad>', '<PAD>', '[PAD]'])

# 获取对应的 ID，如果找不到特殊的 key，就默认用 0 或 1
unk_id = token2id[UNK_KEY] if UNK_KEY else 0
cls_id = token2id[CLS_KEY] if CLS_KEY else 0
pad_id = token2id[PAD_KEY] if PAD_KEY else 0

print(f"📖 字典检查完毕:")
print(f"   UNK token: '{UNK_KEY}' -> ID: {unk_id}")
print(f"   CLS token: '{CLS_KEY}' -> ID: {cls_id}")
print(f"   PAD token: '{PAD_KEY}' -> ID: {pad_id}")


# ------------------------------------

def process_glucose(value):
    """将血糖值转换为 Token ID，增加鲁棒性防止 NaN"""
    try:
        # 处理可能的 NaN 或非数字
        if pd.isna(value):
            return unk_id

        val_float = float(value)
        # 限制范围 40-300
        if val_float < 40: val_float = 40
        if val_float > 300: val_float = 300

        # 尝试转为字符串 Key
        # 有些字典的 key 是 '100' (str), 有些可能是 100 (int)
        val_int = int(val_float)
        val_str = str(val_int)

        # 优先找字符串 key
        if val_str in token2id:
            return token2id[val_str]
        # 其次找数字 key
        elif val_int in token2id:
            return token2id[val_int]
        else:
            return unk_id

    except Exception as e:
        # 遇到任何解析错误，统统返回 UNK，绝不返回 None
        return unk_id


# 2. 读取原始数据
CSV_PATH = "my_cgm_data.csv"
if not os.path.exists(CSV_PATH):
    raise FileNotFoundError(f"找不到数据文件: {CSV_PATH}，请先运行 generate_mock_data.py")

df = pd.read_csv(CSV_PATH)
print(f"📊 正在处理 {len(df)} 条数据...")

data_list = []
for index, row in df.iterrows():
    # 提取血糖列 (假设 id 在第0列, label 在最后一列, 中间是血糖)
    # 根据 generate_mock_data.py: cols = ["id"] + [g_0...g_287] + ["label"]
    # 所以血糖是从 第1列 到 倒数第2列
    glucose_values = row.iloc[1:-1].values

    # 转化为 Token ID
    input_ids = [process_glucose(v) for v in glucose_values]

    # 头部添加 CLS
    input_ids = [cls_id] + input_ids

    # 确保没有 None/NaN 混进去
    # double check: 如果有任何非整数，强行转为 0
    input_ids = [int(x) if x is not None and not pd.isna(x) else unk_id for x in input_ids]

    # 构造样本
    data_list.append({
        "input_ids": input_ids,
        "label": int(row['label'])
    })

# 3. 创建并保存 Dataset
dataset = Dataset.from_list(data_list)
# 保存到文件夹
OUTPUT_PATH = "../labels_classify/my_processed_input"
dataset.save_to_disk(OUTPUT_PATH)
print(f"✅ 数据修复完成！已保存至 {OUTPUT_PATH}")
print(f"   样本长度示例: {len(data_list[0]['input_ids'])} (应为 289)")