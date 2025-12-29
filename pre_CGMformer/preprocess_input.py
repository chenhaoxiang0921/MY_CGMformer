import pymongo
from datasets import Dataset
import pickle
import os
from tqdm import tqdm
import config  # 导入刚才创建的配置文件

# ================= 1. 配置路径 =================
# 项目根目录
PROJECT_ROOT = r"C:\Users\haoxiang.chen\PycharmProjects\CGMformer"

# 字典路径
DICT_PATH = os.path.join(PROJECT_ROOT, "cgm_ckp", "token2id.pkl")

# 输出数据集保存路径
OUTPUT_PATH = os.path.join(PROJECT_ROOT, "pre_CGMformer", "my_pretrain_dataset_mongodb")

# 序列长度配置
GLUCOSE_LEN = 480
MAX_SEQ_LEN = GLUCOSE_LEN + 1
# ===============================================

print(f"正在加载字典: {DICT_PATH}")
if not os.path.exists(DICT_PATH):
    raise FileNotFoundError(f"找不到字典文件: {DICT_PATH}")

with open(DICT_PATH, 'rb') as f:
    token2id = pickle.load(f)

unk_id = token2id.get('<unk>', token2id.get('<UNK>', 0))
cls_id = token2id.get('<cls>', token2id.get('<CLS>', 0))

print(f"字典加载完毕。CLS ID: {cls_id}, UNK ID: {unk_id}")


def process_glucose_value(value_str):
    """
    处理单个血糖值：转浮点 -> 单位换算 (mmol/L to mg/dL) -> 转整型字符串 -> 查字典
    """
    try:
        val_float = float(value_str)
        # 根据之前的经验，数据库中的值可能是 mmol/L (如 5.0)，模型需要 mg/dL (如 90)
        val_float = val_float * 18.0
        val_int_str = str(int(val_float))
        return token2id.get(val_int_str, unk_id)
    except:
        return unk_id


def main():
    # 1. 连接 MongoDB
    print(f"正在连接 MongoDB: {config.DB_NAME} ...")
    try:
        client = pymongo.MongoClient(config.MONGO_URI)
        db = client[config.DB_NAME]
        collection = db[config.COLLECTION_NAME]

        # 估算文档总数用于进度条
        total_docs = collection.count_documents({})
        print(f"连接成功！集合 '{config.COLLECTION_NAME}' 中共有 {total_docs} 条数据。")

    except Exception as e:
        print(f"数据库连接失败: {e}")
        return

    data_samples = []

    # 2. 从数据库流式读取数据
    # 使用 projection 只拉取需要的 'data' 字段，减少网络带宽消耗
    cursor = collection.find({}, {config.DATA_FIELD: 1})

    for doc in tqdm(cursor, total=total_docs, desc="Reading from MongoDB"):
        try:
            # 获取文档中的 data 列表
            # 根据 Result.json，格式为: "data": [{"value": "5.0", ...}, ...]
            raw_data_list = doc.get(config.DATA_FIELD, [])

            if not raw_data_list:
                continue

            # 提取 value 值
            # 注意：这里假设 list 里是 dict 对象。如果数据库里存的是字符串形式的 JSON，可能需要 json.loads
            # 根据 Result.json 分析，Mongo里通常存的是 Object Array，直接取即可
            glucose_values = [item['value'] for item in raw_data_list if isinstance(item, dict) and 'value' in item]

            if not glucose_values:
                continue

            # --- 长度处理 ---
            if len(glucose_values) > GLUCOSE_LEN:
                glucose_values = glucose_values[:GLUCOSE_LEN]

            # --- 数值映射转 ID ---
            token_ids = [process_glucose_value(v) for v in glucose_values]

            # --- 长度不足填充 UNK ---
            if len(token_ids) < GLUCOSE_LEN:
                token_ids = token_ids + [unk_id] * (GLUCOSE_LEN - len(token_ids))

            # --- 添加 [CLS] ---
            input_ids = [cls_id] + token_ids

            # --- Attention Mask ---
            attention_mask = [1] * len(input_ids)

            data_samples.append({
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": input_ids.copy()
            })

        except Exception as e:
            # 捕获单条数据处理异常，不中断整个流程
            # print(f"Skipping document due to error: {e}")
            continue

    # 关闭数据库连接
    client.close()

    # 3. 生成 Dataset 并保存
    if len(data_samples) == 0:
        print("未提取到任何有效数据，请检查数据库字段结构是否与代码预期一致。")
        return

    print(f"正在构建 HuggingFace Dataset，有效样本数: {len(data_samples)}...")
    dataset = Dataset.from_list(data_samples)

    print(f"正在保存数据集至: {OUTPUT_PATH}")
    dataset.save_to_disk(OUTPUT_PATH)

    print("全部完成")


if __name__ == "__main__":
    main()