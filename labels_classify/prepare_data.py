import pandas as pd
import pymysql
import pymongo
from datasets import Dataset
import pickle
import os
import config_db  # 导入配置文件
from tqdm import tqdm

# ================= 1. 项目路径配置 =================
PROJECT_ROOT = r"C:\Users\haoxiang.chen\PycharmProjects\CGMformer"

# 输入：包含 userId 的 CSV 文件路径
INPUT_USER_IDS_CSV = os.path.join(PROJECT_ROOT, "labels_classify", "user_ids.csv")

# 字典路径
DICT_PATH = os.path.join(PROJECT_ROOT, "cgm_ckp", "token2id.pkl")

# 输出：run_labels_classify.py 需要的 Dataset 路径
OUTPUT_PATH = os.path.join(PROJECT_ROOT, "labels_classify", "finetune_db_dataset")

# 序列参数
MAX_SEQ_LEN = 512  # 输入长度 (包含 CLS)
GLUCOSE_LEN_LIMIT = 480  # 每天最大点数

# ================= 2. 加载字典 =================
print(f"📖 加载字典: {DICT_PATH}")
if not os.path.exists(DICT_PATH):
    raise FileNotFoundError(f"找不到字典文件: {DICT_PATH}")

with open(DICT_PATH, 'rb') as f:
    token2id = pickle.load(f)

UNK_ID = token2id.get('<unk>', token2id.get('<UNK>', 0))
CLS_ID = token2id.get('<cls>', token2id.get('<CLS>', 101))
PAD_ID = 0


# ================= 3. 辅助函数 =================

def get_mysql_conn():
    return pymysql.connect(**config_db.MYSQL_CONFIG)


def get_mongo_collection():
    client = pymongo.MongoClient(config_db.MONGO_URI)
    db = client[config_db.MONGO_DB_NAME]
    return db[config_db.MONGO_COLLECTION]


def determine_label(diabete_type):
    try:
        dtype = int(diabete_type)
        if dtype in [6, 7, 8]:
            return 0
        elif dtype in [1, 2, 3, 4, 5]:
            return 1
        else:
            return None
    except:
        return None


def process_glucose_seq(glucose_values):
    token_ids = []
    for val in glucose_values:
        try:
            val_float = float(val) * 18.0
            val_str = str(int(val_float))
            token_ids.append(token2id.get(val_str, UNK_ID))
        except:
            token_ids.append(UNK_ID)

    if len(token_ids) > GLUCOSE_LEN_LIMIT:
        token_ids = token_ids[:GLUCOSE_LEN_LIMIT]

    input_ids = [CLS_ID] + token_ids
    curr_len = len(input_ids)
    if curr_len < MAX_SEQ_LEN:
        pad_len = MAX_SEQ_LEN - curr_len
        input_ids = input_ids + [PAD_ID] * pad_len
        attention_mask = [1] * curr_len + [0] * pad_len
    else:
        input_ids = input_ids[:MAX_SEQ_LEN]
        attention_mask = [1] * MAX_SEQ_LEN

    return input_ids, attention_mask


# ================= 4. 主流程 =================

def main():
    # --- A. 读取输入的 User ID ---
    if not os.path.exists(INPUT_USER_IDS_CSV):
        print(f"❌ 找不到 User ID 输入文件: {INPUT_USER_IDS_CSV}")
        return

    print("📋 读取目标 UserId 列表...")
    df_ids = pd.read_csv(INPUT_USER_IDS_CSV, header=None, dtype=str)
    target_user_ids = df_ids.iloc[:, 0].tolist()
    target_user_ids = list(set(target_user_ids))
    print(f"🎯 待处理用户数: {len(target_user_ids)}")

    # --- B. 查询 MySQL 获取标签 ---
    print("Connecting to MySQL...")
    user_label_map = {}

    try:
        conn = get_mysql_conn()
        cursor = conn.cursor()
        format_strings = ','.join(['%s'] * len(target_user_ids))
        sql = f"SELECT id, diabete_type FROM t_archives_user WHERE id IN ({format_strings})"

        print(f"🔍 正在 MySQL 中查询 {len(target_user_ids)} 个用户的疾病类型...")
        cursor.execute(sql, tuple(target_user_ids))
        results = cursor.fetchall()

        for row in results:
            uid = str(row[0])
            dtype = row[1]
            label = determine_label(dtype)
            if label is not None:
                user_label_map[uid] = label

        cursor.close()
        conn.close()
        print(f"✅ MySQL 查询完成，获取到有效标签用户数: {len(user_label_map)}")

        # 打印一下哪些用户找到了标签
        print(f"   已获取标签的用户ID: {list(user_label_map.keys())}")

    except Exception as e:
        print(f"❌ MySQL 查询失败: {e}")
        return

    # --- C. 查询 MongoDB 获取数据 ---
    print("Connecting to MongoDB...")
    col = get_mongo_collection()
    final_dataset_list = []
    valid_ids = list(user_label_map.keys())

    print(f"🔍 开始遍历 MongoDB...")

    for uid in tqdm(valid_ids, desc="Processing Users"):
        label = user_label_map[uid]

        # ⚠️⚠️⚠️ 关键修改：同时查找 String 类型和 Int 类型的 ID ⚠️⚠️⚠️
        search_criteria = [uid]
        if uid.isdigit():
            search_criteria.append(int(uid))

        # 使用 $in 操作符，只要匹配其中任何一个格式就算找到
        # 投影只取 data 字段
        cursor = col.find({"userId": {"$in": search_criteria}}, {"data": 1})

        # 将 cursor 转为 list 方便统计数量
        user_docs = list(cursor)

        if len(user_docs) == 0:
            print(f"⚠️ 警告: 用户 {uid} 在 MongoDB 中未找到任何记录！(尝试了 String 和 Int 格式)")
            continue

        valid_doc_count = 0
        skipped_short_count = 0

        for doc in user_docs:
            raw_data_list = doc.get("data", [])
            if not raw_data_list:
                continue

            glucose_vals = [item.get('value') for item in raw_data_list if 'value' in item]

            # 过滤过短数据
            if len(glucose_vals) < 100:
                skipped_short_count += 1
                continue

            input_ids, attention_mask = process_glucose_seq(glucose_vals)

            final_dataset_list.append({
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": label,
                "userid": uid
            })
            valid_doc_count += 1

        # 打印每个用户的处理详情
        tqdm.write(
            f"用户 {uid}: 找到 {len(user_docs)} 条记录 -> 有效 {valid_doc_count} 条 (跳过 {skipped_short_count} 条过短)")

    # --- D. 保存数据集 ---
    if not final_dataset_list:
        print("❌ 未生成任何有效数据，请检查数据库。")
        return

    print(f"📦 正在构建 HuggingFace Dataset，共 {len(final_dataset_list)} 条样本...")
    dataset = Dataset.from_list(final_dataset_list)
    dataset_split = dataset.train_test_split(test_size=0.1)

    print(f"💾 保存至: {OUTPUT_PATH}")
    dataset_split.save_to_disk(OUTPUT_PATH)

    print(f"✅ 全部完成！")
    print(f"   Train集: {len(dataset_split['train'])}")
    print(f"   Test集:  {len(dataset_split['test'])}")


if __name__ == "__main__":
    main()