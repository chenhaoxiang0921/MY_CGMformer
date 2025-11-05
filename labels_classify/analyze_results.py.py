import pickle
import numpy as np
import torch
import pandas as pd
import os
import glob

# ================= 配置区域 =================
# 1. 设置结果的根目录 (和你 run_labels_classify.py 里的 --output_path 保持一致)
OUTPUT_ROOT = "./my_results"

# 2. 定义标签映射 (根据你的任务修改，参考 run_labels_classify.py 中的 target_name_id_dict)
# 如果你是做二分类（比如 0=正常, 1=糖尿病），可以这样写：
ID2LABEL = {
    0: "Normal",
    1: "Diabetes (T2D)",
    2: "Impaired (IGR)"  # 如果是三分类的话
}


# ===========================================

def find_latest_prediction_file(root_dir):
    """自动寻找最近一次生成的 predictions.pickle 文件"""
    # 搜索所有子文件夹下的 predictions.pickle
    search_pattern = os.path.join(root_dir, "**", "predictions.pickle")
    files = glob.glob(search_pattern, recursive=True)

    if not files:
        raise FileNotFoundError(f"在 {root_dir} 下没找到任何 predictions.pickle 文件，请检查路径或确认模型是否运行成功。")

    # 按修改时间排序，找最新的一个
    latest_file = max(files, key=os.path.getmtime)
    print(f"✅ 自动定位到最新的结果文件: {latest_file}")
    return latest_file


def main():
    # 1. 获取文件路径
    try:
        pickle_path = find_latest_prediction_file(OUTPUT_ROOT)
    except Exception as e:
        print(e)
        return

    # 2. 加载 pickle 文件
    print("⏳ 正在加载预测结果...")
    with open(pickle_path, "rb") as f:
        preds = pickle.load(f)

    # 3. 提取数据
    # HuggingFace Trainer 的 predict 输出通常包含 predictions 和 label_ids
    logits = preds.predictions
    label_ids = preds.label_ids

    # 4. 计算概率和类别
    # 使用 Softmax 将分数转换为概率 (0-1之间)
    probs = torch.nn.functional.softmax(torch.tensor(logits), dim=-1).numpy()
    # 获取概率最大的那个类别的索引
    pred_classes = np.argmax(probs, axis=1)

    # 5. 整理结果到 DataFrame
    df_data = {
        "Predicted_Class_ID": pred_classes,
        "Max_Probability": np.max(probs, axis=1)  # 置信度
    }

    # 把每一类的概率都列出来，方便分析
    num_classes = probs.shape[1]
    for i in range(num_classes):
        class_name = ID2LABEL.get(i, f"Class_{i}")
        df_data[f"Prob_{class_name}"] = probs[:, i]

    # 如果输入数据里有真实标签，也放进去对比
    if label_ids is not None:
        df_data["True_Label_ID"] = label_ids
        # 判断预测是否正确
        df_data["Is_Correct"] = (pred_classes == label_ids)

    df = pd.DataFrame(df_data)

    # 6. 映射类别名称 (可选)
    if ID2LABEL:
        df["Predicted_Label"] = df["Predicted_Class_ID"].map(ID2LABEL)
        if "True_Label_ID" in df.columns:
            df["True_Label"] = df["True_Label_ID"].map(ID2LABEL)

    # 7. 打印摘要
    print("\n" + "=" * 30)
    print("📊 预测结果摘要")
    print("=" * 30)
    print(df.head())  # 打印前5行

    if "Is_Correct" in df.columns:
        acc = df["Is_Correct"].mean()
        print(f"\n📈 总体准确率: {acc:.2%}")

    # 8. 保存到文件
    save_path = pickle_path.replace(".pickle", "_analysis.csv")
    df.to_csv(save_path, index=False)
    print(f"\n💾 详细结果已保存至: {save_path}")
    print("你可以用 Excel 打开这个 CSV 文件查看每一条数据的预测详情。")


if __name__ == "__main__":
    main()