import pickle
import numpy as np
import pandas as pd
import torch
import glob
import os

# ================= 配置绝对路径 =================
# 指向 run_mutil_labels_classify.py 输出结果的文件夹
OUTPUT_ROOT = r"C:\Users\haoxiang.chen\PycharmProjects\CGMformer\mutil_labels_classify\my_multilabel_results"
# ===============================================

LABEL_NAMES = ['Macrovascular', 'Microvascular', 'Complication']


def find_latest_prediction_file(root_dir):
    # 递归搜索 predictions.pickle
    search_pattern = os.path.join(root_dir, "**", "predictions.pickle")
    files = glob.glob(search_pattern, recursive=True)
    if not files:
        raise FileNotFoundError(f"在 {root_dir} 下没找到 predictions.pickle，请先运行 run_mutil_labels_classify.py")
    # 按修改时间排序，取最新的
    return max(files, key=os.path.getmtime)


def main():
    try:
        pickle_path = find_latest_prediction_file(OUTPUT_ROOT)
    except FileNotFoundError as e:
        print(e)
        return

    print(f"📖 读取文件: {pickle_path}")

    with open(pickle_path, "rb") as f:
        preds = pickle.load(f)

    logits = preds.predictions
    # 多标签任务使用 Sigmoid 将 Logits 转换为概率 (0~1)
    probs = 1 / (1 + np.exp(-logits))

    # 阈值通常设为 0.5 (概率大于50%认为有该标签)
    threshold = 0.5
    predictions = (probs > threshold).astype(int)

    # 获取真实标签（如果有）
    label_ids = getattr(preds, "label_ids", None)

    print("\n" + "=" * 30)
    print("📊 多标签预测结果摘要 (前 5 条)")
    print("=" * 30)

    # 准备保存到 CSV 的数据列表
    csv_data = []

    for i, (prob_row, pred_row) in enumerate(zip(probs, predictions)):
        # 1. 打印前5条到控制台看看
        if i < 5:
            print(f"\n样本 {i}:")

        # 2. 收集这一行的数据
        row_dict = {"Sample_ID": i}

        for idx, label_name in enumerate(LABEL_NAMES):
            p = prob_row[idx]
            is_positive = pred_row[idx]

            # 如果有真实标签，进行对比
            truth_info = ""
            true_val = None
            if label_ids is not None:
                true_val = label_ids[i][idx]
                match = "正确" if true_val == is_positive else "错误"
                truth_info = f" | 真实: {true_val} ({match})"

            # 仅打印前5条
            if i < 5:
                status = "✅ YES" if is_positive else "❌ NO"
                print(f"  - {label_name}: {p:.2%} -> {status}{truth_info}")

            # 将详细数据写入字典，用于生成 CSV
            row_dict[f"{label_name}_Prob"] = p
            row_dict[f"{label_name}_Pred"] = int(is_positive)
            if true_val is not None:
                row_dict[f"{label_name}_True"] = int(true_val)

        csv_data.append(row_dict)

    df = pd.DataFrame(csv_data)

    # 保存 csv (文件名基于 pickle 路径自动生成)
    csv_path = pickle_path.replace(".pickle", "_analysis.csv")
    df.to_csv(csv_path, index=False)

    print("\n" + "=" * 30)
    print(f"💾 结果已真正保存至 CSV: {csv_path}")
    print("=" * 30)


if __name__ == "__main__":
    main()