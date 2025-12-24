import pickle
import numpy as np
import pandas as pd
import torch
import glob
import os

OUTPUT_ROOT = "./my_multilabel_results"
LABEL_NAMES = ['Macrovascular', 'Microvascular', 'Complication']


def find_latest_prediction_file(root_dir):
    search_pattern = os.path.join(root_dir, "**", "predictions.pickle")
    files = glob.glob(search_pattern, recursive=True)
    if not files: raise FileNotFoundError("没找到结果文件")
    return max(files, key=os.path.getmtime)


def main():
    pickle_path = find_latest_prediction_file(OUTPUT_ROOT)
    print(f"📖 读取文件: {pickle_path}")

    with open(pickle_path, "rb") as f:
        preds = pickle.load(f)

    logits = preds.predictions
    # 多标签任务使用 Sigmoid 而不是 Softmax
    probs = 1 / (1 + np.exp(-logits))

    # 阈值通常设为 0.5 (大于0.5就算有病)
    predictions = (probs > 0.5).astype(int)

    print("\n📊 多标签预测结果:")
    for i, (prob_row, pred_row) in enumerate(zip(probs, predictions)):
        print(f"\n样本 {i}:")
        for label_name, p, is_positive in zip(LABEL_NAMES, prob_row, pred_row):
            status = "✅ YES" if is_positive else "❌ NO"
            print(f"  - {label_name}: {p:.2%} -> {status}")


if __name__ == "__main__":
    main()