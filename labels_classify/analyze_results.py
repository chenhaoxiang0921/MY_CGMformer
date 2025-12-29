import pickle
import numpy as np
import pandas as pd
import os
import torch
from sklearn.metrics import classification_report, confusion_matrix

# ================= 配置 =================
PROJECT_ROOT = r"C:\Users\haoxiang.chen\PycharmProjects\CGMformer"
RESULTS_DIR = os.path.join(PROJECT_ROOT, "labels_classify", "output_finetuned")
# 注意文件名改了
PICKLE_PATH = os.path.join(RESULTS_DIR, "predictions_with_id.pickle")


# =======================================

def main():
    if not os.path.exists(PICKLE_PATH):
        print(f"❌ 找不到文件: {PICKLE_PATH}")
        return

    print("📦 加载结果...")
    with open(PICKLE_PATH, "rb") as f:
        data = pickle.load(f)

    # 1. 解包数据
    # 注意：现在 data 是一个字典，不是原来的 PredictionOutput 对象了
    logits = data["predictions"]
    y_true = data["label_ids"]
    user_ids = data["user_ids"]  # <--- 获取 UserID 列表

    # 2. 计算概率
    probs = torch.nn.functional.softmax(torch.tensor(logits), dim=-1).numpy()
    y_pred = np.argmax(probs, axis=1)

    # 3. 生成 CSV
    print("📝 生成报表...")

    df = pd.DataFrame({
        "Sample_Index": range(len(y_pred)),
        "UserID": user_ids,  # <--- 这一列终于有了！
        "True_Label": y_true,
        "Predicted_Label": y_pred,
        "Prob_Healthy": probs[:, 0].round(4),
        "Prob_Diabetes": probs[:, 1].round(4),
        "Is_Correct": y_true == y_pred
    })

    # 4. 保存
    save_path = os.path.join(RESULTS_DIR, "final_prediction_report.csv")
    df.to_csv(save_path, index=False, encoding="utf-8-sig")

    print(f"✅ 报表已生成: {save_path}")
    print(df.head())  # 预览一下

    # 5. 评估报告 (不变)
    print("\n评估报告:")
    print(classification_report(y_true, y_pred, target_names=['Healthy', 'Diabetes'], zero_division=0))


if __name__ == "__main__":
    main()