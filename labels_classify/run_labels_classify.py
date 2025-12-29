import os
# 如果没有 GPU，这几行可以注释掉或者留着也没关系，Trainer 会自动处理
# GPU_NUMBER = [1]
# os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([str(s) for s in GPU_NUMBER])
# os.environ["NCCL_DEBUG"] = "INFO"

import pickle
import numpy as np
import torch
import sys
from datasets import load_from_disk
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import (
    BertForSequenceClassification,
    Trainer,
    TrainingArguments
)

# ================= 配置区域 =================
PROJECT_ROOT = r"C:\Users\haoxiang.chen\PycharmProjects\CGMformer"

# 1. 预训练模型路径 (指向包含 config.json 和 model.safetensors 的文件夹)
PRETRAINED_MODEL_PATH = r"C:\Users\haoxiang.chen\PycharmProjects\CGMformer\pre_CGMformer\pre_CGMformer\models\251225_152136_mask_480_bs48_TFIDF4560_L4_H8_emb128_SL512_E1_B2_LR0.0004_LSlinear_WU2000_Oadamw_DS2\models"

# 2. 数据集路径 (由 prepare_multilabel_data.py 生成)
# 确保这里指向的是包含 "train" 和 "test" 子文件夹的目录
DATA_PATH = os.path.join(PROJECT_ROOT, "labels_classify", "finetune_db_dataset")
TRAIN_DATA = os.path.join(DATA_PATH, "train")
TEST_DATA = os.path.join(DATA_PATH, "test")

# 3. 输出路径
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "labels_classify", "output_finetuned")

# 模型参数
MAX_SEQ_LEN = 512
NUM_LABELS = 2  # 二分类 (0: 健康, 1: 糖尿病)


# ===========================================

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)

    # 计算二分类指标
    # zero_division=0 防止除以零报错
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary', zero_division=0)
    acc = accuracy_score(labels, preds)

    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }


def main():
    # 0. 检查路径
    if not os.path.exists(TRAIN_DATA):
        print(f"❌ 错误：找不到训练集文件夹: {TRAIN_DATA}")
        print("请先运行 prepare_multilabel_data.py 生成数据集。")
        return

    print("📂 加载数据集...")
    train_dataset = load_from_disk(TRAIN_DATA)
    eval_dataset = load_from_disk(TEST_DATA)

    print(f"   训练集大小: {len(train_dataset)}")
    print(f"   测试集大小: {len(eval_dataset)}")

    print("⚙️ 加载预训练模型...")
    # 加载模型并添加分类头
    # ignore_mismatched_sizes=True 是必须的，因为我们要丢弃原来的 MLM 头
    model = BertForSequenceClassification.from_pretrained(
        PRETRAINED_MODEL_PATH,
        num_labels=NUM_LABELS,
        max_position_embeddings=MAX_SEQ_LEN,
        ignore_mismatched_sizes=True
    )

    # 训练参数
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=5,  # 微调轮数
        per_device_train_batch_size=4,  # CPU 只能设小
        per_device_eval_batch_size=4,
        warmup_steps=10,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=5,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,  # 最多保留2个模型checkpoint
        dataloader_num_workers=0,  # Windows 下必须为 0
        learning_rate=2e-5,  # 微调学习率通常比预训练低一个数量级
        use_cpu=True,  # 强制 CPU
        remove_unused_columns=False  # 关键！防止 Trainer 自动删掉 userid 列
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics
    )

    # 1. 微调训练
    print("🚀 开始微调 (Fine-tuning)...")
    trainer.train()

    print("💾 保存微调后的模型...")
    trainer.save_model(OUTPUT_DIR)

    # 2. 预测评估
    print("🧪 进行预测...")
    # predictions 对象包含: predictions(Logits), label_ids, metrics
    predictions = trainer.predict(eval_dataset)

    # 3. 保存完整结果 (含 UserID)
    print("📦 打包保存预测结果...")

    # 尝试提取 userid
    # 因为我们在 TrainingArguments 里设置了 remove_unused_columns=False
    # 所以 dataset 里的 userid 应该还在
    if "userid" in eval_dataset.column_names:
        user_ids = eval_dataset["userid"]
    else:
        print("⚠️ 警告: 测试集中找不到 'userid' 列，保存的结果将不包含用户ID。")
        user_ids = ["Unknown"] * len(eval_dataset)

    # 构建保存字典
    save_data = {
        "predictions": predictions.predictions,  # 原始 Logits (包含概率信息)
        "label_ids": predictions.label_ids,  # 真实标签
        "metrics": predictions.metrics,  # 整体指标
        "user_ids": user_ids  # 对应用户的 ID
    }

    # 保存为 pickle (这是关键修改，替代原来的 .npy)
    pickle_output_path = os.path.join(OUTPUT_DIR, "predictions_with_id.pickle")

    with open(pickle_output_path, "wb") as fp:
        pickle.dump(save_data, fp)

    print(f"✅ 完整结果已保存至: {pickle_output_path}")
    print("接下来请运行 analyze_results.py 查看详细报表。")


if __name__ == "__main__":
    main()