import os
import torch
from transformers import BertForMaskedLM

# ================= 1. 配置模型路径 =================
# 请将下面的路径修改为您存放 model.safetensors 的文件夹路径
# 注意：路径最后不要带文件名，只要文件夹路径即可
MODEL_DIR = r"C:\Users\haoxiang.chen\PycharmProjects\CGMformer\pre_CGMformer\pre_CGMformer\models\251225_152136_mask_480_bs48_TFIDF4560_L4_H8_emb128_SL512_E1_B2_LR0.0004_LSlinear_WU2000_Oadamw_DS2\models"

# ================= 2. 检查文件 =================
# 自动检测是否存在 safetensors 或 bin 格式的权重
has_safetensors = os.path.exists(os.path.join(MODEL_DIR, "model.safetensors"))
has_bin = os.path.exists(os.path.join(MODEL_DIR, "pytorch_model.bin"))

if not (has_safetensors or has_bin):
    print(f"❌ 错误：在 {MODEL_DIR} 下找不到模型权重文件 (model.safetensors 或 pytorch_model.bin)！")
    print("请检查路径是否正确，确保文件夹里有 config.json 和 .safetensors/.bin 文件。")
    exit()

print(f"✅ 找到权重文件，准备加载...")

# ================= 3. 加载模型 =================
try:
    # from_pretrained 会自动处理 config.json 和权重文件的加载
    model = BertForMaskedLM.from_pretrained(MODEL_DIR)

    # 切换到评估模式 (关闭 Dropout 等)
    model.eval()

    print("\n🎉 恭喜！模型加载成功！")
    print("-" * 30)
    print(f"模型类型: {type(model).__name__}")
    print(f"词表大小: {model.config.vocab_size}")
    print(f"隐藏层维度: {model.config.hidden_size}")
    print(f"层数: {model.config.num_hidden_layers}")
    print("-" * 30)

    # ================= 4. 简单推理测试 =================
    print("\n🧪 正在进行推理测试...")

    # 模拟一个 batch_size=1, 长度=481 的输入 (1个CLS + 480个数据点)
    # 这里随机生成 0~100 的整数模拟 Token ID
    dummy_input = torch.randint(0, model.config.vocab_size, (1, 481))

    # 放入模型进行计算
    with torch.no_grad():
        outputs = model(dummy_input)

    # 检查输出
    # outputs.logits 的形状应该是 [Batch_Size, Sequence_Length, Vocab_Size]
    print(f"✅ 推理成功！")
    print(f"输入形状: {dummy_input.shape}")
    print(f"输出形状 (Logits): {outputs.logits.shape}")
    print("模型可以正常工作！")

except Exception as e:
    print(f"\n❌ 加载或推理失败: {e}")