import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from pathlib import Path

# 1. 确定模型路径
MODEL_PATH = Path("./models/Qwen2.5-1.5B-Instruct")

# 2. 加载分词器和模型
print("正在加载 Tokenizer 与模型...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,  # 使用 16 位浮点减少显存开销
    device_map="cuda"            # 自动将权重放置在 5060 Ti 显存中
)

# 3. 准备测试文本并分词
prompt = "人工智能是什么？"
# return_tensors="pt" 表示直接返回 PyTorch 张量格式（默认是 Python list）
encoded = tokenizer(prompt, return_tensors="pt")

# 将输入数据移动到 GPU 上
encoded = encoded.to("cuda")

# 观察输入张量的维度：[batch_size, sequence_length]
input_ids = encoded["input_ids"]
batch_size, prompt_len = input_ids.shape
print(f"\n【输入分析】")
print(f"输入张量 Shape: {input_ids.shape}")
print(f"Batch 大小: {batch_size}, 输入 Token 数量: {prompt_len}")
print(f"输入 Token IDs: {input_ids[0].tolist()}")

# 逐个观察：每个 ID 到底长什么样？
print("Token ID 对应碎片解析:")
for token_id in input_ids[0]:
    # convert_ids_to_tokens: 看它在分词器词表里的原始 key
    token_str = tokenizer.convert_ids_to_tokens(token_id.item())
    # decode: 看它被渲染成人类字符后的样子
    decoded_char = tokenizer.decode([token_id.item()])
    print(f"  ID: {token_id.item():<6} -> 词表标记: {token_str:<10} -> 实际字符: '{decoded_char}'")

# 4. 调用模型生成
print(f"\n【开始生成】")
outputs = model.generate(
    **encoded,              # 解包字典：传入 input_ids 和 attention_mask
    max_new_tokens=32,      # 纯增量生成 32 个新 token
    do_sample=False,        # 关闭随机采样，使用 Greedy 贪心解码（每次选 logit 最大的 token）
)

total_len = outputs.shape[1]
gen_len = total_len - prompt_len

print(f"输出张量 Shape: {outputs.shape}")
print(f"总 Token 数: {total_len}, 新生成 Token 数: {gen_len}")

# 5. 提取新生成的 Token 并解码
# outputs[0] 包含了 [原 Prompt Tokens + 新生成 Tokens]
generated_ids = outputs[0, prompt_len:]

print(f"\n【生成文本对比】")
print("完整输出（包含 Prompt）:")
print(tokenizer.decode(outputs[0], skip_special_tokens=True))

print("\n纯增量生成内容:")
print(tokenizer.decode(generated_ids, skip_special_tokens=True))