## 1. 一条请求从 generate() 到生成结束，中间经历的过程

example.py

```python
def main():
    path = os.path.expanduser("~/huggingface/Qwen3-0.6B/")
    tokenizer = AutoTokenizer.from_pretrained(path)
    llm = LLM(path, enforce_eager=True, tensor_parallel_size=1)

    sampling_params = SamplingParams(temperature=0.6, max_tokens=256)
    prompts = [
        "introduce yourself",
        "list all prime numbers within 100",
    ]
    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for prompt in prompts
    ]
    outputs = llm.generate(prompts, sampling_params)

    for prompt, output in zip(prompts, outputs):
        print("\n")
        print(f"Prompt: {prompt!r}")
        print(f"Completion: {output['text']!r}")

```
阅读 `example.py` 代码：
1. 加载模型 和 tokenizer，准备 prompt，并将 prompt 转为 chat 格式
2. 调用 llm.generate() 函数得到输出
3. 打印 prompt 和 输出

重点是 llm.generate() 函数。

核心请求链路如下：

```text
example.py --> generate() --> add_request() (encoding happens here) --> Sequence() --> self.scheduler.add() --> self.step() --> model_runner.call() --> postprocess()
```

