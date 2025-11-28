"""
Qwen3-8B 使用 vLLM 部署，支持 FP8 量化

量化说明：
1. FP8 量化可以将模型大小和显存占用减少约50%
2. 首次运行时vLLM会自动对模型进行量化，可能需要一些时间
3. 量化后的模型精度损失很小，但推理速度和显存效率显著提升
4. 如果GPU不支持FP8，vLLM会自动回退到其他精度

使用方法：
- 设置 use_fp8_w8a8=True 启用量化
- 设置 use_fp8_w8a8=False 使用原始精度
"""

import json
import os
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from modelscope import snapshot_download

model_dir = snapshot_download(
    'Qwen/Qwen3-8B', cache_dir='/root/autodl-tmp', revision='master')


# 自动下载模型时，指定使用modelscope; 否则，会从HuggingFace下载
os.environ['VLLM_USE_MODELSCOPE'] = 'True'


def get_completion(prompts, model, tokenizer=None, temperature=0.6, top_p=0.95, top_k=20, min_p=0, max_tokens=4096, max_model_len=8192, use_fp8_w8a8=False):
    stop_token_ids = [151645, 151643]
    # 创建采样参数。temperature 控制生成文本的多样性，top_p 控制核心采样的概率，top_k 通过限制候选词的数量来控制生成文本的质量和多样性, min_p 通过设置概率阈值来筛选候选词，从而在保证文本质量的同时增加多样性
    sampling_params = SamplingParams(temperature=temperature, top_p=top_p, top_k=top_k, min_p=min_p,
                                     # max_tokens 用于限制模型在推理过程中生成的最大输出长度
                                     max_tokens=max_tokens, stop_token_ids=stop_token_ids)

    # 初始化 vLLM 推理引擎，支持Q8量化
    if use_fp8_w8a8:
        # 使用FP8 W8A8量化（权重和激活都是8位）
        # 这将显著减少显存使用并提高推理速度
        llm = LLM(
            model=model,
            tokenizer=tokenizer,
            max_model_len=max_model_len,
            trust_remote_code=True,
            quantization="fp8",  # 启用FP8量化
            kv_cache_dtype="fp8"  # KV缓存也使用FP8，进一步节省显存
        )
    else:
        # 不使用量化
        llm = LLM(model=model, tokenizer=tokenizer,
                  max_model_len=max_model_len, trust_remote_code=True)

    outputs = llm.generate(prompts, sampling_params)
    return outputs


if __name__ == "__main__":
    # 初始化 vLLM 推理引擎
    model = '/root/autodl-tmp/Qwen/Qwen3-8B'  # 指定模型路径
    tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False)  # 加载分词器

    prompt = "给我一个关于大模型的简短介绍。"
    messages = [
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True  # 是否开启思考模式，默认为 True
    )

    # 启用Q8量化部署
    # use_fp8_w8a8=True 将使用FP8量化（8位浮点），可以减少约50%的显存使用
    # 注意：首次运行时vLLM会自动量化模型，可能需要一些时间
    outputs = get_completion(
        text,
        model,
        tokenizer=None,
        temperature=0.6,
        top_p=0.95,
        top_k=20,
        min_p=0,
        use_fp8_w8a8=True  # 启用Q8量化
    )  # 对于思考模式，官方建议使用以下参数：temperature = 0.6，TopP = 0.95，TopK = 20，MinP = 0。

    # 输出是一个包含 prompt、生成文本和其他信息的 RequestOutput 对象列表。
    # 打印输出。
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}, \nResponse: {generated_text!r}")
