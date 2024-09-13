import json
import pandas as pd
import torch
from datasets import Dataset
from modelscope import snapshot_download, AutoTokenizer
from swanlab.integration.huggingface import SwanLabCallback
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
)
import os
import swanlab

# 常量定义
MODEL_DIR = "./qwen/Qwen2-1.5B-Instruct"
TRAIN_JSONL_PATH = "train.jsonl"
TEST_JSONL_PATH = "test.jsonl"
NEW_TRAIN_JSONL_PATH = "new_train.jsonl"
NEW_TEST_JSONL_PATH = "new_test.jsonl"
MAX_LENGTH = 384


def convert_dataset_format(input_path, output_path):
    """
    将原始数据集转换为大模型微调所需数据格式。
    """
    reformatted_data = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            message = {
                "instruction": "你是一个文本分类领域的专家，你会接收到一段文本和几个潜在的分类选项，请输出文本内容的正确类型",
                "input": f"文本: {data['text']}, 类型选项: {data['category']}",
                "output": data["output"],
            }
            reformatted_data.append(message)

    with open(output_path, "w", encoding="utf-8") as f:
        for message in reformatted_data:
            f.write(json.dumps(message, ensure_ascii=False) + "\n")


def preprocess_data(example, tokenizer):
    """
    对数据进行预处理，包括tokenize、截断和padding。
    """
    instruction = tokenizer(
        f"<|im_start|>system\n你是一个文本分类领域的专家...\n<|im_end|>\n<|im_start|>user\n{example['input']}<|im_end|>\n<|im_start|>assistant\n",
        add_special_tokens=False,
    )
    response = tokenizer(example["output"], add_special_tokens=False)

    input_ids = instruction["input_ids"] + response["input_ids"] + [tokenizer.pad_token_id]
    attention_mask = instruction["attention_mask"] + response["attention_mask"] + [1]
    labels = [-100] * len(instruction["input_ids"]) + response["input_ids"] + [tokenizer.pad_token_id]

    if len(input_ids) > MAX_LENGTH:
        input_ids = input_ids[:MAX_LENGTH]
        attention_mask = attention_mask[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]

    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def generate_response(messages, model, tokenizer, device="cuda"):
    """
    根据用户输入生成模型的预测响应。
    """
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([text], return_tensors="pt").to(device)

    generated_ids = model.generate(inputs.input_ids, max_new_tokens=512)
    response_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, generated_ids)]
    response = tokenizer.batch_decode(response_ids, skip_special_tokens=True)[0]

    return response


def main():
    # 下载模型
    model_dir = snapshot_download("qwen/Qwen2-1.5B-Instruct", cache_dir="./", revision="master")

    # 初始化模型与tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, use_fast=False, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, device_map="auto", torch_dtype=torch.bfloat16)
    model.enable_input_require_grads()

    # 检查数据集并转换格式
    if not os.path.exists(NEW_TRAIN_JSONL_PATH):
        convert_dataset_format(TRAIN_JSONL_PATH, NEW_TRAIN_JSONL_PATH)
    if not os.path.exists(NEW_TEST_JSONL_PATH):
        convert_dataset_format(TEST_JSONL_PATH, NEW_TEST_JSONL_PATH)

    # 加载和处理数据集
    train_df = pd.read_json(NEW_TRAIN_JSONL_PATH, lines=True)
    train_dataset = Dataset.from_pandas(train_df).map(lambda x: preprocess_data(x, tokenizer), remove_columns=train_df.columns)

    # 配置LoRA微调
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        inference_mode=False,
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
    )
    model = get_peft_model(model, lora_config)

    # 训练参数
    training_args = TrainingArguments(
        output_dir="./output/Qwen1.5",
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        logging_steps=10,
        num_train_epochs=2,
        save_steps=100,
        learning_rate=1e-4,
        save_on_each_node=True,
        gradient_checkpointing=True,
        report_to="none",
    )

    # 使用SwanLab Callback
    swanlab_callback = SwanLabCallback(
        project="Qwen2-finetune",
        experiment_name="Qwen2-1.5B-Instruct",
        description="微调Qwen2-1.5B-Instruct模型。",
        config={"model": "qwen/Qwen2-1.5B-Instruct", "dataset": "zh_cls_fudan-news"},
    )

    # 训练模型
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
        callbacks=[swanlab_callback],
    )

    trainer.train()

    # 预测测试集前10条
    test_df = pd.read_json(NEW_TEST_JSONL_PATH, lines=True).head(10)
    test_results = []
    for _, row in test_df.iterrows():
        messages = [
            {"role": "system", "content": row["instruction"]},
            {"role": "user", "content": row["input"]},
        ]
        response = generate_response(messages, model, tokenizer)
        test_results.append(swanlab.Text(f"Q: {row['input']}\nA: {response}", caption=response))

    swanlab.log({"Prediction": test_results})
    swanlab.finish()


if __name__ == "__main__":
    main()
