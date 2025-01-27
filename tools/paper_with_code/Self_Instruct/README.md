# Self-Instruct 数据生成工具

这是一个基于Self-Instruct方法的指令数据生成工具,可以通过少量种子任务自动生成大规模的指令数据集。该工具可用于训练和微调大型语言模型。基于[self-instruct](https://github.com/yizhongw/self-instruct)

## 功能特点

- 从种子任务自动生成新的指令
- 自动分类指令为分类任务或生成任务
- 为每个指令生成多个输入输出实例
- 过滤和处理生成的实例
- 支持转换为OpenAI格式的训练数据

## 环境变量配置

在运行之前需要设置以下环境变量:
```bash
export OPENAI_API_KEY="你的OpenAI API密钥"
export OPENAI_API_BASE_URL="https://api.openai.com/v1" # 可选
export OPENAI_MODEL_NAME="gpt-4" # 可选
```


## 使用方法
1.准备种子任务数据,格式为JSONL:
```
{
  "instruction": "将以下句子翻译成英文",
  "is_classification": false,
  "instances": [
    {
      "input": "今天天气很好",
      "output": "The weather is nice today"
    }
  ]
}
```
2. 生成指令数据:
```
python main.py \
  --work_dir data/output \
  --seed_tasks_path data/seed_tasks.jsonl \
  --output_file data/generated_data.json \
  --num_instructions_to_generate 100
  ```
  主要参数说明:
- work_dir: 输出工作目录
- seed_tasks_path: 种子任务文件路径
- output_file: 最终输出文件路径
- num_instructions_to_generate: 需要生成的指令数量
- classification_tasks_only: 仅生成分类任务
- generation_tasks_only: 仅生成生成任务
- max_instances_to_generate: 每个指令生成的最大实例数

## 数据格式
```json
{
  "messages": [
    {
      "role": "system",
      "content": "系统提示"
    },
    {
      "role": "user", 
      "content": "用户输入"
    },
    {
      "role": "assistant",
      "content": "助手回复"
    }
  ]
}
```