# Prompt Evolution Framework

这是一个用于生成和演化AI提示词的框架,可以通过多种策略来增强原始提示词的复杂度和深度,基于[WizardLM: Empowering Large Language Models to Follow Complex Instructions](https://arxiv.org/abs/2304.12244)。

## 功能特点

- 支持多种提示词演化策略:
  - 约束增强 - 添加额外的限制条件
  - 深度扩展 - 增加查询的深度和广度 
  - 具体化处理 - 将抽象概念转换为具体实例
  - 推理强化 - 将简单思维过程转化为多步骤推理
  - 广度扩展 - 生成相同领域的稀有提示词

- 支持多进程并行处理
- 兼容OpenAI和Alpaca数据格式
- 内置重试机制确保API调用稳定性

## 安装要求

- Python 3.7+
- OpenAI API密钥
- 相关Python包:
  - openai
  - tqdm
  - requests

## 环境变量配置

在运行之前需要设置以下环境变量:
```bash
export OPENAI_API_KEY="你的OpenAI API密钥"
export OPENAI_API_BASE_URL="https://api.openai.com/v1" # 可选
export OPENAI_MODEL_NAME="gpt-4" # 可选
```


## 使用方法

1. 准备输入数据(OpenAI格式的JSON文件)

2. 运行主程序:

```bash
python main.py --input_file input.json --output_file output.json --num_processes 4
```

参数说明:
- `--input_file`: 输入JSON文件路径
- `--output_file`: 输出JSON文件路径
- `--num_processes`: 并行处理的进程数

## 代码结构

- `main.py`: 主程序入口
- `depth.py`: 实现深度相关的提示词演化策略
- `breadth.py`: 实现广度相关的提示词演化策略
- `openai_access.py`: OpenAI API访问接口
- `utils.py`: 工具函数,包括数据格式转换等

## 数据格式

### 输入格式(OpenAI):
```json
{
  "messages": [
    {"role": "system", "content": "系统提示"},
    {"role": "user", "content": "用户输入"},
    {"role": "assistant", "content": "助手回复"}
  ]
}
```

### 输出格式(OpenAI):
```json
{
  "messages": [
    {"role": "system", "content": "系统提示"},
    {"role": "user", "content": "用户输入"},
    {"role": "assistant", "content": "助手回复"}
  ]
}
```

## 许可证

MIT

## 贡献指南

欢迎提交Issue和Pull Request来帮助改进这个项目。
