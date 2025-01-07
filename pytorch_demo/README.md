# 中文问答Transformer模型

这是一个基于PyTorch实现的中文问答Transformer模型。该项目包含完整的数据处理、模型定义、训练和预测功能。

## 项目结构

- `data_processor.py`: 数据处理模块，包含数据集类和数据准备函数
- `model.py`: 模型定义模块，包含自定义Transformer模型的实现
- `trainer.py`: 训练模块，包含模型训练相关的功能
- `predictor.py`: 预测模块，用于使用训练好的模型进行预测
- `requirements.txt`: 项目依赖文件

## 安装依赖

```bash
pip install -r requirements.txt
```

## 使用方法

1. 训练模型：
```python
python trainer.py
```

2. 预测：
```python
python predictor.py
```

## 模型特点

- 使用自定义Transformer架构
- 支持中文问答
- 包含位置编码
- 使用BERT分词器
- 支持GPU训练（如果可用）

## 注意事项

- 当前使用的是示例训练数据，实际使用时请替换为您自己的数据
- 模型参数可以根据需要在代码中调整
- 训练完成后的模型会保存为`transformer_qa_model.pth`
