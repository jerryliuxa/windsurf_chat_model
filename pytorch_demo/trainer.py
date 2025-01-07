import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from data_processor import QADataset, prepare_data, create_tokenizer
from model import CustomTransformer

def train_model(model, train_dataloader, num_epochs, device, learning_rate=0.001):
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # 忽略padding token
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.01)
    
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for batch_idx, batch in enumerate(train_dataloader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            # 前向传播
            outputs = model(
                input_ids,
                labels[:, :-1],  # 去除最后一个token
                src_padding_mask=(input_ids == 0),
                tgt_padding_mask=(labels[:, :-1] == 0)
            )
            
            # 计算损失
            loss = criterion(outputs.reshape(-1, outputs.size(-1)), labels[:, 1:].reshape(-1))
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % 10 == 0:
                print(f'Epoch: {epoch+1}, Batch: {batch_idx}, Loss: {loss.item():.4f}')
        
        avg_loss = total_loss / len(train_dataloader)
        print(f'Epoch {epoch+1} completed. Average Loss: {avg_loss:.4f}')
    
    return model

def setup_training():
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 创建并训练tokenizer
    tokenizer = create_tokenizer()
    vocab_size = tokenizer.get_piece_size()
    print(f"Vocabulary size: {vocab_size}")
    
    # 准备数据
    questions, answers = prepare_data()
    dataset = QADataset(questions, answers, tokenizer, max_length=128)  # 减小序列长度
    train_dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
    
    # 初始化模型
    model = CustomTransformer(
        vocab_size=vocab_size,
        d_model=256,  # 减小模型大小
        nhead=8,
        num_encoder_layers=3,  # 减少层数
        num_decoder_layers=3   # 减少层数
    ).to(device)
    print("Model initialized")
    
    # 训练模型
    num_epochs = 30
    trained_model = train_model(model, train_dataloader, num_epochs, device)
    print("Training completed")
    
    # 保存模型
    torch.save({
        'model_state_dict': trained_model.state_dict(),
        'tokenizer_model': 'spm_qa.model',
        'tokenizer_vocab': 'spm_qa.vocab'
    }, 'transformer_qa_model.pth')
    print("Model saved")
    
    return trained_model, tokenizer

if __name__ == "__main__":
    setup_training()
