import torch
import sentencepiece as spm
from model import CustomTransformer
import sys

class QAPredictor:
    def __init__(self, model_path, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        print(f"Using device: {device}")
        
        # 设置控制台编码
        sys.stdout.reconfigure(encoding='utf-8')
        
        # 加载模型和tokenizer
        checkpoint = torch.load(model_path, map_location=device)
        self.tokenizer = spm.SentencePieceProcessor()
        self.tokenizer.load(checkpoint['tokenizer_model'])
        vocab_size = self.tokenizer.get_piece_size()
        print(f"Vocabulary size: {vocab_size}")
        
        # 初始化模型
        self.model = CustomTransformer(
            vocab_size=vocab_size,
            d_model=256,
            nhead=8,
            num_encoder_layers=3,
            num_decoder_layers=3
        ).to(device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        print("Model loaded and ready for predictions")
    
    def predict(self, question, max_length=100):
        # 对输入问题进行编码
        question_ids = self.tokenizer.encode_as_ids(question)
        question_ids = [1] + question_ids + [2]  # 1: BOS, 2: EOS
        
        # 填充到固定长度
        input_length = len(question_ids)
        question_ids = question_ids + [0] * (max_length - input_length)
        
        # 转换为tensor
        input_ids = torch.tensor([question_ids]).to(self.device)
        src_padding_mask = torch.zeros(1, max_length, dtype=torch.bool).to(self.device)
        src_padding_mask[0, input_length:] = True
        
        # 初始化输出序列
        output_ids = torch.tensor([[1]]).to(self.device)  # 开始符号
        
        # 逐个生成输出token
        for _ in range(max_length):
            # 创建目标掩码
            tgt_padding_mask = torch.zeros(1, output_ids.size(1), dtype=torch.bool).to(self.device)
            
            # 模型预测
            with torch.no_grad():
                output = self.model(
                    input_ids,
                    output_ids,
                    src_padding_mask=src_padding_mask,
                    tgt_padding_mask=tgt_padding_mask
                )
            
            # 获取下一个token
            next_token = output[:, -1:].argmax(dim=-1)
            output_ids = torch.cat([output_ids, next_token], dim=1)
            
            # 如果生成了结束符，停止生成
            if next_token.item() == 2:  # EOS token
                break
        
        # 解码输出序列
        output_ids = output_ids[0].tolist()
        # 移除特殊token
        output_ids = [id for id in output_ids if id not in [0, 1, 2]]  # 移除PAD, BOS, EOS
        answer = self.tokenizer.decode(output_ids)
        return answer

def test_prediction(model_path):
    predictor = QAPredictor(model_path)
    test_questions = [
        "什么是人工智能？",
        "Python的主要特点是什么？"
    ]
    
    for question in test_questions:
        print(f"\n问题: {question}")
        answer = predictor.predict(question)
        print(f"回答: {answer}")

if __name__ == "__main__":
    model_path = "transformer_qa_model.pth"
    test_prediction(model_path)
