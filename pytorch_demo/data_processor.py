import torch
from torch.utils.data import Dataset
import sentencepiece as spm

class QADataset(Dataset):
    def __init__(self, questions, answers, tokenizer, max_length=128):  # 更新默认最大长度
        self.questions = questions
        self.answers = answers
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.vocab_size = self.tokenizer.get_piece_size()
    
    def __len__(self):
        return len(self.questions)
    
    def __getitem__(self, idx):
        question = self.questions[idx]
        answer = self.answers[idx]
        
        # 编码问题和答案
        q_ids = self.tokenizer.encode_as_ids(question)
        a_ids = self.tokenizer.encode_as_ids(answer)
        
        # 添加特殊token
        q_ids = [1] + q_ids + [2]  # 1: BOS, 2: EOS
        a_ids = [1] + a_ids + [2]  # 1: BOS, 2: EOS
        
        # 填充或截断到固定长度
        q_ids = q_ids[:self.max_length] + [0] * (self.max_length - len(q_ids))
        a_ids = a_ids[:self.max_length] + [0] * (self.max_length - len(a_ids))
        
        # 创建attention mask
        attention_mask = [1 if id != 0 else 0 for id in q_ids]
        
        return {
            'input_ids': torch.tensor(q_ids),
            'attention_mask': torch.tensor(attention_mask),
            'labels': torch.tensor(a_ids)
        }

def create_tokenizer(vocab_size=300):
    # 准备训练数据
    questions, answers = prepare_data()
    corpus = questions + answers
    
    # 将语料写入临时文件
    with open('temp_corpus.txt', 'w', encoding='utf-8') as f:
        for text in corpus:
            f.write(text + '\n')
    
    # 训练tokenizer
    spm.SentencePieceTrainer.train(
        input='temp_corpus.txt',
        model_prefix='spm_qa',
        vocab_size=vocab_size,
        character_coverage=0.9995,
        model_type='bpe',
        pad_id=0,
        bos_id=1,
        eos_id=2,
        unk_id=3
    )
    
    # 加载训练好的tokenizer
    tokenizer = spm.SentencePieceProcessor()
    tokenizer.load('spm_qa.model')
    return tokenizer

def prepare_data():
    """准备训练数据"""
    qa_pairs = [
        ("什么是人工智能？", "人工智能是一门研究如何让计算机模拟人类智能的科学，包括机器学习、深度学习、自然语言处理等领域。"),
        ("深度学习是什么？", "深度学习是机器学习的一个分支，通过多层神经网络从数据中学习特征表示，已在图像识别、自然语言处理等领域取得突破性进展。"),
        ("什么是神经网络？", "神经网络是一种模仿生物神经系统的计算模型，由多个相互连接的神经元组成，能够通过训练学习复杂的模式和关系。"),
        ("注意力机制是什么？", "注意力机制是一种让模型关注输入中重要部分的技术，在自然语言处理和计算机视觉中广泛应用，是Transformer等模型的核心组件。"),
        ("什么是自然语言处理？", "自然语言处理是人工智能的一个重要分支，研究计算机理解和处理人类语言的方法，包括机器翻译、文本分类、问答系统等应用。"),
        ("机器学习的基本类型有哪些？", "机器学习主要包括监督学习、无监督学习和强化学习三种基本类型，分别用于不同的学习任务和场景。"),
        ("Python的主要特点是什么？", "Python是一种简单易学、可读性强的编程语言，支持面向对象编程，有丰富的第三方库，广泛应用于数据科学和人工智能领域。"),
        ("什么是卷积神经网络？", "卷积神经网络是一种专门用于处理网格化数据（如图像）的深度学习模型，通过卷积层提取特征，在计算机视觉任务中表现优异。"),
        ("强化学习是什么？", "强化学习是一种通过试错学习获得最优策略的方法，智能体通过与环境交互，根据获得的奖励来改进自己的行为。"),
        ("什么是迁移学习？", "迁移学习是一种将已训练模型的知识迁移到新任务上的方法，可以减少数据需求和训练时间，提高模型性能。"),
        ("什么是模型评估？", "模型评估是衡量机器学习模型性能的过程，常用指标包括准确率、精确率、召回率、F1分数等，帮助我们了解模型的优缺点。"),
        ("什么是过拟合？", "过拟合是指模型在训练数据上表现很好，但在新数据上表现差的现象，通常是由于模型过于复杂或训练数据不足导致。"),
        ("什么是梯度下降？", "梯度下降是一种优化算法，通过计算损失函数的梯度，沿着梯度的反方向更新模型参数，使损失函数逐步减小。"),
        ("什么是批量归一化？", "批量归一化是一种深度学习中的正则化技术，通过标准化每一层的输入来加速训练过程，提高模型的稳定性和性能。"),
        ("什么是循环神经网络？", "循环神经网络是一种处理序列数据的神经网络，通过循环连接可以保持历史信息，适用于自然语言处理和时间序列分析。"),
        ("什么是生成对抗网络？", "生成对抗网络是一种深度学习模型，由生成器和判别器组成，通过对抗训练可以生成高质量的图像、文本等数据。"),
        ("什么是词嵌入？", "词嵌入是将词语映射到低维向量空间的技术，可以捕捉词语之间的语义关系，是自然语言处理中的重要基础技术。"),
        ("什么是计算机视觉？", "计算机视觉是让计算机理解和处理图像数据的技术，包括图像分类、目标检测、图像分割等任务，广泛应用于自动驾驶、医疗诊断等领域。"),
        ("什么是语音识别？", "语音识别是将语音信号转换为文本的技术，基于深度学习模型，在智能助手、语音输入等应用中发挥重要作用。"),
        ("什么是推荐系统？", "推荐系统是根据用户历史行为和兴趣为其推荐内容的系统，广泛应用于电商、视频网站等平台，提高用户体验和平台价值。")
    ]
    
    # 保存问答对到临时文件
    with open("temp_corpus.txt", "w", encoding="utf-8") as f:
        for question, answer in qa_pairs:
            f.write(f"{question}\n{answer}\n")
    
    return qa_pairs
