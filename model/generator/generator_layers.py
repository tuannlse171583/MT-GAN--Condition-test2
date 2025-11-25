import torch
from torch import nn

from model.utils import MaskedAttention


class GRU(nn.Module):
    def __init__(self, code_num, hidden_dim, max_len, device=None):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.max_len = max_len
        self.device = device

        self.gru_cell = nn.GRUCell(input_size=code_num, hidden_size=hidden_dim)
        self.hidden2codes = nn.Sequential(
            nn.Linear(hidden_dim, code_num),
            nn.Sigmoid()
        )

    def step(self, x, h=None):
        h_n = self.gru_cell(x, h)
        codes = self.hidden2codes(h_n)
        return codes, h_n

    def forward(self, noise):
        codes = self.hidden2codes(noise)
        h = torch.zeros(len(codes), self.hidden_dim, device=self.device)
        samples, hiddens = [], []
        for _ in range(self.max_len):
            samples.append(codes)
            codes, h = self.step(codes, h)
            hiddens.append(h)
        samples = torch.stack(samples, dim=1)
        hiddens = torch.stack(hiddens, dim=1)

        return samples, hiddens


class SmoothCondition(nn.Module):
    def __init__(self, code_num, attention_dim, embed_dim=None):
        super().__init__()
        if embed_dim is None:
            embed_dim = attention_dim  # dùng attention_dim làm kích thước embedding mặc định
        self.attention = MaskedAttention(code_num, attention_dim)
        self.embedding = nn.Embedding(code_num, embed_dim)
        # Chuyển đổi embedding thành vector điều kiện có cùng độ dài code_num
        self.cond_transform = nn.Linear(embed_dim, code_num) if embed_dim != code_num else nn.Identity()
        # Lớp tạo vector gating cho mỗi chiều đặc trưng dựa trên embedding của mã mục tiêu
        self.gate_layer = nn.Linear(embed_dim, code_num)
        # Tham số scale có thể học để điều chỉnh cường độ tín hiệu điều kiện
        self.scale = nn.Parameter(torch.tensor(1.0))
        
    def forward(self, x, lens, target_codes):
        # Tính trọng số chú ý (attention scores) theo chuỗi (masked)
        score = self.attention(x, lens)            # shape: (B, T)
        # Lấy embedding của mã đích cho mỗi phần tử batch
        embed_vec = self.embedding(target_codes)   # shape: (B, embed_dim)
        # Tạo vector điều kiện từ embedding và chuẩn hóa về [0,1] (sigmoid) để tránh giá trị âm hoặc >1
        cond_vec = torch.sigmoid(self.cond_transform(embed_vec))  # shape: (B, code_num)
        # Tạo vector gating cho từng chiều đặc trưng (giá trị trong [0,1])
        gate_vec = torch.sigmoid(self.gate_layer(embed_vec))      # shape: (B, code_num)
        gate_vec = gate_vec.unsqueeze(1)        # shape: (B, 1, code_num) để broadcast theo thời gian
        # Mở rộng tín hiệu điều kiện theo thời gian với trọng số attention và hệ số scale học được
        cond_tensor = score.unsqueeze(-1) * cond_vec.unsqueeze(1) * self.scale  # shape: (B, T, code_num)
        # Kết hợp có kiểm soát giữa tín hiệu gốc và tín hiệu điều kiện qua gating
        x = x * (1 - gate_vec) + cond_tensor * gate_vec   # shape: (B, T, code_num)
        return x
