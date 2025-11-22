import torch
from torch import nn

from model.utils import MaskedAttention


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
        
        # ⭐️ Sửa 1: Tách hidden2codes thành Linear và Sigmoid
        self.hidden2logits = nn.Linear(hidden_dim, code_num)
        self.sigmoid = nn.Sigmoid() 
        # self.hidden2codes đã bị loại bỏ/thay thế

    def step(self, x, h=None):
        h_n = self.gru_cell(x, h)
        
        # ⭐️ Sửa 2: Tính Logits (z)
        logits = self.hidden2logits(h_n) 
        
        # ⭐️ Sửa 3: Trả về Logits (để áp dụng SmoothCondition) và Hidden State (h_n)
        return logits, h_n

    def forward(self, noise, smooth_conditioner=None, lens=None, target_codes=None):
        # ⭐️ Sửa 4: Logits và Codes ban đầu
        logits = self.hidden2logits(noise)
        codes = self.sigmoid(logits)
        
        # Nếu không có điều kiện, codes được sử dụng như bình thường.
        if smooth_conditioner is None:
            # Nếu không có SmoothCondition, logic giữ nguyên
            pass
        else:
            # Nếu có SmoothCondition, ta áp dụng ngay đầu tiên
            # Đảm bảo lens và target_codes được truyền vào
            if lens is None or target_codes is None:
                 raise ValueError("Phải cung cấp lens và target_codes khi sử dụng smooth_conditioner.")
            
            # ⭐️ Áp dụng SmoothCondition (Logits -> Codes)
            # Chúng ta sẽ truyền Logits vào SmoothCondition.
            codes = smooth_conditioner(logits, lens[:, 0], target_codes[:, 0])
            
        h = torch.zeros(len(codes), self.hidden_dim, device=self.device)
        samples, hiddens = [], []
        
        # Vòng lặp GRU
        for t in range(self.max_len):
            samples.append(codes)
            
            # ⭐️ Sửa 5: step trả về logits và h
            logits, h = self.step(codes, h)
            
            # ⭐️ Sửa 6: Áp dụng SmoothCondition trong vòng lặp (Logits -> Codes)
            if smooth_conditioner is not None:
                # Sử dụng dữ liệu cho bước thời gian t
                codes = smooth_conditioner(logits, lens[:, t], target_codes[:, t])
            else:
                # Nếu không có conditioner, áp dụng Sigmoid
                codes = self.sigmoid(logits)
            
            hiddens.append(h)
        
        # Giữ nguyên đầu ra samples và hiddens
        samples = torch.stack(samples, dim=1)
        hiddens = torch.stack(hiddens, dim=1)

        return samples, hiddens


class SmoothCondition(nn.Module):
    def __init__(self, code_num, attention_dim):
        super().__init__()
        self.attention = MaskedAttention(code_num, attention_dim)
        # Thêm hàm Sigmoid để áp dụng cuối cùng sau khi điều kiện hóa Logits
        self.sigmoid = nn.Sigmoid() 

    # ⭐️ Đầu vào x bây giờ là LOGITS (z)
    # Output x vẫn là codes (x)
    def forward(self, x, lens, target_codes):
        # 1. Tính Codes/Xác suất từ Logits (để MaskedAttention sử dụng)
        codes_for_attention = self.sigmoid(x) 
        # Giả định MaskedAttention vẫn cần đầu vào là Codes/Xác suất
        score = self.attention(codes_for_attention, lens) 

        # 2. Tạo score_tensor cùng kích thước với Logits (x)
        score_tensor = torch.zeros_like(x) 
        
        # Sử dụng tuple indexing để gán điểm số chú ý
        # Lấy chỉ số (batch_idx, time_step=0, target_code)
        # Vì SmoothCondition được gọi trong vòng lặp GRU, đầu vào x chỉ có 2 chiều: (Batch_size, Code_Num)
        # Tức là chỉ có 1 bước thời gian (time step)
        
        # Chúng ta cần điều chỉnh logic gán cho đầu vào (B, C)
        
        if x.dim() == 2: # Trường hợp khi gọi trong vòng lặp GRU: (B, C)
            # score có shape (B,) vì chỉ có 1 bước thời gian
            score_tensor[torch.arange(len(x)), target_codes.long()] = score
        else: # Trường hợp gọi ban đầu hoặc nếu x có shape (B, T, C)
            # Sử dụng logic gán cũ (giả định T=max_len)
            b_idx = torch.arange(x.size(0)).unsqueeze(1).repeat(1, x.size(1))
            t_idx = torch.arange(x.size(1)).unsqueeze(0).repeat(x.size(0), 1)
            score_tensor[b_idx, t_idx, target_codes] = score

        # 3. ⭐️ Áp dụng phép cộng lên LOGITS (Pre-Activation Conditioning)
        x = x + score_tensor
        
        # 4. ⭐️ Áp dụng Sigmoid MƯỢT MÀ, giữ nguyên tên biến đầu ra x
        x = self.sigmoid(x)
        
        # Loại bỏ torch.clip(x, max=1)
        # Giữ nguyên tên biến đầu ra x
        return x
