# rare_boost_sampler.py
import random

class RareBoostSampler:
    def __init__(self, freq_per_id, p_boost=0.1):
        """
        freq_per_id: list[int] – tần suất từng ICD theo ID (mapping từ code2id)
        p_boost: xác suất boost ICD có freq < 4 (default 20%)
        """
        self.code_num = len(freq_per_id)
        self.p_boost = p_boost

        # Nhóm ICD xuất hiện < 4 lần
        self.boost_group = [i for i, f in enumerate(freq_per_id) if f < 4]

        if len(self.boost_group) == 0:
            print("⚠ WARNING: Không có ICD nào xuất hiện < 2 lần!")
            self.boost_group = list(range(self.code_num))

        print(f"🔍 Boost group (ICD < 4 lần),p=0.1: {len(self.boost_group)} mã")

    def sample(self):
        """
        Trả về target ICD ID:
        - 80% giữ nguyên uniform sampling như MTGAN gốc
        - 20% boost nhóm ICD hiếm (freq < 4)
        """
        # Sampling gốc của MTGAN
        original_target = random.randint(0, self.code_num - 1)

        # Boost ICD hiếm
        if random.random() < self.p_boost:
            return random.choice(self.boost_group)

        return original_target
