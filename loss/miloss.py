import torch
import torch.nn.functional as F

class MILOSS(torch.nn.Module):
    def __init__(self, temperature=0.1):
        """
        互信息蒸馏损失（InfoNCE）。
        :param temperature: 控制对比学习强度的超参数（越小，对比效果越强）
        """
        super(MILOSS, self).__init__()
        self.temperature = temperature

    def forward(self, f_teacher, f_student):

        N, D = f_teacher.shape
        # 归一化特征
        f_teacher_norm = F.normalize(f_teacher, dim=-1)  # (N, D)
        f_student_norm = F.normalize(f_student, dim=-1)  # (N, D)

        # 计算相似度矩阵 (N, N)
        similarity_matrix = torch.mm(f_teacher_norm, f_student_norm.T) / self.temperature  # (N, N)

        # InfoNCE 损失：希望第 i 行第 i 列最大，即 (i,i) 是正样本
        labels = torch.arange(N).to(similarity_matrix.device)
        loss = F.cross_entropy(similarity_matrix, labels)

        return loss