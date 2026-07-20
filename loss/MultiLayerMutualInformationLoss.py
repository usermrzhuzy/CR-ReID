import torch
import torch.nn as nn
import torch.nn.functional as F
class MultiLayerPurificationDistillLoss(nn.Module):
    def __init__(self, temperature=0.1, weight3=1.0, weight12=1.0, weight24=1.0, negative_weight=0.5):

        super(MultiLayerPurificationDistillLoss, self).__init__()
        self.temperature = temperature
        self.weight3 = weight3
        self.weight12 = weight12
        self.weight24 = weight24
        self.negative_weight = negative_weight

    def forward(self, teacher_feats, student_feats):
        assert len(teacher_feats) == len(student_feats), "different layer"
        total_loss = 0

        for idx, (t_feat, s_feat) in enumerate(zip(teacher_feats, student_feats)):
            if idx == 0:
                weight = self.weight3
            elif idx == 1:
                weight = self.weight12
            elif idx == 2:
                weight = self.weight24
            else:
                weight = 1.0

            total_loss += self.purification_info_nce_loss(s_feat, t_feat, weight)

        return total_loss

    def purification_info_nce_loss(self, student_feat, teacher_feat, weight=1.0):


        student_feat = student_feat.view(-1, student_feat.size(-1))
        teacher_feat = teacher_feat.view(-1, teacher_feat.size(-1))
        student_norm = F.normalize(student_feat, dim=-1)
        teacher_norm = F.normalize(teacher_feat, dim=-1)

        N = student_norm.size(0)
        labels = torch.arange(N).to(student_norm.device)


        sim_pos = torch.matmul(student_norm, teacher_norm.T) / self.temperature
        loss_pos = F.cross_entropy(sim_pos, labels)


        sim_neg = torch.matmul(student_norm, student_norm.T) / self.temperature
        mask = torch.eye(N, device=student_norm.device).bool()
        sim_neg = sim_neg.masked_fill(mask, float('-inf'))
        loss_neg = torch.logsumexp(sim_neg, dim=1).mean()


        return weight * (loss_pos + self.negative_weight * loss_neg)