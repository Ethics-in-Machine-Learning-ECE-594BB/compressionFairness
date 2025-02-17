import torch
import torch.nn as nn
import torch.nn.functional as F

class DistillationLoss(nn.Module):
    def __init__(self, alpha=0.5, temperature=3.0):
        """
        Knowledge Distillation Loss
        :param alpha: Weight for ground truth loss
        :param temperature: Softening factor for logits
        """
        super(DistillationLoss, self).__init__()
        self.alpha = alpha
        self.temperature = temperature
        self.criterion_hard = nn.BCELoss()  # Hard labels loss
        self.criterion_soft = nn.KLDivLoss(reduction="batchmean")  # Soft labels loss

    def forward(self, student_logits, teacher_logits, true_labels):
        """
        Compute loss for knowledge distillation
        :param student_logits: Logits from student model
        :param teacher_logits: Logits from teacher model
        :param true_labels: Ground truth labels
        """
        teacher_probs = F.softmax(teacher_logits / self.temperature, dim=1)
        student_probs = F.log_softmax(student_logits / self.temperature, dim=1)

        soft_loss = self.criterion_soft(student_probs, teacher_probs) * (self.temperature ** 2)
        hard_loss = self.criterion_hard(student_logits, true_labels)

        return self.alpha * hard_loss + (1 - self.alpha) * soft_loss
