import torch
from transformers import Trainer

from torch.nn import functional as F
class DistillationTrainer(Trainer):
    def __init__(self, *args, teacher_model=None, temperature=2.0, alpha=0.5, **kwargs):
        assert teacher_model is not None
        super().__init__(*args, **kwargs)
        self.teacher_model = teacher_model
        self.teacher_model.eval()
        self.temperature = temperature
        self.alpha = alpha

    def compute_loss(self, model, inputs, return_outputs=False):
        student_outputs = model(**inputs)
        student_logits = student_outputs.logits

        with torch.no_grad():
            teacher_outputs = self.teacher_model(**inputs)
            teacher_logits = teacher_outputs.logits

        # print(inputs.keys())
        # print(*map(lambda x: x.shape, inputs.values()))
        # print(student_logits.shape, teacher_logits.shape)

        # Reshape for cross_entropy
        num_classes = student_logits.shape[-1]
        logits_for_loss = student_logits.view(-1, num_classes)
        labels_for_loss = inputs['labels'].view(-1)

        # hard loss
        hard_loss = F.cross_entropy(logits_for_loss, labels_for_loss)

        # soft loss
        soft_labels = F.softmax(teacher_logits / self.temperature, dim=-1)
        # soft predictions (log-probs for KL loss)
        soft_predictions = F.log_softmax(student_logits / self.temperature, dim=-1)

        distillation_loss = F.kl_div(soft_predictions, soft_labels, reduction='batchmean') * (self.temperature ** 2)
        total_loss = self.alpha * distillation_loss + (1 - self.alpha) * hard_loss

        return (total_loss, student_outputs) if return_outputs else total_loss
