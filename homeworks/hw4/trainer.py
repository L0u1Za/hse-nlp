from transformers import TrainingArguments, Trainer
import torch
import torch.nn as nn
import torch.nn.functional as F

class DistilTrainer(Trainer):
    def __init__(self, *args, teacher_model, temperature, lambda_param, **kwargs):
        super().__init__(*args, **kwargs)
        self.teacher = teacher_model
        self.student = self.model
        self.loss_distillation = nn.KLDivLoss(reduction="batchmean")
        self.loss_student = nn.CrossEntropyLoss()

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.teacher.to(device)
        self.teacher.eval()
        self.temperature = temperature
        self.lambda_param = lambda_param

    def compute_loss(self, inputs, return_outputs=False):
        student_output = self.student(**inputs)
        print(student_output)

        with torch.no_grad():
          teacher_output = self.teacher(**inputs)
          print(teacher_output)
        # Compute soft targets for teacher and student
        soft_teacher = F.softmax(teacher_output.logits / self.temperature, dim=-1)
        soft_student = F.log_softmax(student_output.logits / self.temperature, dim=-1)

        # Compute the loss
        distillation_loss = self.loss_distillation(soft_student, soft_teacher) * (self.temperature ** 2)

        # Compute the true label loss
        student_loss = student_output.loss

        # Calculate final loss
        loss = (1. - self.lambda_param) * student_loss + self.lambda_param * distillation_loss
        return (loss, student_output) if return_outputs else loss