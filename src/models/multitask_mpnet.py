import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

class MultiTaskMPNet(nn.Module):
    def __init__(self, model_name='all-mpnet-base-v2', num_classes_task_a=4, num_classes_task_b=2):
        """
        Initializes the MultiTaskMPNet model with separate classification heads for each task.
        
        Args:
            model_name (str): Name of the pre-trained transformer model.
            num_classes_task_a (int): Number of classes for Task A (Sentence Classification).
            num_classes_task_b (int): Number of classes for Task B (Sentiment Analysis).
        """
        super(MultiTaskMPNet, self).__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size
        
        # Task A: Sentence Classification Head
        self.classifier_a = nn.Linear(hidden_size, num_classes_task_a)
        
        # Task B: Sentiment Analysis Head
        self.classifier_b = nn.Linear(hidden_size, num_classes_task_b)
        
        # Initialize the tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
    def forward(self, input_ids, attention_mask):
        """
        Forward pass of the model.
        
        Args:
            input_ids (torch.Tensor): Token IDs.
            attention_mask (torch.Tensor): Attention masks.
        
        Returns:
            dict: Logits for both tasks.
        """
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output  # [CLS] token representation
        
        # Task A: Sentence Classification
        logits_a = self.classifier_a(pooled_output)  # Shape: (batch_size, num_classes_task_a)
        
        # Task B: Sentiment Analysis
        logits_b = self.classifier_b(pooled_output)  # Shape: (batch_size, num_classes_task_b)
        
        return {'logits_a': logits_a, 'logits_b': logits_b}
