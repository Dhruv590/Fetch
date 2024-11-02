import torch
import pandas as pd
from transformers import AutoModel, AutoTokenizer
from models.multitask_mpnet import MultiTaskMPNet
import json
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, classification_report
import warnings

# Ignore all warnings
warnings.filterwarnings("ignore")

# Define paths
model_path = '/Users/dhruv590/Projects/Fetch/models/multitask_mpnet'
test_data_path = '/Users/dhruv590/Projects/Fetch/processed_data/test.csv'
label_mapping_path = '/Users/dhruv590/Projects/Fetch/processed_data/label_mappings.json'

# Load label mappings
with open(label_mapping_path, 'r') as f:
    label_mappings = json.load(f)

sentiment_mapping = label_mappings['sentiment_mapping']
category_mapping = label_mappings['category_mapping']
inverse_sentiment_mapping = {v: k for k, v in sentiment_mapping.items()}
inverse_category_mapping = {v: k for k, v in category_mapping.items()}

# Custom Dataset Class for the Test Set
class MultiTaskDataset(Dataset):
    def __init__(self, df):
        self.sentences = df['Sentence'].tolist()
        self.labels_a = df['category_label'].tolist()
        self.labels_b = df['sentiment_label'].tolist()
    
    def __len__(self):
        return len(self.sentences)
    
    def __getitem__(self, idx):
        return {
            'sentence': self.sentences[idx],
            'label_a': torch.tensor(self.labels_a[idx], dtype=torch.long),
            'label_b': torch.tensor(self.labels_b[idx], dtype=torch.long)
        }

# Function to evaluate the model and calculate relevant scores
def evaluate_model(model, dataloader, device):
    model.eval()
    true_labels_a, pred_labels_a = [], []
    true_labels_b, pred_labels_b = [], []
    
    with torch.no_grad():
        for batch in dataloader:
            # Prepare inputs
            sentences = batch['sentence']
            labels_a = batch['label_a'].to(device)
            labels_b = batch['label_b'].to(device)
            inputs = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt').to(device)
            
            # Make predictions
            outputs = model(inputs['input_ids'], inputs['attention_mask'])
            logits_a = outputs['logits_a']
            logits_b = outputs['logits_b']
            
            # Collect true labels and predictions for Task A
            pred_a = torch.argmax(logits_a, dim=1)
            true_labels_a.extend(labels_a.cpu().numpy())
            pred_labels_a.extend(pred_a.cpu().numpy())
            
            # Collect true labels and predictions for Task B
            pred_b = torch.argmax(logits_b, dim=1)
            true_labels_b.extend(labels_b.cpu().numpy())
            pred_labels_b.extend(pred_b.cpu().numpy())
    
    # Calculate accuracy for each task
    accuracy_a = accuracy_score(true_labels_a, pred_labels_a)
    accuracy_b = accuracy_score(true_labels_b, pred_labels_b)
    
    # Calculate precision, recall, and F1-score for each task
    report_a = classification_report(true_labels_a, pred_labels_a, target_names=list(inverse_category_mapping.values()))
    report_b = classification_report(true_labels_b, pred_labels_b, target_names=list(inverse_sentiment_mapping.values()))
    
    return accuracy_a, accuracy_b, report_a, report_b

# Main evaluation function
def main():
    # Load the tokenizer and model
    global tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = MultiTaskMPNet(model_name=model_path, num_classes_task_a=len(category_mapping), num_classes_task_b=len(sentiment_mapping))
    model.encoder = AutoModel.from_pretrained(model_path)
    model.classifier_a.load_state_dict(torch.load(f"{model_path}/classifier_a.pt"))
    model.classifier_b.load_state_dict(torch.load(f"{model_path}/classifier_b.pt"))
    
    # Move model to device
    device = torch.device('mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu'))
    model.to(device)
    
    # Load the test data
    test_df = pd.read_csv(test_data_path)
    test_dataset = MultiTaskDataset(test_df)
    test_dataloader = DataLoader(test_dataset, batch_size=8, shuffle=False)
    
    # Evaluate the model
    accuracy_a, accuracy_b, report_a, report_b = evaluate_model(model, test_dataloader, device)
    
    # Print results in a nicely formatted way
    print("=== Task A (Sentence Classification) Results ===")
    print(f"Accuracy: {accuracy_a * 100:.2f}%")
    print("Classification Report:")
    print(report_a)
    
    print("\n=== Task B (Sentiment Analysis) Results ===")
    print(f"Accuracy: {accuracy_b * 100:.2f}%")
    print("Classification Report:")
    print(report_b)

if __name__ == "__main__":
    main()