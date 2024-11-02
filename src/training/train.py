import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AdamW, get_linear_schedule_with_warmup
from src.models.multitask_mpnet import MultiTaskMPNet
from src.utils.align_labels import align_labels
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import json

# Define constants
MODEL_NAME = 'sentence-transformers/all-mpnet-base-v2'
NUM_CLASSES_TASK_A = 4  # Based on your Category labels
NUM_CLASSES_TASK_B = 2  # Positive, Negative Sentiment
BATCH_SIZE = 8
EPOCHS = 3
BASE_LR = 1e-5
HEAD_LR = 1e-4
DEVICE = torch.device('mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu'))
TRAIN_CSV = 'processed_data/train.csv'
TEST_CSV = 'processed_data/test.csv'
LABEL_MAPPING_PATH = 'processed_data/label_mappings.json'

print(f"Using device: {DEVICE}")

# Custom Dataset class for multi-task data
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

# Load label mappings
def load_label_mappings(mapping_path):
    try:
        with open(mapping_path, 'r') as f:
            mappings = json.load(f)
        return mappings['sentiment_mapping'], mappings['category_mapping']
    except Exception as e:
        print(f"Error loading label mappings: {e}")
        return None, None

# Collate function for batch processing
def collate_fn(batch):
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    sentences = [item['sentence'] for item in batch]
    labels_a = torch.stack([item['label_a'] for item in batch])
    labels_b = torch.stack([item['label_b'] for item in batch])

    encoded = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')

    return {
        'input_ids': encoded['input_ids'].to(DEVICE),
        'attention_mask': encoded['attention_mask'].to(DEVICE),
        'labels_a': labels_a.to(DEVICE),
        'labels_b': labels_b.to(DEVICE)
    }

def main():
    # Load the label mappings
    sentiment_mapping, category_mapping = load_label_mappings(LABEL_MAPPING_PATH)
    if not sentiment_mapping or not category_mapping:
        print("Error: Could not load label mappings.")
        return

    # Load the datasets
    print("Loading training and testing data...")
    try:
        train_df = pd.read_csv(TRAIN_CSV)
        test_df = pd.read_csv(TEST_CSV)
        print(f"Training Set Size: {len(train_df)}")
        print(f"Testing Set Size: {len(test_df)}")
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # Initialize the model
    print("Initializing the model...")
    try:
        model = MultiTaskMPNet(model_name=MODEL_NAME, num_classes_task_a=NUM_CLASSES_TASK_A, num_classes_task_b=NUM_CLASSES_TASK_B)
        model.to(DEVICE)
        print("Model initialized and moved to device.")
    except Exception as e:
        print(f"Error initializing model: {e}")
        return

    # Create the datasets and dataloaders
    print("Creating datasets and dataloaders...")
    try:
        train_dataset = MultiTaskDataset(train_df)
        test_dataset = MultiTaskDataset(test_df)
        train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
        test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
        print("Dataloaders created.")
    except Exception as e:
        print(f"Error creating datasets or dataloaders: {e}")
        return

    # Set up the optimizer and scheduler
    optimizer_grouped_parameters = [
        {'params': model.encoder.embeddings.parameters(), 'lr': BASE_LR * 0.5},
        {'params': model.encoder.encoder.layer[:12].parameters(), 'lr': BASE_LR},
        {'params': model.encoder.encoder.layer[12:].parameters(), 'lr': BASE_LR * 2},
        {'params': model.classifier_a.parameters(), 'lr': HEAD_LR},
        {'params': model.classifier_b.parameters(), 'lr': HEAD_LR},
    ]
    optimizer = AdamW(optimizer_grouped_parameters, weight_decay=1e-2)
    total_steps = len(train_dataloader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(0.1 * total_steps), num_training_steps=total_steps)

    # Loss functions
    criterion_a = nn.CrossEntropyLoss()
    criterion_b = nn.CrossEntropyLoss()

    # Training loop
    print("Starting training...")
    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch + 1}/{EPOCHS}")
        print("-" * 10)
        
        # Training
        model.train()
        total_train_loss = 0
        for batch_idx, batch in enumerate(train_dataloader):
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            labels_a = batch['labels_a']
            labels_b = batch['labels_b']
            
            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask)
            logits_a = outputs['logits_a']
            logits_b = outputs['logits_b']
            
            loss_a = criterion_a(logits_a, labels_a)
            loss_b = criterion_b(logits_b, labels_b)
            loss = loss_a + loss_b
            total_train_loss += loss.item()
            
            loss.backward()
            optimizer.step()
            scheduler.step()

            if batch_idx % 10 == 0:
                print(f"Batch {batch_idx}/{len(train_dataloader)}, Loss: {loss.item():.4f}")
        
        avg_train_loss = total_train_loss / len(train_dataloader)
        print(f"Average Training Loss: {avg_train_loss:.4f}")
        
        # Validation
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for batch_idx, batch in enumerate(test_dataloader):
                input_ids = batch['input_ids']
                attention_mask = batch['attention_mask']
                labels_a = batch['labels_a']
                labels_b = batch['labels_b']
                
                outputs = model(input_ids, attention_mask)
                logits_a = outputs['logits_a']
                logits_b = outputs['logits_b']
                
                loss_a = criterion_a(logits_a, labels_a)
                loss_b = criterion_b(logits_b, labels_b)
                loss = loss_a + loss_b
                total_val_loss += loss.item()

                if batch_idx % 10 == 0:
                    print(f"Validation Batch {batch_idx}/{len(test_dataloader)}, Loss: {loss.item():.4f}")
        
        avg_val_loss = total_val_loss / len(test_dataloader)
        print(f"Average Validation Loss: {avg_val_loss:.4f}")
    
    # Save the model
    output_dir = 'models/multitask_mpnet'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    model.encoder.save_pretrained(output_dir)
    model.tokenizer.save_pretrained(output_dir)
    torch.save(model.classifier_a.state_dict(), os.path.join(output_dir, 'classifier_a.pt'))
    torch.save(model.classifier_b.state_dict(), os.path.join(output_dir, 'classifier_b.pt'))
    print(f"Model saved to {output_dir}")

if __name__ == "__main__":
    main()
