"""
Job Title Classification Module

This module implements two approaches for classifying job titles:
1. Baseline TF-IDF + Logistic Regression
2. Fine-tuned DistilBERT transformer model

The module also includes evaluation metrics and model saving functionality.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score
from sklearn.pipeline import Pipeline
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments
from torch.utils.data import Dataset
import pickle
import os
import warnings
warnings.filterwarnings('ignore')

# Custom dataset class for transformer model
class JobTitleDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# Function to create job categories based on keywords
def create_job_categories(job_title):
    """
    Create job categories based on keywords in job titles
    
    Args:
        job_title (str): Job title text
        
    Returns:
        str: Job category
    """
    title = job_title.lower()
    
    # Data Scientist roles
    if any(keyword in title for keyword in ['data scientist', 'data science']):
        return 'Data Scientist'
    
    # Data Engineer roles
    elif any(keyword in title for keyword in ['data engineer', 'data engineering']):
        return 'Data Engineer'
    
    # Data Analyst roles
    elif any(keyword in title for keyword in ['data analyst', 'analytics']):
        return 'Data Analyst'
    
    # Machine Learning roles
    elif any(keyword in title for keyword in ['machine learning', 'ml engineer', 'ai']):
        return 'ML Engineer'
    
    # Business Intelligence roles
    elif any(keyword in title for keyword in ['business intelligence', 'bi']):
        return 'BI Developer'
    
    # Database roles
    elif any(keyword in title for keyword in ['database', 'db']):
        return 'Database Admin'
    
    # Other data-related roles
    elif 'data' in title:
        return 'Data Specialist'
    
    # Default category
    else:
        return 'Other'

def prepare_data(df):
    """
    Prepare data for classification
    
    Args:
        df (pd.DataFrame): Input dataframe
        
    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    # Create job categories
    df['job_category'] = df['job_title_clean'].apply(create_job_categories)
    
    # Filter out 'Other' category for better training
    df_filtered = df[df['job_category'] != 'Other']
    
    # Prepare features and labels
    X = df_filtered['job_title_clean']
    y = df_filtered['job_category']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    return X_train, X_test, y_train, y_test

def train_baseline_model(X_train, X_test, y_train, y_test):
    """
    Train baseline TF-IDF + Logistic Regression model
    
    Args:
        X_train, X_test, y_train, y_test: Training and testing data
        
    Returns:
        dict: Model and evaluation metrics
    """
    print("Training baseline TF-IDF + Logistic Regression model...")
    
    # Create pipeline
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
        ('clf', LogisticRegression(random_state=42, max_iter=1000))
    ])
    
    # Train model
    pipeline.fit(X_train, y_train)
    
    # Predictions
    y_pred = pipeline.predict(X_test)
    
    # Evaluation metrics
    f1 = f1_score(y_test, y_pred, average='weighted')
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    
    print("Baseline Model Performance:")
    print(f"F1 Score: {f1:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print("\nDetailed Classification Report:")
    print(classification_report(y_test, y_pred))
    
    # Save model
    with open('models/baseline_model.pkl', 'wb') as f:
        pickle.dump(pipeline, f)
    
    return {
        'model': pipeline,
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'predictions': y_pred
    }

class DistilBertClassifier:
    def __init__(self, model_name='distilbert-base-uncased', num_labels=6):
        self.model_name = model_name
        self.num_labels = num_labels
        self.tokenizer = DistilBertTokenizer.from_pretrained(model_name)
        self.model = DistilBertForSequenceClassification.from_pretrained(
            model_name, 
            num_labels=num_labels
        )
        self.label_to_id = {}
        self.id_to_label = {}

    def prepare_labels(self, y_train, y_test):
        """Prepare label mappings"""
        unique_labels = sorted(list(set(list(y_train) + list(y_test))))
        self.label_to_id = {label: i for i, label in enumerate(unique_labels)}
        self.id_to_label = {i: label for label, i in self.label_to_id.items()}
        
        return (
            [self.label_to_id[label] for label in y_train],
            [self.label_to_id[label] for label in y_test]
        )

    def train(self, X_train, y_train, X_test, y_test, output_dir='./results'):
        """Train the DistilBERT model"""
        print("Training DistilBERT model...")
        
        # Prepare labels
        y_train_ids, y_test_ids = self.prepare_labels(y_train, y_test)
        
        # Create datasets
        train_dataset = JobTitleDataset(X_train.tolist(), y_train_ids, self.tokenizer)
        eval_dataset = JobTitleDataset(X_test.tolist(), y_test_ids, self.tokenizer)
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=3,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir='./logs',
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
        )
        
        # Create trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
        )
        
        # Train model
        trainer.train()
        
        # Evaluate model
        eval_result = trainer.evaluate()
        print(f"Evaluation results: {eval_result}")
        
        # Save model
        self.model.save_pretrained('models/distilbert_model')
        self.tokenizer.save_pretrained('models/distilbert_model')
        
        return trainer

    def predict(self, texts):
        """Predict job categories for new texts"""
        predictions = []
        
        for text in texts:
            inputs = self.tokenizer(
                text,
                return_tensors='pt',
                truncation=True,
                padding=True,
                max_length=128
            )
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                predicted_class_id = outputs.logits.argmax().item()
                predictions.append(self.id_to_label[predicted_class_id])
                
        return predictions

def main():
    """
    Main function to run the job classification pipeline
    """
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    # Load data
    df = pd.read_csv('data/data_jobs_clean.csv')
    print(f"Loaded {len(df)} job postings")
    
    # Prepare data
    X_train, X_test, y_train, y_test = prepare_data(df)
    print(f"Training set size: {len(X_train)}")
    print(f"Test set size: {len(X_test)}")
    
    # Train baseline model
    baseline_results = train_baseline_model(X_train, X_test, y_train, y_test)
    
    # Train DistilBERT model (now enabled)
    # Initialize classifier
    classifier = DistilBertClassifier(num_labels=len(set(y_train)))
    
    # Train model
    trainer = classifier.train(X_train, y_train, X_test, y_test)
    
    # Make predictions
    y_pred_distilbert = classifier.predict(X_test.tolist())
    
    # Evaluation metrics
    f1_distilbert = f1_score(y_test, y_pred_distilbert, average='weighted')
    precision_distilbert = precision_score(y_test, y_pred_distilbert, average='weighted')
    recall_distilbert = recall_score(y_test, y_pred_distilbert, average='weighted')
    
    print("DistilBERT Model Performance:")
    print(f"F1 Score: {f1_distilbert:.4f}")
    print(f"Precision: {precision_distilbert:.4f}")
    print(f"Recall: {recall_distilbert:.4f}")
    print("\nDetailed Classification Report:")
    print(classification_report(y_test, y_pred_distilbert))
    
    print("\nJob classification pipeline completed!")
    print(f"Baseline model saved to models/baseline_model.pkl")
    print(f"DistilBERT model saved to models/distilbert_model")

if __name__ == "__main__":
    main()