import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer

class HateDetectionTransformer(nn.Module):
    def __init__(self, model_name='bert-base-multilingual-uncased'):
        super().__init__()
        self.transformer = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.transformer.config.hidden_size, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, input_ids, attention_mask):
        outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return self.sigmoid(logits)

class TransformerPredictor:
    def __init__(self, model_path=None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = HateDetectionTransformer().to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-multilingual-uncased')
        
        if model_path:
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        
        self.model.eval()
    
    def predict(self, text):
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=128,
            padding='max_length',
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)
        
        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask)
            probability = outputs.item()
            
        return {
            'is_toxic': probability > 0.5,
            'probability': probability
        }