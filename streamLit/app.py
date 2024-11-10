import streamlit as st
import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel
from transformers import BertTokenizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
label_columns = [
    "ableism", "anti_religion", "harm", "homophobia", "islamophobia", "lookism",
    "political_polarisation", "racism", "religious_intolerance", "sexism", "vulgarity", "xenophobia"
]


class LSTMHateSpeech(nn.Module):
    def __init__(self, num_labels):
        super(LSTMHateSpeech, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.dropout = nn.Dropout(0.3)
        
        # Multiple dense layers with batch normalization
        self.dense1 = nn.Linear(768, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.dense2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.classifier = nn.Linear(256, num_labels)
        
    def forward(self, input_ids, attention_mask, token_type_ids):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        
        pooled_output = outputs.pooler_output
        x = self.dropout(pooled_output)
        
        x = torch.relu(self.bn1(self.dense1(x)))
        x = self.dropout(x)
        x = torch.relu(self.bn2(self.dense2(x)))
        x = self.dropout(x)
        
        return self.classifier(x)

model = LSTMHateSpeech(len(label_columns)).to(device)
model.load_state_dict(torch.load('../model/hate_speech_model.pth'))
model.eval()

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def predict(tweet, model, tokenizer, threshold=0.7):
    model.eval()
    encoded = tokenizer.encode_plus(
        tweet,
        add_special_tokens=True,
        max_length=128,
        return_token_type_ids=True,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors="pt"
    )
    
    input_ids = encoded['input_ids'].to(device)
    attention_mask = encoded['attention_mask'].to(device)
    token_type_ids = encoded['token_type_ids'].to(device)
    
    with torch.no_grad():
        outputs = model(input_ids, attention_mask, token_type_ids)
    
    preds = torch.sigmoid(outputs) >= threshold
    pred_labels = [label_columns[i] for i, pred in enumerate(preds.squeeze()) if pred == 1]

    return pred_labels


####################################################################################################################
# Streamlit app
####################################################################################################################

st.title("Hate Speech Label Predictor")
st.write("Type a tweet below to see if it matches any of the predefined hate speech labels:")

# Text input for the tweet
tweet = st.text_area("Enter your tweet here:", height=150)

# Prediction button
if st.button("Predict Labels"):
    if tweet.strip():
        # Predict labels
        predicted_labels = predict(tweet, model, tokenizer)
        
        # Display results
        if predicted_labels:
            st.write("### Predicted Labels:")
            st.write(", ".join(predicted_labels))
        else:
            st.write("No hate speech labels detected.")
    else:
        st.write("Please enter a tweet to analyze.")
