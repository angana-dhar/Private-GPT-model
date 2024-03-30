#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import requests
from bs4 import BeautifulSoup

# URL of the list
url = 'https://hospitals.webometrics.info/en/world'

# Send a GET request to the URL
response = requests.get(url)

# Parse the HTML content
soup = BeautifulSoup(response.text, 'html.parser')

# Find the top hospitals
table = soup.find('table', class_='sticky-enabled')

# Extract relevant information
hospital_data = []
rows = table.find_all('tr')[1:]  # Exclude header row
for row in rows:
    cells = row.find_all('td')
    rank = cells[0].text.strip()
    name = cells[1].text.strip()
    country = cells[2].text.strip()
    size = cells[3].text.strip()
    hospital_data.append({'Rank': rank, 'Name': name, 'Country': country, 'Size': size})

# Print the scraped data
for hospital in hospital_data:
    print(hospital)


# In[10]:


# Assuming hospital_data contains the scraped data
cleaned_hospital_data = []

for hospital in hospital_data:
    # Remove leading/trailing whitespace and normalize the name
    name = hospital['Name'].strip().title()  # Convert to title case
    
    # Remove any unnecessary characters or whitespace from other fields
    rank = hospital['Rank']
    country = hospital['Country']
    size = hospital['Size']
    
    # Add the cleaned data to the list
    cleaned_hospital_data.append({'Rank': rank, 'Name': name, 'Country': country, 'Size': size})


# In[11]:


# Print the cleaned hospital data
for hospital in cleaned_hospital_data:
    print(hospital)


# In[12]:


# Print the first 5 cleaned hospital data entries
for hospital in cleaned_hospital_data[:5]:
    print(hospital)


# In[13]:


from transformers import GPT2Tokenizer, GPT2LMHeadModel
from torch.utils.data import Dataset, DataLoader
import torch

# Convert data into text format suitable for training the GPT model
training_text = "\n".join([hospital['Name'] for hospital in cleaned_hospital_data])

# Tokenize the text data
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenized_text = tokenizer(training_text, return_tensors='pt', truncation=True, padding=True)

# Define a custom dataset
class HospitalDataset(Dataset):
    def __init__(self, tokenized_text):
        self.input_ids = tokenized_text['input_ids']
        self.attention_mask = tokenized_text['attention_mask']
    
    def __len__(self):
        return len(self.input_ids)
    
    def __getitem__(self, idx):
        return {'input_ids': self.input_ids[idx], 'attention_mask': self.attention_mask[idx]}

# Create an instance of the custom dataset
dataset = HospitalDataset(tokenized_text)

# Initialize the GPT model
model = GPT2LMHeadModel.from_pretrained('gpt2')

# Fine-tune the model
train_loader = DataLoader(dataset, batch_size=8, shuffle=True)

optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
criterion = torch.nn.CrossEntropyLoss()

for epoch in range(3):  # Example: Train for 3 epochs
    model.train()
    for batch in train_loader:
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
        loss = outputs.loss
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# Save the fine-tuned model
model.save_pretrained('./fine_tuned_gpt_model')


# In[14]:


from transformers import GPT2Tokenizer, GPT2LMHeadModel
from torch.utils.data import Dataset, DataLoader
import torch

# Convert data into text format suitable for training the GPT model
training_text = "\n".join([hospital['Name'] for hospital in cleaned_hospital_data])

# Tokenize the text data
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token  # Set end-of-sequence token as padding token
tokenized_text = tokenizer(training_text, return_tensors='pt', truncation=True, padding=True)

# Define a custom dataset
class HospitalDataset(Dataset):
    def __init__(self, tokenized_text):
        self.input_ids = tokenized_text['input_ids']
        self.attention_mask = tokenized_text['attention_mask']
    
    def __len__(self):
        return len(self.input_ids)
    
    def __getitem__(self, idx):
        return {'input_ids': self.input_ids[idx], 'attention_mask': self.attention_mask[idx]}

# Create an instance of the custom dataset
dataset = HospitalDataset(tokenized_text)

# Initialize the GPT model
model = GPT2LMHeadModel.from_pretrained('gpt2')

# Fine-tune the model
train_loader = DataLoader(dataset, batch_size=8, shuffle=True)

optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
criterion = torch.nn.CrossEntropyLoss()

for epoch in range(3):  # Example: Train for 3 epochs
    model.train()
    for batch in train_loader:
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
        loss = outputs.loss
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# Save the fine-tuned model
model.save_pretrained('./fine_tuned_gpt_model')


# In[21]:


from torch.utils.data import Dataset

class ValidationDataset(Dataset):
    def __init__(self, data):
        self.data = data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        return {'input_ids': sample['input_ids'], 'attention_mask': sample['attention_mask'], 'labels': sample['labels']}

# Assuming you have validation data in the same format as training data
validation_data = [
    {'input_ids': [1, 2, 3, 4, 5], 'attention_mask': [1, 1, 1, 1, 1], 'labels': [2, 3, 4, 5, 6]},
    {'input_ids': [2, 3, 4, 5, 6], 'attention_mask': [1, 1, 1, 1, 1], 'labels': [3, 4, 5, 6, 7]},
    {'input_ids': [1, 3, 4, 2, 6], 'attention_mask': [1, 1, 1, 1, 1], 'labels': [1, 3, 5, 6, 7]},
    {'input_ids': [1, 5, 3, 4, 6], 'attention_mask': [1, 1, 1, 1, 1], 'labels': [3, 2, 4, 7, 6]},
    {'input_ids': [2, 3, 4, 1, 6], 'attention_mask': [1, 1, 1, 1, 1], 'labels': [3, 4, 2, 6, 7]},
    {'input_ids': [2, 6, 4, 5, 5], 'attention_mask': [1, 1, 1, 1, 1], 'labels': [3, 4, 5, 6, 6]}
    
]

# Create an instance of the validation dataset
validation_dataset = ValidationDataset(validation_data)


# In[29]:


# Assuming you have a validation dataset called validation_dataset
validation_loader = DataLoader(validation_dataset, batch_size=8, shuffle=False)

# Evaluate the model
model.eval()
total_loss = 0
total_batches = 0

with torch.no_grad():
    for batch in validation_loader:
        input_ids = batch['input_ids']
       


# In[30]:


print(model)


# In[31]:


for name, param in model.named_parameters():
    print(f'Parameter name: {name}, Shape: {param.shape}')


# In[ ]:


import time  # Add this line to import the time module

# Monitoring
def monitor_performance():
    # Collect and log performance metrics
    ...

def alert_on_thresholds():
    # Check metrics against predefined thresholds
    # Send alerts if thresholds are exceeded
    ...

# Model updating
def collect_new_data():
    # Collect new data from sources
    ...

def retrain_model():
    # Retrain the model using new data
    ...

def evaluate_updated_model():
    # Evaluate the performance of the updated model
    ...

# Maintenance
def track_issues():
    # Track and prioritize issues
    ...

def perform_regular_maintenance():
    # Schedule regular maintenance tasks
    ...

# Main loop
while True:
    # Monitoring
    monitor_performance()
    alert_on_thresholds()
    
    # Model updating
    collect_new_data()
    retrain_model()
    evaluate_updated_model()
    
    # Maintenance
    track_issues()
    perform_regular_maintenance()
    
    # Sleep for a defined interval before the next iteration
    time.sleep(3600)  # Sleep for 1 hour before the next iteration


# In[ ]:


print(perform_regular_maintenance()
      


# In[ ]:


#not responding as it is in sleep mode for 1 hour before the next iteration

