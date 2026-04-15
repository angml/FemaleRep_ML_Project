import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

df = pd.read_csv("TMDB_female_representation.csv") 
unneccessary_cols = ['tmdb_id', 'title', 'release_date', 'decade', 'overview', 'original_language', 'genre_names', 'primary_country', 
                     'primary_company', 'language_count', 'lead_gender', 'lead_actor_name', 'lead_actor_age', 'top5_female', 'top5_male', 
                     'pct_female_top5', 'cast_female', 'cast_male', 'cast_total_known','pct_female_cast', 'director_gender', 'director_name', 
                     'director_age', 'num_directors', 'pct_female_writers', 'pct_female_producers','female_writers', 'female_producers', 'vote_count']
clean_df = df.drop(columns = unneccessary_cols)
X = clean_df.drop(columns = ["representation_tier"])
y = clean_df["representation_tier"]

le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Log transform skewed features
X['budget'] = np.log1p(X['budget'])
X['revenue'] = np.log1p(X['revenue'])
X['popularity'] = np.log1p(X['popularity'])

# Standardize numeric features
scaler = StandardScaler()
X[['popularity', 'budget', 'revenue', 'runtime']] = scaler.fit_transform(X[['popularity', 'budget', 'revenue', 'runtime']])

X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.3, random_state=42)

# Convert to PyTorch tensors
X_train = torch.tensor(X_train.values, dtype=torch.float32)
X_test  = torch.tensor(X_test.values,  dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long) 
y_test  = torch.tensor(y_test,  dtype=torch.long)

train_dataset = TensorDataset(X_train, y_train)
train_loader  = DataLoader(train_dataset, batch_size=64, shuffle=True) #batching-how much data the model sees at once

class MLP(nn.Module):
    def __init__(self, input_size, num_classes):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 64),  
            nn.ReLU(),
            nn.Linear(64, num_classes) 
        )

    def forward(self, x):
        return self.network(x)

input_size  = X_train.shape[1]
num_classes = len(np.unique(y))
model = MLP(input_size, num_classes)

# Loss function 
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.003) #Updates model weights to reduce loss

# Training loop
epochs = 100
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()         # clear old gradients
        outputs = model(X_batch)      # make predictions
        loss = criterion(outputs, y_batch) # compute loss
        loss.backward()               # compute gradient/how to adjust weight
        optimizer.step()              # update weights
        total_loss += loss.item()

    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}/{epochs} — Loss: {total_loss/len(train_loader):.4f}")

# Evaluate
model.eval()
with torch.no_grad():
    outputs = model(X_test)
    predictions = torch.argmax(outputs, dim=1)
    accuracy = (predictions == y_test).float().mean()
    print(f"Test accuracy: {accuracy:.4f}")
