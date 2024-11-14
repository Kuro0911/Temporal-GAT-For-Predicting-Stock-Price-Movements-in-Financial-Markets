import os
import json
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
from tqdm import tqdm
import torch.optim as optim

# Autoencoder for Sentiment Data Embedding
class SentimentEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, embedding_dim, dropout_prob=0.2):
        super(SentimentEncoder, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(hidden_size, embedding_dim),
            nn.ReLU()
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(embedding_dim, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(hidden_size, input_size)
        )

    def forward(self, x):
        embedding = self.encoder(x)
        reconstructed = self.decoder(embedding)
        return embedding, reconstructed

# Function to create sequences of monthly average sentiment data
def create_monthly_sequences(data, window_size=12):
    sequences = []
    num_records = len(data)
    
    for i in range(num_records - window_size + 1):
        window_data = data.iloc[i:i + window_size][['avg_sentiment']].values
        sequences.append(window_data)
    
    return np.array(sequences)

# Function to train the model and get embeddings
def get_sentiment_embedding_model(sentiment_df, patience=5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    monthly_sequences = create_monthly_sequences(sentiment_df, window_size=12)
    
    flattened_sequences = [seq.flatten() for seq in monthly_sequences]
    flattened_sequences_tensor = torch.tensor(flattened_sequences, dtype=torch.float32)
    print(f"Total monthly sequences created: {len(monthly_sequences)}")
    print(flattened_sequences_tensor.shape)

    batch_size = 32
    dataset = TensorDataset(flattened_sequences_tensor)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    sequence_length = 12  # 12 months in each sequence
    input_features = 1  # Only 'avg_sentiment' feature per month
    input_size = sequence_length * input_features
    hidden_size = 64
    embedding_dim = 32
    num_epochs = 50
    learning_rate = 0.0005
    
    autoencoder = SentimentEncoder(input_size, hidden_size, embedding_dim).to(device)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(autoencoder.parameters(), lr=learning_rate)
    
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=learning_rate * 10, 
        steps_per_epoch=len(train_loader), epochs=num_epochs
    )

    # Early stopping variables
    best_loss = float('inf')
    epochs_without_improvement = 0
    
    autoencoder.train()
    for epoch in range(num_epochs):
        epoch_loss = 0
        with tqdm(train_loader, desc=f"Epoch [{epoch+1}/{num_epochs}]", leave=False) as progress_bar:
            for batch in progress_bar:
                batch = batch[0].to(device)
                optimizer.zero_grad()
                
                embeddings, reconstructions = autoencoder(batch)
                loss = criterion(reconstructions, batch)
                
                loss.backward()
                optimizer.step()
                scheduler.step()
                
                epoch_loss += loss.item()
                progress_bar.set_postfix(loss=loss.item())
        
        average_epoch_loss = epoch_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Average Loss: {average_epoch_loss:.4f}")
        
        # Early stopping check
        if average_epoch_loss < best_loss:
            best_loss = average_epoch_loss
            epochs_without_improvement = 0  # Reset if we improve
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"Stopping early at epoch {epoch+1} due to no improvement in loss for {patience} consecutive epochs.")
                break

    return autoencoder

# Function to generate embeddings dictionary for each company's 5-year period
def generate_sentiment_embeddings_dict(autoencoder, sentiment_df, year_ranges, batch_size=128, device=None, base_path="."):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    autoencoder = autoencoder.to(device)
    autoencoder.eval()

    embeddings_dict = {period: {} for period in year_ranges.keys()}

    for foldername, (start_year, end_year) in year_ranges.items():
        print(f"Processing period: {start_year}-{end_year}")
        
        node_to_tic_path = os.path.join(base_path, 'data', foldername, "sector_node_to_tic.json")
        with open(node_to_tic_path, 'r') as f:
            node_to_tic = json.load(f)
        
        nodes_set = set(list(node_to_tic.values())[:100])
        
        for ticker in tqdm(nodes_set, desc=f"Processing Companies for {foldername}"):
            period_data = sentiment_df[(sentiment_df['Ticker'] == ticker) & 
                                       (sentiment_df['Year'] >= start_year) & 
                                       (sentiment_df['Year'] <= end_year)]
            
            if not period_data.empty:
                period_sequences = create_monthly_sequences(period_data, window_size=12)
                
                if period_sequences.size > 0:
                    period_sequences_tensor = torch.tensor([seq.flatten() for seq in period_sequences], dtype=torch.float32).to(device)
                    
                    embeddings = []
                    for i in range(0, len(period_sequences_tensor), batch_size):
                        batch_tensor = period_sequences_tensor[i:i + batch_size]
                        with torch.no_grad():
                            batch_embeddings = autoencoder.encoder(batch_tensor).cpu().numpy()
                        embeddings.extend(batch_embeddings)
                    
                    five_year_embedding = np.mean(embeddings, axis=0)
                    embeddings_dict[foldername][ticker] = five_year_embedding

    return embeddings_dict