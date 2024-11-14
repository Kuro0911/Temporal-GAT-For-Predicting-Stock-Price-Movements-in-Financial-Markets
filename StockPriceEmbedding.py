import os
import time
import json
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
from tqdm import tqdm
import torch.optim as optim 

class StockAutoencoder(nn.Module):
    def __init__(self, input_size, hidden_size, embedding_dim, dropout_prob=0.2):
        super(StockAutoencoder, self).__init__()
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


# Function to create sequences
def create_daily_avg_sequences(data, window_size=30):
    """
    Creates sequences of daily average stock data with a specified window size.

    Parameters:
    - data: DataFrame containing daily averaged stock data with columns 'avg_prccd', 'avg_prchd', 'avg_prcld', 'avg_prcod'.
    - window_size: Integer specifying the number of days per sequence (default is 30 days for a full month).

    Returns:
    - sequences: Numpy array where each element is a sequence of `window_size` days with 4 features.
    """
    
    sequences = []
    num_records = len(data)
    
    # Loop to generate sequences with the specified window size
    for i in range(num_records - window_size + 1):
        # Extract a window of data with `window_size` days
        window_data = data.iloc[i:i + window_size][['avg_prccd', 'avg_prchd', 'avg_prcld', 'avg_prcod']].values
        sequences.append(window_data)  # Add this sequence to the list of sequences
    
    # Convert the list of sequences to a numpy array for easier handling later
    return np.array(sequences)


def get_daily_avg_data(full_df):
    """
    Prepares daily average stock data by aggregating over year, month, and day.

    Parameters:
    - full_df: DataFrame containing stock data with columns 'datadate', 'prccd', 'prchd', 'prcld', 'prcod'.

    Returns:
    - daily_avg_df: DataFrame with daily average values for each feature ('avg_prccd', 'avg_prchd', 'avg_prcld', 'avg_prcod').
    """
    
    # Create a copy of full_df to avoid SettingWithCopyWarning
    full_df = full_df.copy()
    
    # Ensure 'datadate' column is in datetime format
    if not pd.api.types.is_datetime64_any_dtype(full_df['datadate']):
        full_df['datadate'] = pd.to_datetime(full_df['datadate'])
    
    # Add year, month, and day columns for easier grouping
    full_df['year'] = full_df['datadate'].dt.year
    full_df['month'] = full_df['datadate'].dt.month
    full_df['day'] = full_df['datadate'].dt.day
    
    # Group by year, month, and day to calculate daily averages for each feature
    daily_avg_df = full_df.groupby(['year', 'month', 'day'], as_index=False)[
        ['prccd', 'prchd', 'prcld', 'prcod']
    ].mean()
    
    # Rename the columns to indicate these are daily averages
    daily_avg_df.columns = ['year', 'month', 'day', 'avg_prccd', 'avg_prchd', 'avg_prcld', 'avg_prcod']
    
    # Handle any missing values in 'avg_prcod' by filling with the column's mean
    daily_avg_df['avg_prcod'] = daily_avg_df['avg_prcod'].fillna(daily_avg_df['avg_prcod'].mean())
    
    return daily_avg_df


# Function to train the model and get embeddings

def get_embedding_model(full_df, patience=5):
    """
    Train an autoencoder on daily average stock data to generate embeddings.

    Parameters:
    - full_df: DataFrame containing daily stock data.

    Returns:
    - autoencoder: Trained autoencoder model.
    """
    
    # Set device to GPU if available, else use CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Prepare the daily average data
    daily_avg_df = get_daily_avg_data(full_df)  # Assumes existence of function to calculate daily averages
    daily_sequences = create_daily_avg_sequences(daily_avg_df, window_size=30)  # 30-day window
    
    # Flatten each sequence to make it compatible with the model input
    flattened_sequences = [seq.flatten() for seq in daily_sequences]
    flattened_sequences_tensor = torch.tensor(flattened_sequences, dtype=torch.float32)
    print(f"Total daily sequences created: {len(daily_sequences)}")
    print(flattened_sequences_tensor.shape)

    # Create a DataLoader for batch processing during training
    batch_size = 32
    dataset = TensorDataset(flattened_sequences_tensor)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Define model parameters
    sequence_length = 30  # 30 days in each sequence
    input_features = 4  # Features per day (e.g., 'avg_prccd', 'avg_prchd', 'avg_prcld', 'avg_prcod')
    input_size = sequence_length * input_features  # Total input size for each flattened sequence
    hidden_size = 256  # Size of hidden layer
    embedding_dim = 64  # Size of the embedding layer
    num_epochs = 100  # Number of training epochs
    learning_rate = 0.0005  # Initial learning rate
    
    # Initialize the autoencoder model and move it to the specified device
    autoencoder = StockAutoencoder(input_size, hidden_size, embedding_dim).to(device)
    
    # Define loss function and optimizer
    criterion = nn.MSELoss()  # Mean Squared Error for reconstruction loss
    optimizer = optim.Adam(autoencoder.parameters(), lr=learning_rate)
    
    # Learning rate scheduler to adjust the learning rate dynamically
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=learning_rate * 10, 
        steps_per_epoch=len(train_loader), epochs=num_epochs
    )

    # Training loop
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




def generate_embeddings_dict(autoencoder, data, year_ranges, batch_size=128, device=None, base_path="."):
    """
    Generate embeddings dictionary for each company's 5-year period.
    
    Parameters:
    - autoencoder: Pre-trained autoencoder model.
    - data: DataFrame containing stock data with columns 'tic', 'datadate', 'prccd', 'prchd', 'prcld', 'prcod'.
    - year_ranges: Dictionary defining 5-year ranges, e.g., {"2000": (2000, 2004), "2005": (2005, 2009), ...}.
    - batch_size: Batch size for GPU processing.
    - device: Device for computation (e.g., "cuda" or "cpu"). If None, auto-detects GPU if available.
    - base_path: Base path for loading additional files like `sector_node_to_tic.json`.
    
    Returns:
    - embeddings_dict: Dictionary with embeddings for each period and ticker.
    """
    
    # Set device if not provided
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    autoencoder = autoencoder.to(device)
    autoencoder.eval()

    # Initialize dictionary for embeddings
    embeddings_dict = {period: {} for period in year_ranges.keys()}

    # Loop through each period and generate embeddings
    for foldername, (start_year, end_year) in year_ranges.items():
        print(f"Processing period: {start_year}-{end_year}")
        
        # Load node_to_tic.json for the current period
        node_to_tic_path = os.path.join(base_path, 'data', foldername, "sector_node_to_tic.json")
        with open(node_to_tic_path, 'r') as f:
            node_to_tic = json.load(f)
        
        # Define nodes_set from node_to_tic values (tickers)
        # nodes_set = set(node_to_tic.values()) for now we will use ony 100 please
        nodes_set = set(list(node_to_tic.values())[:100])
        
        
        # Calculate embeddings with tqdm for progress tracking
        for ticker in tqdm(nodes_set, desc=f"Processing Companies for {foldername}"):
            # Filter data for the specific ticker and year range
            start_date = f"{start_year}-01-01"
            end_date = f"{end_year}-12-31"
            stock_df = data[(data['tic'] == ticker) & 
                            (data['datadate'] >= start_date) & 
                            (data['datadate'] <= end_date)].sort_values(by='datadate')
            
            if not stock_df.empty:
                stock_data = stock_df[['prccd', 'prchd', 'prcld', 'prcod']].values
                yearly_sequences = []

                # Split data into yearly sequences of 30 days
                for i in range(0, len(stock_data), 30):
                    if i + 30 <= len(stock_data):  # Ensure we have a full year
                        yearly_sequence = stock_data[i:i + 30].flatten()
                        yearly_sequences.append(yearly_sequence)
                
                # Batch process the yearly sequences
                if yearly_sequences:
                    yearly_sequences_tensor = torch.tensor(yearly_sequences, dtype=torch.float32).to(device)
                    
                    # Process sequences in batches
                    yearly_embeddings = []
                    for i in range(0, len(yearly_sequences_tensor), batch_size):
                        batch_tensor = yearly_sequences_tensor[i:i + batch_size]
                        with torch.no_grad():
                            batch_embeddings = autoencoder.encoder(batch_tensor).cpu().numpy()
                        yearly_embeddings.extend(batch_embeddings)
                    
                    # Calculate the mean embedding for the 5-year period
                    five_year_embedding = np.mean(yearly_embeddings, axis=0)
                    embeddings_dict[foldername][ticker] = five_year_embedding

    return embeddings_dict
