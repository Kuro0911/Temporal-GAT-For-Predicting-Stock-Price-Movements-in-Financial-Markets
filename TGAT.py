import torch
import torch.nn as nn
from torch_geometric_temporal.nn.recurrent import A3TGCN
import torch_geometric_temporal.signal as tx
import numpy as np
import pandas as pd

class TemporalGAT_RNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, tgat_out_dim, rnn_hidden_dim, output_dim):
        super(TemporalGAT_RNN, self).__init__()
        
        # TGAT Layer for both price and sentiment embeddings
        self.tgat_price = A3TGCN(in_channels=input_dim, out_channels=tgat_out_dim, periods=5)
        self.tgat_sentiment = A3TGCN(in_channels=input_dim, out_channels=tgat_out_dim, periods=5)
        
        # Linear layer to reduce Kronecker product dimensionality
        self.reduce_dim = nn.Linear(tgat_out_dim * tgat_out_dim, hidden_dim)  # Example reduction to 'hidden_dim'

        # RNN layer to process sequential outputs from TGAT layers
        self.rnn = nn.LSTM(input_size=hidden_dim, hidden_size=rnn_hidden_dim, batch_first=True)
        
        # Fully connected layers for final prediction
        self.fc1 = nn.Linear(rnn_hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x_price, x_sentiment, edge_index, edge_attr, temporal_features):
        # TGAT layer for price and sentiment embeddings
        price_tgat_out = self.tgat_price(x_price, edge_index, edge_attr, temporal_features)
        sentiment_tgat_out = self.tgat_sentiment(x_sentiment, edge_index, edge_attr, temporal_features)
        
        # Kronecker product for each node (resulting in a [num_nodes, tgat_out_dim * tgat_out_dim] matrix)
        combined_tgat_out = []
        for i in range(price_tgat_out.size(0)):  # Iterate over nodes
            kronecker_product = torch.kron(price_tgat_out[i], sentiment_tgat_out[i]).view(-1)
            reduced_product = self.reduce_dim(kronecker_product)  # Reduce dimensionality
            combined_tgat_out.append(reduced_product)
        
        combined_tgat_out = torch.stack(combined_tgat_out)  # Shape: [num_nodes, hidden_dim]

        # Reshape for RNN: [batch_size, seq_len, feature_dim]
        combined_tgat_out = combined_tgat_out.view(1, -1, combined_tgat_out.size(1))
        
        # RNN layer
        rnn_out, _ = self.rnn(combined_tgat_out)
        rnn_out = rnn_out[:, -1, :]  # Take the last hidden state for prediction

        # Fully connected layers for final prediction
        hidden = self.fc1(rnn_out)
        hidden = self.relu(hidden)
        output = self.fc2(hidden)
        
        return output

def get_tgat_model(input_dim, hidden_dim, tgat_out_dim, rnn_hidden_dim, output_dim, device):
    """Initialize the TemporalGAT_RNN model."""
    model = TemporalGAT_RNN(input_dim, hidden_dim, tgat_out_dim, rnn_hidden_dim, output_dim).to(device)
    return model


def prepare_embeddings(embeddings_df, sentiment_embeddings_df, period):
    """Prepare embeddings for price and sentiment as input tensors for a specified period."""
    
    x_price_df = embeddings_df[embeddings_df['period'] == period].drop(columns=['period', 'ticker'])
    x_sentiment_df = sentiment_embeddings_df[sentiment_embeddings_df['period'] == period].drop(columns=['period', 'ticker'])
    
    x_price = torch.tensor(x_price_df.values, dtype=torch.float32)
    x_sentiment = torch.tensor(x_sentiment_df.values, dtype=torch.float32)
    
    return x_price, x_sentiment


def get_graph_data(TG, period):
    """Retrieve edge connections and features from TemporalDiGraph for a specified period."""
    
    edge_index = []
    edge_attr = []

    # Convert NodeView to a list and unpack it if necessary
    node_list = list(TG.nodes()[0])

    # Create a mapping from nodes to integer indices
    node_mapping = {node: idx for idx, node in enumerate(node_list)}

    # Convert OutEdgeDataView to a list of edges with data
    edges = list(TG.edges(data=True)[0])

    # Iterate through each edge to process based on period
    for u, v, data in edges:
        # Check if the edge data contains the 'time' attribute and matches the specified period
        if data.get('time') == period:
            # Convert node labels to integer indices and add to edge_index
            edge_index.append([node_mapping[u], node_mapping[v]])
            # Append edge weight if it exists; otherwise, use a default weight of 1.0
            edge_attr.append(data.get('weight', 1.0))  # Use data['weight'] if available, else 1.0

    # Convert to tensors
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attr, dtype=torch.float32) if edge_attr else None
    
    return edge_index, edge_attr 


def train_tgat_model(model, train_periods, embeddings_df, sentiment_embeddings_df, TG, y_targets, device, num_epochs=50, learning_rate=0.001):
    """Train the TGAT model."""
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    model.train()

    for epoch in range(num_epochs):
        total_loss = 0
        for period in train_periods:
            # Prepare node features and graph data
            x_price, x_sentiment = prepare_embeddings(embeddings_df, sentiment_embeddings_df, period)
            edge_index, edge_attr = get_graph_data(TG, period)
            
            # Move data to device
            x_price, x_sentiment = x_price.to(device), x_sentiment.to(device)
            edge_index, edge_attr = edge_index.to(device), edge_attr.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            y_pred = model(x_price, x_sentiment, edge_index, edge_attr)
            
            # Define target stock price movement for training period
            y_true = torch.tensor(y_targets[period], dtype=torch.float32).to(device)
            loss = criterion(y_pred, y_true)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(train_periods):.4f}")
        return model


def test_tgat_model(model, test_period, embeddings_df, sentiment_embeddings_df, TG, y_target, device):
    """Evaluate the TGAT model on the test period."""
    model.eval()
    with torch.no_grad():
        # Prepare node features and graph data for the test period
        x_price, x_sentiment = prepare_embeddings(embeddings_df, sentiment_embeddings_df, test_period)
        edge_index, edge_attr = get_graph_data(TG, test_period)
        
        x_price, x_sentiment = x_price.to(device), x_sentiment.to(device)
        edge_index, edge_attr = edge_index.to(device), edge_attr.to(device)
        
        # Predict on the test period
        y_test_pred = model(x_price, x_sentiment, edge_index, edge_attr)
        y_true = torch.tensor(y_target, dtype=torch.float32).to(device)
        
        # Calculate test loss or accuracy
        criterion = nn.MSELoss()
        test_loss = criterion(y_test_pred, y_true).item()
        print(f"Test Loss: {test_loss:.4f}")
        
        return y_test_pred, test_loss