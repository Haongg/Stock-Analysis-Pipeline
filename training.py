import copy
import json
import pickle

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.preprocessing import RobustScaler
from torch import nn
from torch.utils.data import DataLoader, Dataset


LOOKBACK = 90
BATCH_SIZE = 32
EPOCHS = 100
LEARNING_RATE = 5e-4
PATIENCE = 15
SEED = 42
WEIGHT_DECAY = 1e-5



def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(SEED)


class SequenceDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class ImprovedLSTMRegressor(nn.Module):
    """Enhanced LSTM with bidirectional architecture and batch normalization."""
    def __init__(
        self, 
        n_features: int, 
        hidden_size: int = 128, 
        num_layers: int = 3, 
        dropout: float = 0.3
    ):
        super().__init__()
        
        # Bidirectional LSTM for better temporal context
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        # Layer normalization (better for small batches than BatchNorm)
        self.ln = nn.LayerNorm(hidden_size * 2)
        
        # Multi-layer fully connected network
        self.fc1 = nn.Linear(hidden_size * 2, 64)
        self.relu = nn.ReLU()
        self.dropout_fc = nn.Dropout(dropout)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        # LSTM forward pass
        output, _ = self.lstm(x)
        
        # Take the last timestep from bidirectional output
        last_output = output[:, -1, :]
        
        # Apply layer normalization        last_output = self.ln(last_output)
        
        # Multi-layer FC with dropout
        x = self.relu(self.fc1(last_output))
        x = self.dropout_fc(x)
        x = self.fc2(x)
        
        return x.squeeze(-1)


def load_data(path: str) -> pd.DataFrame:
    data = pd.read_csv(path)
    data["Date"] = pd.to_datetime(data["Date"], utc=True, errors="coerce")
    data = data.dropna(subset=["Date", "Close"]).sort_values("Date").reset_index(drop=True)
    return data


def split_time_series(data: pd.DataFrame, train_ratio=0.7, valid_ratio=0.15):
    n_samples = len(data)
    train_end = int(n_samples * train_ratio)
    valid_end = int(n_samples * (train_ratio + valid_ratio))
    return data.iloc[:train_end], data.iloc[train_end:valid_end], data.iloc[valid_end:]


def create_sequences(features: np.ndarray, target: np.ndarray, lookback: int):
    X, y = [], []
    for i in range(lookback, len(features)):
        X.append(features[i - lookback:i])
        y.append(target[i])
    return np.array(X), np.array(y)


def get_all_features():
    """Return all available feature columns for the dataset."""
    return [
        "Open", "High", "Low", "Close", "Volume",
        "SMA10", "SMA20", "SMA50", "EMA20", "RSI_14",
        "MACD", "Signal_Line", "Histogram", "Daily_Return",
        "Rolling_Volatility", "Close_lag_1", "Close_lag_5"
    ]


def build_sequences(
    frame: pd.DataFrame,
    x_scaler: RobustScaler,
    y_scaler: RobustScaler,
    lookback: int,
    context_df: pd.DataFrame | None = None,
):
    feature_cols = get_all_features()
    target_col = "Close"

    if context_df is not None:
        combined = pd.concat([context_df.tail(lookback), frame], ignore_index=True)
    else:
        combined = frame.copy()

    if len(combined) <= lookback:
        raise ValueError("Not enough rows to build sequences with the configured lookback window.")

    X_scaled = x_scaler.transform(combined[feature_cols].values)
    y_scaled = y_scaler.transform(combined[[target_col]].values).flatten()
    return create_sequences(X_scaled, y_scaled, lookback)


def prepare_splits(train_df, valid_df, test_df, lookback: int):
    feature_cols = get_all_features()
    target_col = "Close"

    # Use RobustScaler for better handling of outliers in financial data
    x_scaler = RobustScaler()
    y_scaler = RobustScaler()

    x_scaler.fit(train_df[feature_cols].values)
    y_scaler.fit(train_df[[target_col]].values)

    X_train_seq, y_train_seq = build_sequences(train_df, x_scaler, y_scaler, lookback)
    X_valid_seq, y_valid_seq = build_sequences(valid_df, x_scaler, y_scaler, lookback, context_df=train_df)
    X_test_seq, y_test_seq = build_sequences(
        test_df,
        x_scaler,
        y_scaler,
        lookback,
        context_df=pd.concat([train_df, valid_df]),
    )

    return (X_train_seq, y_train_seq), (X_valid_seq, y_valid_seq), (X_test_seq, y_test_seq), x_scaler, y_scaler


def train_model(model, train_loader, valid_loader, device):
    # Use SmoothL1Loss for robustness
    criterion = nn.SmoothL1Loss(beta=1.0)
    
    # AdamW optimizer with weight decay for L2 regularization
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )
    
    # CosineAnnealingWarmRestarts for better learning rate scheduling
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=10,
        T_mult=2,
        eta_min=1e-6
    )
    
    best_state = copy.deepcopy(model.state_dict())
    best_valid_loss = float("inf")
    patience_counter = 0
    history = {"train_loss": [], "valid_loss": [], "learning_rate": []}

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_losses = []
        
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            preds = model(X_batch)
            loss = criterion(preds, y_batch)
            loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        valid_losses = []
        with torch.no_grad():
            for X_batch, y_batch in valid_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                preds = model(X_batch)
                valid_losses.append(criterion(preds, y_batch).item())

        avg_train_loss = float(np.mean(train_losses))
        avg_valid_loss = float(np.mean(valid_losses))
        current_lr = optimizer.param_groups[0]['lr']
        
        history["train_loss"].append(avg_train_loss)
        history["valid_loss"].append(avg_valid_loss)
        history["learning_rate"].append(current_lr)
        
        scheduler.step()

        if avg_valid_loss < best_valid_loss:
            best_valid_loss = avg_valid_loss
            best_state = copy.deepcopy(model.state_dict())
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            patience_counter = 0
            print(
                f"Epoch {epoch:03d}/{EPOCHS} | "
                f"Train Loss: {avg_train_loss:.6f} | "
                f"Valid Loss: {avg_valid_loss:.6f} | "
                f"LR: {current_lr:.2e} | "
                f"✓ Model saved"
            )
        else:
            patience_counter += 1
            print(
                f"Epoch {epoch:03d}/{EPOCHS} | "
                f"Train Loss: {avg_train_loss:.6f} | "
                f"Valid Loss: {avg_valid_loss:.6f} | "
                f"LR: {current_lr:.2e}"
            )

        if patience_counter >= PATIENCE:
            print(f"\nEarly stopping triggered after {epoch} epochs.")
            break

    model.load_state_dict(best_state)
    return history


def evaluate(model, test_loader, y_scaler, device):
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            preds = model(X_batch).cpu().numpy()
            y_pred.extend(preds)
            y_true.extend(y_batch.numpy())

    y_true = np.array(y_true).reshape(-1, 1)
    y_pred = np.array(y_pred).reshape(-1, 1)

    y_true_inv = y_scaler.inverse_transform(y_true).flatten()
    y_pred_inv = y_scaler.inverse_transform(y_pred).flatten()

    mae = mean_absolute_error(y_true_inv, y_pred_inv)
    rmse = root_mean_squared_error(y_true_inv, y_pred_inv)
    denominator = np.clip(np.abs(y_true_inv), 1e-8, None)
    mape = np.mean(np.abs((y_true_inv - y_pred_inv) / denominator)) * 100
    r2 = r2_score(y_true_inv, y_pred_inv)
    
    print(f"\n{'='*60}")
    print("TEST METRICS")
    print(f"{'='*60}")
    print(f"MAE:  ${mae:.4f}")
    print(f"RMSE: ${rmse:.4f}")
    print(f"MAPE: {mape:.2f}%")
    print(f"R²:   {r2:.4f}")
    print(f"{'='*60}\n")
    
    return {"mae": mae, "rmse": rmse, "mape": mape, "r2": r2}


def save_model_package(model, x_scaler, y_scaler, metrics, model_dir: str = r"E:\hc\BIG DATA\btl"):
    """Save model weights, scalers, metrics, and config."""
    import os
    os.makedirs(model_dir, exist_ok=True)
    
    # Save model weights
    model_weights_path = os.path.join(model_dir, "lstm_model_weights_optimized.pt")
    torch.save(model.state_dict(), model_weights_path)
    print(f"✓ Model weights saved: {model_weights_path}")
    
    # Save x_scaler (needed for preprocessing new data)
    x_scaler_path = os.path.join(model_dir, "x_scaler_optimized.pkl")
    with open(x_scaler_path, "wb") as f:
        pickle.dump(x_scaler, f)
    print(f"✓ X scaler saved: {x_scaler_path}")
    
    # Save y_scaler
    y_scaler_path = os.path.join(model_dir, "y_scaler_optimized.pkl")
    with open(y_scaler_path, "wb") as f:
        pickle.dump(y_scaler, f)
    print(f"✓ Y scaler saved: {y_scaler_path}")
    
    # Save model config and metrics
    metrics_serializable = {k: float(v) for k, v in metrics.items()}
    config = {
        "model_type": "ImprovedLSTMRegressor",
        "n_features": len(get_all_features()),
        "feature_columns": get_all_features(),
        "hidden_size": 128,
        "num_layers": 3,
        "dropout_rate": 0.3,
        "lookback": LOOKBACK,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "batch_size": BATCH_SIZE,
        "scaler_type": "RobustScaler",
        "metrics": metrics_serializable,
    }
    config_path = os.path.join(model_dir, "model_config_optimized.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"✓ Model config saved: {config_path}")
    
    return {
        "weights": model_weights_path,
        "x_scaler": x_scaler_path,
        "y_scaler": y_scaler_path,
        "config": config_path,
    }


def load_model_package(model_dir: str , use_optimized: bool = True):
    """Load model, scalers, and config from files."""
    import os
    
    suffix = "_optimized" if use_optimized else ""
    
    # Load config
    config_path = os.path.join(model_dir, f"model_config{suffix}.json")
    with open(config_path, "r") as f:
        config = json.load(f)
    
    # Recreate model
    model = ImprovedLSTMRegressor(
        n_features=config["n_features"],
        hidden_size=config["hidden_size"],
        num_layers=config["num_layers"],
        dropout=config["dropout_rate"],
    )
    
    # Load weights
    model_weights_path = os.path.join(model_dir, f"lstm_model_weights{suffix}.pt")
    model.load_state_dict(torch.load(model_weights_path, map_location="cpu"))
    model.eval()
    
    # Load scalers
    x_scaler_path = os.path.join(model_dir, f"x_scaler{suffix}.pkl")
    with open(x_scaler_path, "rb") as f:
        x_scaler = pickle.load(f)
    
    y_scaler_path = os.path.join(model_dir, f"y_scaler{suffix}.pkl")
    with open(y_scaler_path, "rb") as f:
        y_scaler = pickle.load(f)
    
    print(f"✓ Model loaded from {model_dir}")
    print(f"  Features: {len(config['feature_columns'])} ({', '.join(config['feature_columns'][:5])}...)")
    print(f"  Architecture: {config['num_layers']}-layer bidirectional LSTM")
    print(f"  Test Metrics: MAE=${config['metrics']['mae']:.4f}, RMSE=${config['metrics']['rmse']:.4f}, MAPE={config['metrics']['mape']:.2f}%")
    
    return model, x_scaler, y_scaler, config


CSV_PATH = r""# Change to the desired path for the input CSV file containing the stock data 
MODEL_SAVE_PATH = r""  # Change to the desired path for saving the model weights
def main():
    print(f"Device: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")
    print(f"Loading data from {CSV_PATH}...")
    data = load_data(CSV_PATH)
    print(f"Loaded {len(data)} records\n")
    
    train_df, valid_df, test_df = split_time_series(data)
    print(f"Train: {len(train_df)} | Valid: {len(valid_df)} | Test: {len(test_df)}")

    (X_train, y_train), (X_valid, y_valid), (X_test, y_test), x_scaler, y_scaler = prepare_splits(
        train_df,
        valid_df,
        test_df,
        LOOKBACK,
    )
    print(f"Sequences created - Train: {X_train.shape} | Valid: {X_valid.shape} | Test: {X_test.shape}")
    print(f"Features: {len(get_all_features())} - {', '.join(get_all_features()[:5])}...\n")

    train_loader = DataLoader(SequenceDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
    valid_loader = DataLoader(SequenceDataset(X_valid, y_valid), batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(SequenceDataset(X_test, y_test), batch_size=BATCH_SIZE, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ImprovedLSTMRegressor(n_features=X_train.shape[-1]).to(device)
    print(f"Model initialized - {X_train.shape[-1]} input features, 128 hidden units, 3 layers\n")

    print("="*60)
    print("TRAINING LSTM MODEL")
    print("="*60)
    history = train_model(model, train_loader, valid_loader, device)
    
    print("\n" + "="*60)
    print("EVALUATING ON TEST SET")
    print("="*60)
    metrics = evaluate(model, test_loader, y_scaler, device)
    
    print("="*60)
    print("SAVING MODEL PACKAGE")
    print("="*60)
    save_model_package(model, x_scaler, y_scaler, metrics)
    
    print("\n✅ Training complete! Model ready for inference.")


if __name__ == "__main__":
    main()
