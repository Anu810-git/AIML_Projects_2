"""Train a beginner-friendly PyTorch MLP for next-category prediction."""

import json
import joblib
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

try:
    from config import DB_PATH, MODEL_DIR, REPORT_DIR, RANDOM_STATE
    from database import query_dataframe
except ImportError:
    from src.config import DB_PATH, MODEL_DIR, REPORT_DIR, RANDOM_STATE
    from src.database import query_dataframe


class NextCategoryMLP(nn.Module):
    """Simple multi-layer perceptron."""

    def __init__(self, n_categories: int):
        super().__init__()

        # Input = 5 category IDs.
        self.network = nn.Sequential(
            nn.Linear(5, 64),
            nn.ReLU(),
            nn.Dropout(0.20),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.20),
            nn.Linear(32, n_categories),
        )

    def forward(self, x):
        return self.network(x)


def make_sequences(df: pd.DataFrame, encoder: LabelEncoder):
    """Convert browsing histories into 5-category -> next-category samples."""
    X, y = [], []

    for _, group in df.sort_values("event_time").groupby("customer_id"):
        sequence = group["product_category"].tolist()

        if len(sequence) < 6:
            continue

        encoded = encoder.transform(sequence)

        for i in range(len(encoded) - 5):
            X.append(encoded[i:i + 5].astype(np.float32))
            y.append(encoded[i + 5])

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64)


def train():
    MODEL_DIR.mkdir(exist_ok=True)
    REPORT_DIR.mkdir(exist_ok=True)

    query = """
    SELECT customer_id, event_time, product_category
    FROM fact_browsing
    ORDER BY customer_id, event_time
    """
    browsing = query_dataframe(query)

    encoder = LabelEncoder()
    encoder.fit(browsing["product_category"])

    X, y = make_sequences(browsing, encoder)

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=RANDOM_STATE, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=RANDOM_STATE, stratify=y_temp
    )

    train_loader = DataLoader(
        TensorDataset(torch.tensor(X_train), torch.tensor(y_train)),
        batch_size=64,
        shuffle=True,
    )

    val_x = torch.tensor(X_val)
    val_y = torch.tensor(y_val)
    test_x = torch.tensor(X_test)
    test_y = torch.tensor(y_test)

    torch.manual_seed(RANDOM_STATE)
    model = NextCategoryMLP(len(encoder.classes_))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    history = {"train_loss": [], "val_loss": [], "val_accuracy": []}

    best_val_loss = float("inf")
    best_state = None
    patience = 12
    patience_counter = 0

    for epoch in range(80):
        model.train()
        train_losses = []

        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        with torch.no_grad():
            val_logits = model(val_x)
            val_loss = criterion(val_logits, val_y).item()
            val_accuracy = (
                (val_logits.argmax(dim=1) == val_y).float().mean().item()
            )

        train_loss = float(np.mean(train_losses))
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_accuracy)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {
                key: value.detach().clone()
                for key, value in model.state_dict().items()
            }
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        test_logits = model(test_x)
        test_accuracy = (
            (test_logits.argmax(dim=1) == test_y).float().mean().item()
        )

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "n_categories": len(encoder.classes_),
            "input_size": 5,
        },
        MODEL_DIR / "next_category_mlp.pt",
    )
    joblib.dump(encoder, MODEL_DIR / "category_encoder.joblib")

    (REPORT_DIR / "dl_history.json").write_text(
        json.dumps({
            "epochs_ran": len(history["train_loss"]),
            "test_accuracy": test_accuracy,
            "history": history,
        }, indent=2),
        encoding="utf-8",
    )

    # Save a simple loss curve for the report.
    import matplotlib.pyplot as plt
    plt.figure(figsize=(8, 4))
    plt.plot(history["train_loss"], label="Train loss")
    plt.plot(history["val_loss"], label="Validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-entropy loss")
    plt.title("PyTorch Next-Category Loss Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(REPORT_DIR / "dl_loss_curve.png", dpi=150)
    plt.close()

    print(f"Sequences: {len(X):,}")
    print(f"Test accuracy: {test_accuracy:.3f}")


if __name__ == "__main__":
    train()
