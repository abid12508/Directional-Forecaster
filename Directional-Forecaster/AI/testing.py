import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from . import model
from . import preprocessing as pp
from . import training as tr

def test_model(model_instance, test_loader, criterion, device):

    model_instance.eval()

    total_loss = 0.0

    with torch.no_grad():

        for batch_X, batch_y in test_loader:

            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)

            output = model_instance(batch_X)

            loss = criterion(output, batch_y)

            total_loss += (
                loss.item() * batch_X.size(0)
            )

    average_loss = (
        total_loss / len(test_loader.dataset)
    )

    return average_loss


def predict_next_step(model_instance, closing_price, selected_features, windows, scaler, device):

    next_step_input = pp.build_next_step_input(
        closing_price, selected_features, windows, scaler
    )

    model_instance.eval()

    with torch.no_grad():
        logits = model_instance(next_step_input.to(device))
        probabilities = torch.softmax(logits, dim=1)

    prediction = torch.argmax(probabilities, dim=1).item()
    confidence = probabilities[0, prediction].item()

    return prediction, confidence

def run_testing(closing_price, selected_features, best_params):

    volatility_window = int(best_params[0])
    rsi_window = int(best_params[1])
    learning_rate = float(best_params[2])
    hidden_size = int(best_params[3])
    layers = int(best_params[4])

    windows = {
        "volatility": volatility_window,
        "rsi": rsi_window
    }

    X_train, y_train, X_val, y_val, X_test, y_test, scaler = tr.obtain_data(
        closing_price,
        selected_features,
        windows
    )

    train_loader = DataLoader(
        TensorDataset(X_train, y_train),
        batch_size=tr.BATCH_SIZE,
        shuffle=True
    )

    test_loader = DataLoader(
        TensorDataset(X_test, y_test),
        batch_size=tr.BATCH_SIZE,
        shuffle=False
    )

    device = torch.device(
        'cuda' if torch.cuda.is_available() else 'cpu'
    )

    # Create final model
    model_instance = model.StockLSTM(
        input_size=X_train.shape[2],
        hidden_size=hidden_size,
        layers=layers
    ).to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model_instance.parameters(),
        lr=learning_rate
    )

    # Train final model
    tr.training_model(
        model=model_instance,
        train_loader=train_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        epochs=tr.EPOCHS
    )

    # Test final trained model
    test_loss = test_model(
        model_instance=model_instance,
        test_loader=test_loader,
        criterion=criterion,
        device=device
    )

    # Predict the next, not-yet-observed time step (uses full closing_price
    # history and the scaler fit during this model's training)
    prediction, confidence = predict_next_step(
        model_instance=model_instance,
        closing_price=closing_price,
        selected_features=selected_features,
        windows=windows,
        scaler=scaler,
        device=device
    )

    return test_loss, prediction, confidence