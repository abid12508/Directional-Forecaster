import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from skopt import gp_minimize
from skopt.space import Integer, Real

from . import model
from . import preprocessing as pp

EPOCHS = 100
BATCH_SIZE = 32

search_space = [
    Integer(5, 20, name="volatility_window"),
    Integer(5, 20, name="rsi_window"),
    Real(1e-4, 1e-2, prior="log-uniform", name="learning_rate"),
    Integer(16, 128, name="hidden_size"),
    Integer(1, 3, name="layers")
]

def obtain_data(closing_price, selected_features, windows):
    return pp.tensor_dataset(closing_price, selected_features, windows)

def training_model(model, train_loader, optimizer, criterion, device, epochs):

    loss_vals = []

    for epoch in range(epochs):

        model.train()

        epoch_loss = 0.0

        for batch_X, batch_y in train_loader:

            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()

            output = model(batch_X)

            loss = criterion(output, batch_y)

            loss.backward()

            optimizer.step()

            epoch_loss += loss.item() * batch_X.size(0)

        average_epoch_loss = (
            epoch_loss / len(train_loader.dataset)
        )

        loss_vals.append(average_epoch_loss)

        print(
            f"Epoch: {epoch + 1}/{epochs} | "
            f"Loss: {average_epoch_loss}"
        )

    return loss_vals


def validate_model(model, val_loader, criterion, device):

    model.eval()

    total_loss = 0.0

    with torch.no_grad():

        for batch_X, batch_y in val_loader:

            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)

            output = model(batch_X)

            loss = criterion(output, batch_y)

            total_loss += (
                loss.item() * batch_X.size(0)
            )

    average_loss = (
        total_loss / len(val_loader.dataset)
    )

    return average_loss

def run_training(closing_price, selected_features, volatility_window,
                 rsi_window, learning_rate, hidden_size, layers):

    windows = {
        "volatility": volatility_window,
        "rsi": rsi_window
    }

    X_train, y_train, X_val, y_val, X_test, y_test = obtain_data(
        closing_price,
        selected_features, 
        windows
    )

    train_loader = DataLoader(
        TensorDataset(X_train, y_train),
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    val_loader = DataLoader(
        TensorDataset(X_val, y_val),
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create model
    model_instance = model.StockLSTM(
        input_size=X_train.shape[2],
        hidden_size=hidden_size,
        layers=layers
    ).to(device)

    # Loss
    criterion = nn.CrossEntropyLoss()

    # Optimizer
    optimizer = optim.Adam(
        model_instance.parameters(),
        lr=learning_rate
    )

    # Train
    loss_vals = training_model(
        model=model_instance,
        train_loader=train_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        epochs=EPOCHS
    )

    # Validate
    validation_loss = validate_model(
        model=model_instance,
        val_loader=val_loader,
        criterion=criterion,
        device=device
    )

    return validation_loss

def objective(volatility_window, rsi_window, 
              learning_rate, hidden_size, layers,
              closing_price, selected_features):

    validation_loss = run_training(
            closing_price=closing_price,
            selected_features=selected_features,
            volatility_window=volatility_window,
            rsi_window=rsi_window,
            learning_rate=learning_rate,
            hidden_size=hidden_size,
            layers=layers
        )

    return validation_loss

def bayesian_objective(params, closing_price, selected_features):

    volatility_window = int(params[0])
    rsi_window = int(params[1])
    learning_rate = float(params[2])
    hidden_size = int(params[3])
    layers = int(params[4])

    return objective(
        volatility_window, rsi_window, learning_rate,
        hidden_size, layers, closing_price, selected_features
    )

def run_bayesian_optimization(closing_price, selected_features):

    result = gp_minimize(
        func=lambda params: bayesian_objective(
            params, closing_price, selected_features
        ),
        dimensions=search_space,
        n_calls=5,
        n_initial_points=5,
        random_state=42
    )

    return result

def run_final_optimized_model(closing_price, selected_features, best_params):

    volatility_window = int(best_params[0])
    rsi_window = int(best_params[1])
    learning_rate = float(best_params[2])
    hidden_size = int(best_params[3])
    layers = int(best_params[4])

    validation_loss = run_training(closing_price=closing_price,
                 selected_features=selected_features,
                 volatility_window=volatility_window,
                 rsi_window=rsi_window,
                 learning_rate=learning_rate,
                 hidden_size=hidden_size,
                 layers=layers)

    return validation_loss