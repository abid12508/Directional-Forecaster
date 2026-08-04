import torch
from sklearn.preprocessing import StandardScaler
import numpy as np
from Finance import feature_computation as fc

SEQUENCE_LEN = 5

def pad_features(feature_list, closing_price):

    target_length = len(closing_price)

    for i in range(len(feature_list)):

        difference = target_length - len(feature_list[i])

        if difference > 0:
            feature_list[i] = (
                [float("nan")] * difference
                + feature_list[i]
            )

    return feature_list


def feature_select(closing_price, selected_features, windows):
    feature_list = []

    if "Returns" in selected_features:
        feature_list.append(fc.returns(closing_price))

    if "MA5" in selected_features:
        feature_list.append(fc.moving_Average(closing_price, 5))

    if "MA10" in selected_features:
        feature_list.append(fc.moving_Average(closing_price, 10))

    if "MA20" in selected_features:
        feature_list.append(fc.moving_Average(closing_price, 20))

    if "Volatility" in selected_features:
        feature_list.append(fc.volatility(closing_price, windows["volatility"]))

    if "RSI" in selected_features:
        feature_list.append(fc.rsi(closing_price, windows["rsi"]))

    feature_list = pad_features(feature_list, closing_price)
    feature_list = torch.tensor(feature_list, dtype=torch.float32).T

    valid_rows = ~torch.isnan(feature_list).any(dim=1)

    feature_list = feature_list[valid_rows]

    return feature_list, valid_rows



def x_tensor_preprocess(feature_list):

    training_split = int(len(feature_list) * .8)
    validation_split = int(len(feature_list) * .9)

    training_set, testing_set = feature_list[:training_split], feature_list[validation_split:]
    validation_set = feature_list[training_split:validation_split]

    scaler = StandardScaler()

    train_scaled = scaler.fit_transform(training_set.numpy())

    val_scaled = scaler.transform(validation_set.numpy())
    test_scaled = scaler.transform(testing_set.numpy())

    x_train_tensor = torch.tensor(
        train_scaled,
        dtype=torch.float32
    )

    x_val_tensor = torch.tensor(
        val_scaled,
        dtype=torch.float32
    )

    x_test_tensor = torch.tensor(
        test_scaled,
        dtype=torch.float32
    )

    return x_train_tensor, x_val_tensor, x_test_tensor, training_split, validation_split, scaler


def y_tensor_preprocess(closing_price, valid_rows, training_split, validation_split):

    returns = np.array(fc.returns(closing_price))

    y = (returns > 0).astype(int)

    y = y[valid_rows.numpy()]

    y_train = y[:training_split]
    y_val = y[training_split:validation_split]
    y_test = y[validation_split:]

    y_train_tensor = torch.tensor(
        y_train, 
        dtype=torch.long
    )

    y_val_tensor = torch.tensor(
        y_val,
        dtype=torch.long
    )

    y_test_tensor = torch.tensor(
        y_test,
        dtype=torch.long
    )

    return y_train_tensor, y_val_tensor, y_test_tensor


def create_sequences(x, y, sequence_length):

    X_sequences = []
    y_sequences = []

    for i in range(sequence_length, len(x)):

        X_sequences.append(
            x[i-sequence_length:i]
        )

        y_sequences.append(
            y[i]
        )

    X_sequences = torch.stack(X_sequences)
    y_sequences = torch.stack(y_sequences)

    return X_sequences, y_sequences


def tensor_dataset(closing_price, selected_features, windows):

    feature_list, valid_rows = feature_select(closing_price, selected_features, windows)

    X_train_tensor, X_val_tensor, X_test_tensor, training_split, validation_split, scaler = x_tensor_preprocess(feature_list)
    y_train_tensor, y_val_tensor, y_test_tensor = y_tensor_preprocess(closing_price, valid_rows, training_split, validation_split)

    X_train, y_train = create_sequences(
        X_train_tensor,
        y_train_tensor,
        SEQUENCE_LEN
    )

    X_val, y_val = create_sequences(
        X_val_tensor,
        y_val_tensor,
        SEQUENCE_LEN
    )

    X_test, y_test = create_sequences(
        X_test_tensor,
        y_test_tensor,
        SEQUENCE_LEN
    )

    return X_train, y_train, X_val, y_val, X_test, y_test, scaler


def build_next_step_input(closing_price, selected_features, windows, scaler, sequence_length=SEQUENCE_LEN):

    feature_list, _ = feature_select(closing_price, selected_features, windows)

    if len(feature_list) < sequence_length:
        raise ValueError(
            f"Need at least {sequence_length} valid feature rows to build a "
            f"sequence, but only {len(feature_list)} are available."
        )

    latest_rows = feature_list[-sequence_length:]

    scaled = scaler.transform(latest_rows.numpy())

    next_step_input = torch.tensor(scaled, dtype=torch.float32).unsqueeze(0)  # [1, seq_len, n_features]

    return next_step_input