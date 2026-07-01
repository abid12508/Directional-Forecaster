# ai imports
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# graphing
from matplotlib import pyplot as plt

#import our dataset
from . import first_steps_data as fsd

#other imports
import os
from datetime import datetime
import tkinter as tk

import pandas as pd
pd.set_option('display.max_rows', None)      
pd.set_option('display.max_columns', None)   
pd.set_option('display.width', None)         
pd.set_option('display.max_colwidth', None) 

import numpy as np

#---------------------- test algorithm -----------------------------

#saw this in a tutorial
#torch.manual_seed(34180) #for reproducibility? just picked a random number for now




#### HYPER PARAMETERS #####
HIDDEN_SIZE = 64
LAYERS = 5
EPOCHS = 100
BATCH_SIZE = 64
WINDOW = 3


#---------------------------------------prepare all data-------------------------------------------\
def prepare_data(close_prices):
    X, y = fsd.closing_data(close_prices)

    np_x = np.array(X, dtype=np.float32)
    np_y = np.array(y, dtype=np.float32)

    split_idx = int(len(np_x) * .8)

    x_train = np_x[:split_idx]
    y_train = np_y[:split_idx]

    x_test = np_x[split_idx + WINDOW:]
    y_test = np_y[split_idx + WINDOW:]

    minimumx, maximumx = x_train.min(), x_train.max()
    minimumy, maximumy = y_train.min(), y_train.max()

    x_train = (x_train-minimumx)/(maximumx-minimumx)
    x_test  = (x_test-minimumx)/(maximumx-minimumx)

    y_train = (y_train-minimumy)/(maximumy-minimumy)
    y_test  = (y_test-minimumy)/(maximumy-minimumy)

    x_train = x_train[:,:,np.newaxis]
    x_test = x_test[:,:,np.newaxis]

    X_train = torch.tensor(x_train,dtype=torch.float32)
    y_train = torch.tensor(y_train,dtype=torch.float32)

    X_test = torch.tensor(x_test,dtype=torch.float32)
    y_test = torch.tensor(y_test,dtype=torch.float32)

    train_loader = DataLoader(
        TensorDataset(X_train,y_train),
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    test_loader = DataLoader(
        TensorDataset(X_test,y_test),
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    return (
        train_loader,
        test_loader,
        minimumx,
        maximumx,
        minimumy,
        maximumy
    )


#Model Saving variables
MODEL_NAME = "AbidLSTM"
MODEL_WEIGHT_PATH = "weights"
INPUT_SIZE = 1
os.makedirs(MODEL_WEIGHT_PATH, exist_ok=True)


# Create our model class
class AbidLSTM(nn.Module):
    def __init__(self):
        super(AbidLSTM, self).__init__() # required because its a subclass
        #input size -> input a single list of numbers
        #hidden size -> number of neurons between the input layer and the output layer, "brain"
        #batch first -> idk

        self.lstm = nn.LSTM(
            input_size=INPUT_SIZE,
            hidden_size=HIDDEN_SIZE,
            num_layers=LAYERS,
            batch_first=True            

        )
        self.dropout = nn.Dropout(p=.01)
        self.linear = nn.Linear(HIDDEN_SIZE, 1)
        


    def forward(self, x):
        model_output, _ = self.lstm(x)
        last_time_step = model_output[:, -1, :]
        out = self.dropout(last_time_step)
        return self.linear(out)


#Check if user has GPU, if not, then just use cpu
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# an instance of our baby model
chow = AbidLSTM()
chow.to(device)
# create our loss function and our optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(chow.parameters(), lr=0.001) #no clue about parameters() but our learning rate is 0.01


#time to train our model!
#loops over data set epoch times
loss_vals = []


# -------------------------------------------TRAINING------------------------------------------------------- #
def training_model(train_loader):
    for epoch in range(EPOCHS):
        chow.train()
        epoch_loss = 0.0
        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()

            output = chow(batch_X)

            loss = criterion(output, batch_y)
            loss.backward()

            optimizer.step()

            epoch_loss += loss.item() * batch_X.size(0)

        average_epoch_loss = epoch_loss / len(train_loader.dataset)
        print(f"Epoch: {epoch} | Average Dataset Loss: {average_epoch_loss}")
        loss_vals.append(average_epoch_loss)

#visual representation of predicted vs actual (variables)
model_predictions = []
actual_values = []
percent_changes = []
differences = []

# -------------------------------------------TESTING------------------------------------------------------- #
def testing_model_implement(
    Model,
    Test_set,
    minimumy,
    maximumy,
    model_predictions,
    actual_values,
    percent_changes,
    differences
):

    Model.eval()

    with torch.no_grad():
        for t_seq, actual in Test_set:
            t_seq = t_seq.to(device)

            prediction = Model(t_seq)

            prediction = prediction.cpu().numpy()
            actual = actual.cpu().numpy()

            prediction = prediction * (maximumy - minimumy) + minimumy
            actual = actual * (maximumy - minimumy) + minimumy

            for pred, act in zip(prediction, actual):
                pred = pred.item()
                act = act.item()

                model_predictions.append(pred)
                actual_values.append(act)

                difference = abs(pred - act)
                differences.append(difference)

                if act != 0:
                    percent_changes.append(100 * difference / abs(act))
                else:
                    percent_changes.append(0)

#plotting epocs vs losses
def plt_evl():
    graph_x = [i for i in range(0, len(loss_vals))]
    graph_y = [float(i) for i in loss_vals]

    plt.figure(1)
    plt.plot(graph_x, graph_y, color="b")

    plt.title("Epochs vs Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Losses")

    plt.grid(True)

#plotting predicted vs actual
def plt_pva(time):
    n = min(len(model_predictions), len(actual_values))
    x = time[-len(actual_values):]

    ax = plt.figure(2)
    plt.scatter(x, model_predictions[:n], label="Predicted", color='r')
    plt.scatter(x, actual_values[:n], label="Actual", color='b')

    for i in range(n):
        plt.plot(
            [x[i], x[i]],
            [model_predictions[i], actual_values[i]],
            color='purple',
            linewidth=1
        )
        plt.plot([x[i-1], x[i]], [model_predictions[i-1], model_predictions[i]],
                 color="blue")

    plt.xlabel('Tests')
    plt.xticks(fontsize=8)
    plt.ylabel('Predictions and Actuals')
    plt.title('Predicted vs Actual')
    plt.legend()
    plt.show()


#saving model if it is accurate by .1%
def save_Model():

    #calculate the average for all the percent changes
    MEAN_ERROR = np.mean(percent_changes)

    #if the average error is lower than .1%, save the model
    if (MEAN_ERROR < .1):
        today = datetime.now().strftime("%Y-%m-%d")
        file_path = f"{MODEL_WEIGHT_PATH}/{MODEL_NAME}_{MEAN_ERROR:.5f}_{today}.pth"
        torch.save(chow.state_dict(), file_path)
        print(f"model saved as: {file_path}")


def create_datatable(actual_values, model_predictions, percent_changes, differences, date_set):
    
    WINDOW = 3
    aligned_dates = date_set[-len(actual_values):]

    datatable = {
        "Actual": actual_values,
        "Predicted": model_predictions,
        "%Change": percent_changes,
        "Difference": differences
    }

    dataframe = pd.DataFrame(datatable, index=aligned_dates)
    dataframe.index.name = "Timestamp"

    # -------------------------------- make popup window ---------------------------------
    app_Popup = tk.Toplevel()
    app_Popup.geometry("900x600")
    app_Popup.title("AI model testing table")

    #Dataframe
    df_Text = tk.Text(app_Popup, wrap="none")  # wrap="none" for horizontal scrolling
    df_Text.insert("1.0", dataframe.to_string())
    df_Text.configure(state="disabled")  # make it read-only
    df_Text.pack(side="left", fill="both", expand=True)

    # Scrollbars
    scroll_y = tk.Scrollbar(app_Popup, orient="vertical", command=df_Text.yview)
    scroll_y.pack(side="right", fill="y")
    df_Text.configure(yscrollcommand=scroll_y.set)

def order_Steps(chow,
                train_loader,
                test_loader,
                minimumy,
                maximumy,
                date_set):

    model_predictions.clear()
    actual_values.clear()
    percent_changes.clear()
    differences.clear()

    training_model(train_loader)

    plt_evl()

    testing_model_implement(
        chow,
        test_loader,
        minimumy,
        maximumy,
        model_predictions,
        actual_values,
        percent_changes,
        differences
    )

    plt_pva(date_set)
    create_datatable(
        actual_values,
        model_predictions,
        percent_changes,
        differences,
        date_set
    )

    save_Model()



        
        