import tkinter as tk

from extraction import extract_symbol as es

import UI.search_suggestions as ss

from api import stock_data as sd

from AI import training as tr
from AI import testing as ts

import pandas as pd

class Directional_Forecaster:
    def __init__(self):
        # Main window setup
        self.win = tk.Tk()
        self.win.geometry("800x600")
        self.win.title("Directional_Forecaster")

        # Data
        self.close_value_list = None
        self.date_value_list = None

        self.time_period = ["1d", "5d", "7d", "1mo", "3mo"]  
        self.symbols = es.symbols
        self.time_intervals = ["1m", "2m", "5m", "15m", "30m", "60m", "90m"]

        self.features = ["Returns", 
                         "MA5",
                         "MA10",
                         "MA20",
                         "Volatility",
                         "RSI",
                         "MACD",
                         "Bollinger"]
        
        self.selected_features = None

        # UI elements
        self.create_widgets()

    def run(self):
        # Mainloop
        self.win.mainloop()


    def create_widgets(self):
        # Time Period dropdown
        tp_label = tk.Label(self.win, text="Time Period")
        self.tp_sv = tk.StringVar(value="1d")  
        tp_dropdown = tk.OptionMenu(self.win, self.tp_sv, *self.time_period)
        tp_label.place(relx=.2, rely=.1)
        tp_dropdown.place(relx=.4, rely=.1)

        # Stock entry
        sb_label = tk.Label(self.win, text="Stock")
        self.sb_sv = tk.StringVar()  
        sb_entry = tk.Entry(self.win, textvariable=self.sb_sv)
        sb_label.place(relx=.2, rely=.2)
        sb_entry.place(relx=.4, rely=.2)

        # Time Intervals dropdown
        ti_label = tk.Label(self.win, text="Time Intervals")
        self.ti_iv = tk.StringVar(value="5m")  
        ti_dropdown = tk.OptionMenu(self.win, self.ti_iv, *self.time_intervals)
        ti_label.place(relx=.2, rely=.3)
        ti_dropdown.place(relx=.4, rely=.3)

        # Feature Popup    
        f_button = tk.Button(self.win, text="Select Features", command=self.feature_window)
        f_button.place(relx=.3, rely=.4)

        # Search suggestion
        self.search_engine = ss.search_engine(self.win, sb_entry, self.sb_sv, self.symbols)

        # Uppercase enforcement
        self.sb_sv.trace_add("write", self.on_entry_change)

        # Show Graph    
        show_graph = tk.Button(self.win, text="Submit", command=self.show_table)
        show_graph.place(relx=.3, rely=.5)
    
    def feature_window(self):
        self.feature_popup = tk.Toplevel()
        self.feature_popup.geometry("900x600")
        self.feature_popup.title("Feature Selection")

        tk.Label(self.feature_popup, text="Select the features you want to input into your LSTM.\n\
        (Recommend 10-20 features to ensure optimal performance)").place(relx=0.5, rely=0.05, anchor="center")

        self.feature_checkboxes()
    
    def feature_checkboxes(self):
        self.feature_vars = {}
        i = 0.1

        for f in self.features:
            self.feature_vars[f] = tk.BooleanVar()

            feature_box = tk.Checkbutton(
                self.feature_popup,
                text=f,
                variable=self.feature_vars[f]
            )

            feature_box.place(
                relx=.45,
                rely=i
            )

            i += .05

    def get_selected_features(self):

        self.selected_features = [
            feature for feature, var in self.feature_vars.items() if var.get()
        ]

        return self.selected_features
    

    def on_entry_change(self, *args):
        """Force uppercase and refresh suggestions."""
        current = self.sb_sv.get()
        self.sb_sv.set(current.upper())
        self.search_engine.output()

    def show_table(self):

        self.selected_features = self.get_selected_features()

        pop = sd.Stock(self.sb_sv.get())

        api_info = pop.getInfo().history(
            period=self.tp_sv.get(),
            interval=self.ti_iv.get()
        )

        last_time = api_info.index[-1]
        next_time = last_time + pd.Timedelta(minutes=int(self.ti_iv.get()[:-1]))

        self.close_value_list = api_info["Close"].tolist()
        self.date_value_list = pd.to_datetime(api_info.index).strftime("%m/%d/%y %H:%M").tolist()
        
        print("Training model...")

        training_result = tr.run_bayesian_optimization(closing_price=self.close_value_list, 
                                                       selected_features=self.selected_features)

        print(f"best params: {training_result.x}, best validation loss: {training_result.fun}")
        print("Running final training step and testing...")

        test_result, prediction, confidence = ts.run_testing(closing_price=self.close_value_list, 
                                                             selected_features=self.selected_features,
                                                             best_params=training_result.x)

        print(f"Final training step and test validation loss: {test_result}")
        print(f" Predicted {next_time} step direction: {prediction} with {confidence:.2%}% confidence")




    
        



