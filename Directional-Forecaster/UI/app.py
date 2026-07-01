import tkinter as tk
from extraction import extract_symbol as es
import UI.search_suggestions as ss
import api.ynstockdata as yn
import regression.linreg_main as linreg
from threading import Thread
from AI.playground import first_steps as fs
from AI.playground import first_steps_data as fsd

class StockPredictorApp:
    def __init__(self):
        # Main window setup
        self.win = tk.Tk()
        self.win.geometry("800x600")
        self.win.title("Stock Predictor App")

        # Data
        self.close_value_list = []
        self.date_value_list = []

        self.time_period = ["1d", "5d", "7d", "1mo", "3mo"]  
        self.symbols = es.symbols
        self.time_intervals = ["1m", "2m", "5m", "15m", "30m", "60m", "90m"]

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

        # Search suggestion
        self.search_engine = ss.search_engine(self.win, sb_entry, self.sb_sv, self.symbols)

        # Uppercase enforcement
        self.sb_sv.trace_add("write", self.on_entry_change)

        # Show Graph    
        show_graph = tk.Button(self.win, text="Show Graph", command=self.show_table)
        show_graph.place(relx=.3, rely=.4)

    def on_entry_change(self, *args):
        """Force uppercase and refresh suggestions."""
        current = self.sb_sv.get()
        self.sb_sv.set(current.upper())
        self.search_engine.output()

    def show_table(self):
        """Fetch stock data, update close values, and run regression."""
        pop = yn.Stock(self.sb_sv.get())
        api_info = pop.getInfo().history(
            period=self.tp_sv.get(),
            interval=self.ti_iv.get()
        )
        

        self.close_value_list = api_info["Close"].tolist()
        self.date_value_list = api_info.index.strftime("%D:%H:%M").tolist()

        (
            train_loader,
            test_loader,
            minimumx,
            maximumx,
            minimumy,
            maximumy
        ) = fs.prepare_data(self.close_value_list)

        fs.order_Steps(
            fs.chow,
            train_loader,
            test_loader,
            minimumy,
            maximumy,
            self.date_value_list
        
        )

        

        # -------------------------------- make popup window ---------------------------------
        app_Popup = tk.Toplevel()
        app_Popup.geometry("900x600")
        app_Popup.title(f"{self.sb_sv.get()} stock properties from {self.date_value_list[0]} to {self.date_value_list[-1]} at {self.ti_iv.get()} intervals")

        #Dataframe
        df_Text = tk.Text(app_Popup, wrap="none")  # wrap="none" for horizontal scrolling
        df_Text.insert("1.0", api_info.to_string())
        df_Text.configure(state="disabled")  # make it read-only
        df_Text.pack(side="left", fill="both", expand=True)

        # Scrollbars
        scroll_y = tk.Scrollbar(app_Popup, orient="vertical", command=df_Text.yview)
        scroll_y.pack(side="right", fill="y")
        df_Text.configure(yscrollcommand=scroll_y.set)




    
        



