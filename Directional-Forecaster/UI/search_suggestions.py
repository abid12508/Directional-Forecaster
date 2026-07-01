import tkinter as tk

class search_engine:
    def __init__(self, win, entry, strvar, symbols):
        self.win = win
        self.entry = entry
        self.strvar = strvar
        self.symbols = symbols
        
        self.suggested_text = tk.Listbox(self.win, height=31, width=30,
                                         bg="white", activestyle="dotbox",
                                         font=("Arial", 12), fg="black")
        self.suggested_text.place(relx=.657, rely=0)

        # Bind clicking event to the listbox
        self.suggested_text.bind("<<ListboxSelect>>", self.select_item)

    def find_suggestion(self, *args):
        typed = self.strvar.get()
        self.suggested_text.delete(0, tk.END)
        if typed:
            for symbol in self.symbols:
                if symbol.startswith(typed.upper()):
                    self.suggested_text.insert(tk.END, symbol)

    def output(self, *args):
        self.find_suggestion()

    def select_item(self, event):
        # Get selected item
        selection = event.widget.curselection()
        if selection:
            index = selection[0]
            value = event.widget.get(index)
            self.strvar.set(value)  # Set entry to selected value
