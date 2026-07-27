import yfinance as yf

class Stock:
    def __init__(self, ticker: str):
        self.stk = yf.Ticker(ticker)

    
    def getInfo(self) -> yf.Ticker:
        return self.stk
