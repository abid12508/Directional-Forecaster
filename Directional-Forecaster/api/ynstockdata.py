import yfinance as yf
import UI.app as app

class Stock:
    def __init__(self, ticker: str):
        self.stk = yf.Ticker(ticker)

    
    def getInfo(self) -> yf.Ticker:
        return self.stk
