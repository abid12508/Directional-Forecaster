import math 
import numpy as np

def returns(closing_price):
    return_list = [float("nan")]

    #Iterate from first index (avoid negative indexing)
    for i in range(1, len(closing_price)):

        #define i and i-1 closing price
        pnow = closing_price[i]
        ppast = closing_price[i-1]

        #Compute Log return to feed lstm 
        r = (pnow/ppast)
        logr = math.log(r)

        #Add logr to return list
        return_list.append(logr)

    
    #Output the list to feed to lstm
    return return_list


def moving_Average(closing_price, window):

    #Set up window
    rolling_sum = sum(closing_price[:window])

    #Create list and calculate first MA window
    Ma_list = [rolling_sum / window]

    """From the windowth index to the end of the list, add the rolling sum by the 
    difference of the closing price at point I by the closing price at point I-window"""
    for i in range(window, len(closing_price)):
        rolling_sum += (closing_price[i] - closing_price[i - window])
        Ma_list.append(rolling_sum / window)

    return Ma_list


def volatility(closing_price, window):

    #Make returns an array to compute stdev
    r = np.array(returns(closing_price), dtype=float)

    #Calculate standard deviation of values within window
    vol = []
    for i in range(len(r) - window + 1):
        vol.append(np.std(r[i:i+window], ddof=1))

    return vol

def rsi(closing_price, window):

    r = returns(closing_price)
    rsi_list = []

    for i in range(len(r) - window + 1):

        #Find the gains and losses in window
        gains = [max(j, 0) 
                 for j in r[i:window+i]]
        losses = [abs(min(k, 0)) 
                 for k in r[i:window+i]]

        #Make those gains an average
        gains = np.mean(gains)
        losses = np.mean(losses)

        #Create case if no losses
        if losses == 0:
            rsi_list.append(100)
            continue
        
        #Calculate RSI
        rs = gains/losses
        rsi = 100 - (100 / (1 + rs))

        rsi_list.append(rsi)
    
    return rsi_list



    






        