import random as rand

#training set
def gen_data(dataset, size):
    i = 1
    while i <= size:
        # strt_pt = rand.randint(1, 1000)
        step = rand.randint(1, 20)
        
        # attempts to be inclusive for all numbers
        if i % 2 == 1:
             strt_pt = rand.uniform(0, 500)    # positive start
        else:
             strt_pt = rand.uniform(0, 50)    # around zero
        
        #make sequences longer than 3 to train larger contexts
        
        seq = [strt_pt + (j * step) for j in range(5)]
        ans = seq[-1] + step

        dataset.append((seq, ans))
        i += 1
    return dataset

def closing_data(close_list):

    X_list = []
    y_list = []

    WINDOW = 3

    l = 0
    r = WINDOW - 1

    while r + 1 < len(close_list):
        X_list.append(close_list[l:r+1])
        y_list.append([close_list[r+1]])
        l += 1
        r += 1
    


    return X_list, y_list



    

#AI implementation
""" Convert closing values into a list 
    Check Every 4 closing values (starting from the beginning of list)
    make first 3 into X tensor, 4th is answer (Y tensor).
    Activate LSTM to predict values, check difference, make adjustments 
    Shift tensor values by 1 (sliding window technique)
    Stop LSTM once future price is predicted
"""