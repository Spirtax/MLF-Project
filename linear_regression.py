import parse_data
import math



df = parse_data.df 
    
X = df[['Company Name', 'Model Name', 'Mobile Weight', 'RAM', 'Front Camera', 'Back Camera',
            'Processor', 'Battery Capacity', 'Screen Size', 'Launched Year']].values.tolist()
    
for i in range(len(X)):
    X[i].append(1)
y = df[['Launched Price (USA)']].values.tolist()

data = list(zip(X, y))


change_parameter = .00000001
n = 100
min_error = 1000000000
min_n = 1
min_weights = []
min_change_parameter = 1
y_int = 1


for m in range(n):
    y_int = y_int +5
    print(y_int)
    change_parameter = 1
    for i in range(100): 
        change_parameter = change_parameter / 2
        weights = [0] * len(data[0][0])
        for k in range(10):
            error_total = 0
            for i in range(len(data)):
                x_vals = data[i][0][:]
                y_val = data[i][1][0]
                x_vals[len(x_vals)-1] = y_int
                prediction = 0
                for j in range(len(x_vals)):
                    prediction += weights[j] * x_vals[j] 
                error = y_val - prediction
                for j in range(len(weights)):
                    weights[j] = weights[j] + (error * change_parameter * x_vals[j])
                error_total += abs(error / y_val)
            if abs(error_total) < abs(min_error):
                min_error = abs(error_total)
                min_n = y_int
                min_weights = []
                for p in range(len(weights)):
                    min_weights.append(weights[p])
                min_change_parameter = change_parameter

    if m == n-1:
        print(f'optimal weights are {min_weights}')
        print(f'optimal y int is {min_n}')
        print(f'min_error is {min_error}')
        print(f'optimal learning paramter is {min_change_parameter}')

    