import os
import sys
import math
import matplotlib.pyplot as plt
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import parse_data

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


learning_parameters = []
errors = []

change_parameter = .00001
for i in range(100): 
    # tweak learning parameter to find optimal one
    change_parameter = change_parameter / 2
    weights = [0] * len(data[0][0])
    for m in range(100):
    # create weights array
        error_total = 0
        for i in range(len(data)):
            # get the x and y vals for this data point
            x_vals = data[i][0][:]
            y_val = data[i][1][0]
            # get predicted value
            prediction = 0
            for j in range(len(x_vals)):
                prediction += weights[j] * x_vals[j] 
                # calculate error
            error = y_val - prediction
            # update weights based on error
            for j in range(len(weights)):
                weights[j] = weights[j] + (error * change_parameter * x_vals[j])
                # add error to total
            error_total += abs(error / y_val)
        # if minimum error store results
        if abs(error_total) < abs(min_error):
            min_error = abs(error_total)
            min_n = y_int
            min_weights = []
            for p in range(len(weights)):
                min_weights.append(weights[p])
            min_change_parameter = change_parameter
        if m == 99 and not math.isnan(error_total) and error_total < 10000 :
            errors.append(error_total * 100)
            learning_parameters.append(change_parameter)

print(f'optimal weights are {min_weights}')
print(f'optimal y int is {min_n}')
print(f'min_error is {min_error}')
print(f'optimal learning paramter is {min_change_parameter}')

print(learning_parameters)
print(errors)


plt.plot(learning_parameters, errors, marker='o', label='Total Error')
plt.xscale('log')
plt.xlabel('Learning Rate (in log scale between 0 and 1)')
plt.ylabel('Total Error in % (Using cumulative percent error)')
plt.title('Learning Rate vs Total Error')
plt.legend()
plt.show()