import random
import math
import matplotlib.pyplot as plt
import parse_data
import numpy as np
import copy

df = parse_data.df 
    
x = df[['Company Name', 'Model Name', 'Mobile Weight', 'RAM', 'Front Camera', 'Back Camera',
        'Processor', 'Battery Capacity', 'Screen Size', 'Launched Year']].values.tolist()
    
y = df[['Launched Price (USA)']].values.tolist()


def sigmoid(s):
    s = max(min(s, 10), -10) 
    denominator = 1 + math.exp(-1 * s)
    return 1 / denominator

def sigmoid_derivative(s):
    s = max(min(s, 5), -5) 
    return s * (1 - s)

def matrix_mult(A, B):
    final_matrix = []
    for row in A:
        updated_row = []
        for col in zip(*B):
            sum = 0
            for a, b in zip(row, col):
               sum += a*b
            updated_row.append(sum)
        final_matrix.append(updated_row)
    return final_matrix

def matrix_add(A, B):
    if len(A) != len(B):
        B = np.tile(B, (len(A), 1))
    final_matrix = []
    for A_row, B_row in zip(A, B):
        updated_row = []
        for a, b in zip(A_row, B_row):
            updated_row.append(a+b)
        final_matrix.append(updated_row)
    return final_matrix

def dot_product(A, B):
    sum = 0
    for A_row, B_row in zip(A,B):
        for a,b in zip(A_row, B_row):
            sum += a * b
    return [[sum]]

def initialize(number_layers, nodes_per_layer):
    weights_per_layer = []
    biases_per_layer = []
    for i in range(number_layers):
        weights_per_layer.append(np.random.randn(nodes_per_layer[i], nodes_per_layer[i + 1]))
        biases_per_layer.append(np.random.rand(1, nodes_per_layer[i + 1]))
    return weights_per_layer, biases_per_layer


def feed_forward(x, weights, biases, number_layers):
    # initial layer 
    activation_vals = [x]  
    
    
    # iterate through layers do matrix multiplication of weights, add bias 
    # and activation function, then store for back propogation later 
    for i in range(number_layers-1):
        curr_pre_activation_val = matrix_mult(activation_vals[i], weights[i])
        curr_pre_activation_val = matrix_add(curr_pre_activation_val, biases[i])
        curr_activation_val = []
        # get sigmoid vals for matrix 
        for pre_activation_row in curr_pre_activation_val:
            activation_row = []
            for val in pre_activation_row:
                activation_row.append(sigmoid(val))
            curr_activation_val.append(activation_row)
        activation_vals.append(curr_activation_val)
        
    output_layer = matrix_mult(activation_vals[-1], weights[-1])
    output_layer = matrix_add(output_layer, biases[-1])
    activation_vals.append(output_layer)  

    return activation_vals

def calculate_deltas(activation_vals, y, weights, number_layers):
      
    # use y as list since y is a list of lists
    y_list = []
    for val in y:
        y_list.append(val[0])

    # get list of errors
    total_error = []
    for i in range(len(y_list)):
        total_error.append([activation_vals[-1][i][0] - y_list[i]])
  
    # Calculate the delta for the output layer
    sigmoid_activation_vals = []
    for row in activation_vals[-1]:
        for a_val in row:
            sigmoid_activation_vals.append(sigmoid_derivative(a_val))

    # first step is to get the output layer error * activation derivitive of output layer
    output_layer_delta = dot_product(total_error, [sigmoid_activation_vals])
    deltas = [output_layer_delta] 
    
    
    # Go through hidden layers in reverse order
    for i in range(number_layers - 2, -1, -1):  # skip the input layer

        # get dot product of delta or previous layer with weights of current layer 
        delta = matrix_mult(deltas[-1], zip(*weights[i])) 
        # get sigmoid derivitive of current layer activation vals
        current_layer_sigmoid_activation_vals = []
        for row in activation_vals[i]:
            curr_row = []
            for a_val in row:
                curr_row.append(sigmoid_derivative(a_val))
            current_layer_sigmoid_activation_vals.append(curr_row)
        # multiply the delta by the sigmoid derivitive
        delta = matrix_mult(delta, current_layer_sigmoid_activation_vals)
        deltas.append(delta)
    
    # reverse the order of the delta list 
    for i in range(len(deltas) // 2): 
        index = len(deltas) - i -1
        temp = deltas[i]
        deltas[i] = deltas[index]
        deltas[index] = temp
    return deltas

def update_weights(activation_vals, deltas, weights, biases, learning_rate, number_layers):
    
    for i in range(number_layers):
        
        # multiply current layer activation vals and delta
        weight_update = matrix_mult(zip(*activation_vals[i]), deltas[i])
        
        # set biases to 1
        bias_term = []
        for j in range(len(activation_vals)):
            bias_term.append([1])
        bias_update = matrix_mult(bias_term, deltas[i])
        # update weights based on delta 
        weights[i] = matrix_add(weights[i], matrix_mult(weight_update, [[-learning_rate]]))
        # update biases to the delta 
        biases[i] = matrix_add(biases[i], matrix_mult(bias_update, [[learning_rate]]))

    return weights, biases


def backward_propagation(x, y, activation_vals, weights, biases, number_layers, learning_rate):

    deltas = calculate_deltas(activation_vals, y, weights, number_layers)
    
    weights, biases = update_weights(activation_vals, deltas, weights, biases, learning_rate, number_layers)

    return weights, biases
normal_nn_x = []
normal_nn_loss = []
k_fold_cross_epochs = []
k_fold_cross_error = []

def train(x, y, epochs, learning_rate, number_layers, weights, biases, k_fold_cross, normal_nn):
    total_loss = 0
    with open('./Output_loss.txt', 'a') as f:
        for epoch in range(epochs):
            # get activation vals 
            activation_vals = feed_forward(x, weights, biases, number_layers)
            # update weights and biases
            weights, biases = backward_propagation(x, y, activation_vals, weights, biases, number_layers, learning_rate)
            
            # make y a list instead of list of lists
            y_list = []
            for val in y:
                y_list.append(val[0])
            
            # use mean squared loss 
            total_loss = 0
            for predicted_val, actual_val in zip(activation_vals[-1], y_list):
                total_loss += math.pow(actual_val - predicted_val[0], 2)
            total_loss = total_loss / len(y)
            
            if epoch % 100 == 0:
                f.write(f'Epoch {epoch}, Loss: {total_loss}\n')
                print(f'Epoch {epoch}, Loss: {total_loss}')
                if k_fold_cross:
                    k_fold_cross_epochs.append(epoch)
                    k_fold_cross_error.append(total_loss)
                elif normal_nn: 
                    normal_nn_x.append(epoch)
                    normal_nn_loss.append(total_loss)
    return total_loss

def make_pred(x, weights, biases, number_layers):
    activation_layers = feed_forward(x, weights, biases, number_layers)
    return activation_layers[-1]


number_layers = 5
nodes_per_layer = [10, 7, 2, 5, 7, 1] 

weights, biases = initialize(number_layers, nodes_per_layer) 
train(x, y, epochs=1000, learning_rate=0.001, number_layers=number_layers, weights=weights, biases=biases, k_fold_cross=False, normal_nn=True) 
predictions = make_pred(x, weights, biases, number_layers)
print(predictions[:5])  


# k cross fold validation
k = 10



def k_fold_cross_validation(x, y, k, epochs, learning_rate, number_layers, nodes_per_layer, weights, biases):
    # Split the data into k folds
    fold_size = len(x) // k
    indexes = list(range(len(x)))
    random.shuffle(indexes)  # Shuffle data before splitting into folds
    
    folds = [indexes[i * fold_size:(i + 1) * fold_size] for i in range(k)]
    
    total_x = []
    total_y = []
    for fold in folds:
        curr_x_vals = []
        curr_y_vals = []
        for index in fold:
            curr_x_vals.append(x[index])
            curr_y_vals.append(y[index])
        total_x.append(curr_x_vals)
        total_y.append(curr_y_vals)

    min_loss = 999999999
    min_weight = []
    min_bias = []
    
    for i in range(k):
        x_train = []
        y_train = []
        x_test = total_x[i]
        y_test = total_y[i]

        # Get 75% of data to train on (training is all other folds)
        for j in range(k):
            if j != i:
                x_train.extend(total_x[j])
                y_train.extend(total_y[j])

        # Train the model
        loss = train(x_train, y_train, epochs, learning_rate, number_layers=number_layers, weights=weights, biases=biases, k_fold_cross=False, normal_nn=False)
        if loss < min_loss:
            min_loss = loss
            min_bias = copy.deepcopy(biases)
            min_weight = copy.deepcopy(weights)
    train(x, y, 1000, learning_rate, number_layers, weights=min_weight, biases=min_bias, k_fold_cross=True, normal_nn=False)
    

        
# train with minimum weight and bias

# return x as epoch and y as error 

# track on graph

k_fold_cross_validation(x, y, 50, 100, .001, number_layers=number_layers, nodes_per_layer=nodes_per_layer, weights=weights, biases=biases)


plt.figure(figsize=(10, 6))
plt.plot(normal_nn_x, normal_nn_loss, label="Normal Neural Network Loss", color='blue')

plt.plot(k_fold_cross_epochs, k_fold_cross_error, label="K Fold Cross Validation Loss", color='red')

plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training Loss and K-Fold Cross-Validation Loss')
plt.legend()

plt.show()
