import parse_data 
from sklearn.tree import DecisionTreeRegressor
import random
from math import ceil
from statistics import mean
from models.knn import KNNModel
import matplotlib.pyplot as plt

# Finds the MSE. Lower values mean indicate better performance, meaning 
# the model predicts closer to the actual price
def mean_squared_error(y_true, y_pred):
    total = 0.0
    for true_val, pred_val in zip(y_true, y_pred):
        total += (true_val[0] - pred_val) ** 2 
    return total / len(y_true)

# Finds squared error. 1 means the mode gives a perfect prediction. If it equals 0 the model doesn't 
# do very well. If the squared error is less than 0 then the model sucks and needs to be retrained
def r2_score(y_true, y_pred):
    y_true_flat = [val[0] for val in y_true]    
    mean_y = sum(y_true_flat) / len(y_true_flat)
    
    # denominator
    total_sum_of_squares = sum((y - mean_y) ** 2 for y in y_true_flat)
    
    # numerator
    residual_sum_of_squares = sum((y_t - y_p) ** 2 for y_t, y_p in zip(y_true_flat, y_pred))
    
    r2 = 1 - (residual_sum_of_squares / total_sum_of_squares)
    return r2

# Runs k-fold validation, default number of chunsks is 5. Expects to be passed a model function
# that follows the format as model_interface.py
def kfold_validation(n_folds=5, model_function = None):
    df = parse_data.df # Get all the data
    
    # Features we are using to predict
    X = df[['Company Name', 'Model Name', 'Mobile Weight', 'RAM', 'Front Camera', 'Back Camera',
            'Processor', 'Battery Capacity', 'Screen Size', 'Launched Year']].values.tolist()
    
    # What we are trying to predict
    y = df[['Launched Price (USA)']].values.tolist()

    # uncomment this if you want to test for all the countries
    # y = df[['Launched Price (USA)', 'Launched Price (Dubai)', 'Launched Price (China)',
    #         'Launched Price (India)', 'Launched Price (Pakistan)']].values.tolist()

    # Mix around data before we seperate it into chunks
    data = list(zip(X, y))
    random.seed(42)
    random.shuffle(data)

    # Divide into chunks
    fold_size = ceil(len(data) / n_folds)
    models = []
    mse_scores = []
    r2_scores = []

    # For each chunk train the model and validate on R2 and MSE
    for i in range(n_folds):
        val_data = data[i * fold_size : (i + 1) * fold_size]
        train_data = data[:i * fold_size] + data[(i + 1) * fold_size:]

        X_train, y_train = zip(*train_data)
        X_val, y_val = zip(*val_data)

        # First instantiate the model
        if model_function is None: 
            print("No model found, using scikit-learn Decision Tree Regressor instead")
            model = DecisionTreeRegressor()
        else: 
            model = model_function()
            print(f"Using model with name: {model.__class__.__name__}")
        model = model_function()
        model.fit(X_train, y_train) 
        models.append(model)
        y_pred = model.predict(X_val)

        # Calculate MSE
        mse = mean_squared_error(y_val, y_pred)
        mse_scores.append(mse)
        print(f"Fold {i+1} Model: MSE = {mse:.2f}")

        # Calculate R2 error
        r2 = r2_score(y_val, y_pred)
        r2_scores.append(r2)
        print(f"R-squared {i+1} Model: R2 = {r2:.2f}\n")

    print("\nResults:")
    print(f"Best MSE score: Model {mse_scores.index(min(mse_scores))+1}")
    print(f"Best R2 score: Model {r2_scores.index(max(r2_scores))+1}\n")

    return models, mse_scores, r2_scores

# Runs tests for knn. Will plot points for the average MSE and R2 over 5 models using k-fold validation
def test_knn(upper_limit=15):
    mse_scores_per_k = []
    r2_scores_per_k = []

    for k in range(1, upper_limit+1, 2):
        print(f"Training with k={k}")
        models, mse_scores, r2_scores = kfold_validation(model_function=lambda: KNNModel(k=k))

        # Calculate average MSE and R² for the current k
        avg_mse = mean(mse_scores)
        mse_scores_per_k.append(avg_mse)
        r2_avg = mean(r2_scores)
        r2_scores_per_k.append(r2_avg)

        print(f"Average MSE for k={k}: {avg_mse}")
        print(f"Average R² for k={k}: {r2_avg}\n")

    k_values = list(range(1, upper_limit+1, 2))

    plt.figure(figsize=(12, 6))

    # Plot MSE
    plt.subplot(1, 2, 1)
    plt.plot(k_values, mse_scores_per_k, marker='o', label='MSE', color='blue')
    plt.xlabel('k value')
    plt.ylabel('MSE')
    plt.title('MSE vs k value')
    plt.grid(True)

    # Plot R2
    plt.subplot(1, 2, 2)
    plt.plot(k_values, r2_scores_per_k, marker='o', label='R²', color='red')
    plt.xlabel('k value')
    plt.ylabel('R2')
    plt.title('R2 vs k value')

    plt.grid(True)
    plt.tight_layout()
    plt.savefig('data/knn/knn_plot.png') 



# ==============================================
# ============== Testing the data ==============
# ==============================================

test_knn(upper_limit=15)
