import parse_data 
from sklearn.tree import DecisionTreeRegressor
import random
from math import ceil
from statistics import mean

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

def kfold_validation(n_folds=5):
    df = parse_data.df # Get all the data
    
    # Features we are using to predict
    X = df[['Company Name', 'Model Name', 'Mobile Weight', 'RAM', 'Front Camera', 'Back Camera',
            'Processor', 'Battery Capacity', 'Screen Size', 'Launched Year']].values.tolist()
    
    # What we are trying to predict
    y = df[['Launched Price (USA)']].values.tolist()

    # uncomment this if you want to test for all the countries
    # y = df[['Launched Price (USA)', 'Launched Price (Dubai)', 'Launched Price (China)',
    #         'Launched Price (India)', 'Launched Price (Pakistan)']].values.tolist()

    # Shuffle data
    data = list(zip(X, y))
    random.seed(42)
    random.shuffle(data)

    fold_size = ceil(len(data) / n_folds)
    models = []
    mse_scores = []
    r2_scores = []

    for i in range(n_folds):
        val_data = data[i * fold_size : (i + 1) * fold_size]
        train_data = data[:i * fold_size] + data[(i + 1) * fold_size:]

        X_train, y_train = zip(*train_data)
        X_val, y_val = zip(*val_data)

        # Train the model here
        model = DecisionTreeRegressor()
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
    print(f"Best MSE score: Model {mse_scores.index(min(mse_scores))}")
    print(f"Best R2 score: Model {r2_scores.index(max(r2_scores))}\n")

    return models, mse_scores


example = [[8, 433, 185.0, 12.0, 16.0, 50.0, 143, 5000.0, 6.7, 2023]]  # Example: OnePlus 11 256GB
models, mse_scores = kfold_validation()

for i, model in enumerate(models):
    prediction = model.predict(example)[0]
    print(f"Model {i+1} prediction")
    print("Predicted USA price: USD", prediction)
    # print("Predicted Dubai price: AED", pred_dubai)
    # print("Predicted China price: CNY", pred_china)
    # print("Predicted India price: INR", pred_india)
    # print("Predicted Pakistan price: PKR", pred_pakistan)
