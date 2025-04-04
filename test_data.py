import parse_data
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

# This file is just used for testing our data and making sure it works before we try to implement anything from scratch

df = parse_data.df

# This is just testing the data using sklearn to make sure our data works
# X is the labels we use to predict. See the mobile_data.csv for all possible inputs
# Y is what we are trying to predict. We are mostly interested in seeing what the launched price will be for phones
X = df[['RAM', 'Front Camera', 'Back Camera', 'Battery Capacity', 'Screen Size']]
y = df['Launched Price (USA)']

# Using sklearn for now just to test things

# --- Train/Test Split ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = DecisionTreeRegressor()
model.fit(X_train, y_train)

# Testing our model using random input
example = [[8, 12, 48, 4500, 6.7]]  # RAM, Front, Back, Battery, Screen (Random values based on our X labels)
prediction = model.predict(example)
print("Predicted price:", prediction[0])