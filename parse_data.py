import pandas as pd
import re
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv("mobile_data.csv")

# Parsing functions
def parse_price(price):
    if isinstance(price, str) and price.lower() != 'Not available':
        numeric = re.sub(r'[^\d]', '', price) # Remove all values that arent numbers
        return float(numeric) if numeric else None # We have some instances of "Not available"
    return None # If the price is not available

def parse_battery(battery):
    if isinstance(battery, str):
        return float(re.sub(r'[^\d.]', '', battery))
    return battery

def strip_units(val):
    if isinstance(val, str):
        match = re.search(r'\d+(\.\d+)?', val)  # find the first float like number
        if match:
            return float(match.group())
    return None

# Create columns
df['Battery Capacity'] = df['Battery Capacity'].apply(parse_battery)
df['Launched Price (USA)'] = df['Launched Price (USA)'].apply(parse_price)
df['Launched Price (Pakistan)'] = df['Launched Price (Pakistan)'].apply(parse_price)
df['Launched Price (India)'] = df['Launched Price (India)'].apply(parse_price)
df['Launched Price (China)'] = df['Launched Price (China)'].apply(parse_price)
df['Launched Price (Dubai)'] = df['Launched Price (Dubai)'].apply(parse_price)
df['RAM'] = df['RAM'].apply(strip_units)
df['Front Camera'] = df['Front Camera'].apply(strip_units)
df['Back Camera'] = df['Back Camera'].apply(strip_units)
df['Mobile Weight'] = df['Mobile Weight'].apply(strip_units)
df['Screen Size'] = df['Screen Size'].apply(strip_units)
df.dropna(inplace=True) # Remove all NaN values

# Encode labels for company name and processors
# Ex: Apple -> 0, Samsung -> 1, etc. so the model can read it better
label_encoders = {}
for col in ['Company Name', 'Processor']:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le
 
# # This will print out each possible column
# for column in df.columns:
#     print(f"\n\n\nColumn: {column}")
#     print(df[column])
 
