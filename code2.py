import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from PIL.Image import preinit
from matplotlib import ticker
from matplotlib.pyplot import legend
import matplotlib.cm as cm
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

data = pd.read_csv('data.csv', sep=';')
price_data = pd.read_csv('price2.csv', sep=';')

data['Bestelldatum'] = data['Bestelldatum'].fillna('0.0.0').replace('', '0.0.0')
data[['Bestell-Tag', 'Bestell-Monat', 'Bestell-Jahr']] = data['Bestelldatum'].str.split('.', expand=True)
data['Rechnungsdatum'] = data['Rechnungsdatum'].fillna('0.0.0').replace('', '0.0.0')
data[['Rechnungs-Tag', 'Rechnungs-Monat', 'Rechnungs-Jahr']] = data['Rechnungsdatum'].str.split('.', expand=True)

# Count occurrences of each unique value in the column
articles = data['ArtBez'].value_counts()

# Combine small slices
threshold = 500  # Adjust threshold as needed
articles['Other'] = articles[articles < threshold].sum()
articles = articles[articles >= threshold]

# Convert Datum to datetime with error handling
# price_data['Datum'] = pd.to_datetime(price_data['Datum'], format='%d/%m/%Y', errors='coerce')
price_data[['Tag', 'Monat', 'Jahr']] = price_data['Datum'].str.split('/', expand=True)

price_data['Preis'] = pd.to_numeric(price_data['Preis'], errors='coerce')
price_data['Jahr'] = pd.to_numeric(price_data['Jahr'], errors='coerce')
price_data['Tag'] = pd.to_numeric(price_data['Tag'], errors='coerce')
price_data['Monat'] = pd.to_numeric(price_data['Monat'], errors='coerce')

translation = [0, 0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30]

# Iterate over the list, adding each element to its immediate right neighbor
for i in range(len(translation) - 1):  # Stop at the second last element
    translation[i + 1] += translation[i]

price_data['Day_of_Year'] = price_data['Tag'].copy()

# value = price_data.iloc[row_index, column_index]

for row in range(len(price_data['Tag'])):
    price_data.iloc[row, 5] = translation[int(price_data.iloc[row, 3])] + int(price_data.iloc[row, 2])

price_data['Day_of_Year'] = pd.to_numeric(price_data['Day_of_Year'], errors='coerce')

years_trans = [0, 365, 365 * 2, 365 * 3, 365 * 4, 365 * 5, 365 * 6, 365 * 7]
counter = 0

price_data['Total_Days'] = price_data['Day_of_Year'].copy()

for row in range(len(price_data.Total_Days)):

    # print(data.iloc[row, 19] - 2018)
    if price_data.iloc[row, 5] == 0:
        counter += 1
    else:
        price_data.iloc[row, 6] += years_trans[price_data.iloc[row, 4] - 2018]

# palette = sns.color_palette("magma")
# sns.scatterplot(x=price_data.Day_of_Year,y=price_data.Preis,hue=price_data.Jahr)
# plt.ylabel('Preise')
# plt.title('Development of Heating Oil Prices')
# plt.show()


data['Rechnungs-Jahr'] = pd.to_numeric(data['Rechnungs-Jahr'], errors='coerce')
data['Rechnungs-Tag'] = pd.to_numeric(data['Rechnungs-Tag'], errors='coerce')
data['Rechnungs-Monat'] = pd.to_numeric(data['Rechnungs-Monat'], errors='coerce')

data['Day_of_Year'] = data['Rechnungs-Tag'].copy()

for row in range(len(data['Day_of_Year'])):
    data.iloc[row, 20] = translation[int(data.iloc[row, 18])] + int(data.iloc[row, 17])

data['Total_Days'] = data['Day_of_Year'].copy()

counter = 0

for row in range(len(data.Total_Days)):

    # print(data.iloc[row, 19] - 2018)
    if data.iloc[row, 19] == 0:
        counter += 1
    else:
        data.iloc[row, 21] += years_trans[data.iloc[row, 19] - 2018]

data.sort_values(by='Total_Days', ascending=True, inplace=True)

# Create a full range of numbers for the 'Counter' column
full_range = range(price_data['Total_Days'].min(), price_data['Total_Days'].max() + 1)

# Reindex the DataFrame to the full range
price_data = pd.DataFrame({'Total_Days': full_range}).merge(price_data, on='Total_Days', how='left')

# Fill missing values in the 'Value' column by forward filling
price_data['Preis'] = price_data['Preis'].fillna(method='ffill')

# Filter rows where the 'Score' column meets the threshold
data = data[data['Total_Days'] >= 2]
# data.sort_values(by='Bestellmenge', ascending=True, inplace=True)

mapping = price_data[['Total_Days', 'Preis']]

data = data.merge(mapping, on='Total_Days', how='left')

data.sort_values(by='Total_Days', ascending=True, inplace=True)

# Initialize columns for new_id, previous_order_amount, and time_since_last_order
data['ID'] = None
data['previous_order_amount'] = None
data['time_since_last_order'] = None

# Dictionary to track customer details
customer_tracker = {}
current_id = 1

# Iterate through the DataFrame
for index, row in data.iterrows():
    customer = row['KtoNr']
    timestamp = row['Total_Days']
    order_amount = row['VerkaufsMenge']

    if customer not in customer_tracker:
        # Assign a new ID for a new customer
        data.at[index, 'ID'] = current_id
        current_id += 1

        # Initialize tracker details for this customer
        customer_tracker[customer] = {
            'last_order_time': timestamp,
            'last_order_amount': order_amount
        }

        # No previous order details for a new customer
        data.at[index, 'previous_order_amount'] = 0
        data.at[index, 'time_since_last_order'] = None
    else:
        # Existing customer
        last_order_time = customer_tracker[customer]['last_order_time']
        last_order_amount = customer_tracker[customer]['last_order_amount']

        # Update DataFrame with previous order details
        data.at[index, 'ID'] = data[data['KtoNr'] == customer]['ID'].iloc[0]
        data.at[index, 'previous_order_amount'] = last_order_amount
        data.at[index, 'time_since_last_order'] = timestamp - last_order_time  # Time in minutes

        # Update tracker with new order details
        customer_tracker[customer]['last_order_time'] = timestamp
        customer_tracker[customer]['last_order_amount'] = order_amount

# Reset index for clean output
data = data.reset_index(drop=True)
print(data.columns)

original_numeric_data = data[
    ['ID', 'ArtNr', 'Bestellmenge', 'Day_of_Year', 'Total_Days', 'Preis', 'VKP', 'previous_order_amount',
     'time_since_last_order']]

numeric_data = original_numeric_data.dropna()
# numeric_data.sort_values(by='ID', ascending=True, inplace=True)
print(data)
print(numeric_data)

# year = 5
# month = 4
# day = 3
# price = 2
#
# max = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
# min = [9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999]
# x = ['Jan', 'Feb', 'Mar', 'Apr', 'Mai','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
#
# for row in range(len(price_data['Preis'])):
#     if price_data.iloc[row, year] == 2024:
#         if price_data.iloc[row, day] > max[int(price_data.iloc[row, month]) - 1]:
#             max[int(price_data.iloc[row, month]) - 1] = float(price_data.iloc[row, price])
#         if price_data.iloc[row, day] < min[int(price_data.iloc[row, month]) - 1]:
#             min[int(price_data.iloc[row, month]) - 1] = float(price_data.iloc[row, price])
#
# print(max)
# print(min)

# Initialize max and min lists
max_prices = [0] * 12
min_prices = [9999] * 12
months = ['Jan', 'Feb', 'Mar', 'Apr', 'Mai', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

price_data.dropna(inplace=True)
print(price_data)

# Iterate through the data
for row in range(len(price_data)):
    # Extract year, month, and price
    year = int(price_data.iloc[row]['Jahr'])
    month = int(price_data.iloc[row]['Monat'])
    price = float(price_data.iloc[row]['Preis'])

    # Check if the year is 2024
    if year == 2024:
        # Update max and min prices for the corresponding month
        if price > max_prices[month - 1]:  # Month - 1 because months are 1-based
            max_prices[month - 1] = price
        if price < min_prices[month - 1]:  # Month - 1 because months are 1-based
            min_prices[month - 1] = price

# Print the results
print("Max Prices:", max_prices)
print("Min Prices:", min_prices)


# Plot
plt.figure(figsize=(10, 6))
plt.plot(months, min_prices, label='Min Prices', color='green', marker='o')
plt.plot(months, max_prices, label='Max Prices', color='red', marker='o')

# Adding titles and labels
plt.title('Monthly Min and Max Oil Prices 2024', fontsize=16)
plt.xlabel('Month', fontsize=14)
plt.ylabel('Price [€ / 100 l]', fontsize=14)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


save = [0] * 12
for i in range(12):
    save[i] = (max_prices[i] - min_prices[i]) * 20

print(save)

plt.figure(figsize=(10, 6))
plt.plot(months, save, color='green', marker='o')

# Adding titles and labels
plt.title('Saved Money', fontsize=16)
plt.xlabel('Month', fontsize=14)
plt.ylabel('[€]', fontsize=14)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# Filter data where 'Bestellmenge' is less than 10000
filtered_data = data[(data['Bestellmenge'] < 20000)]  # & (data['Bestellmenge'] > 0)

# Extract the column to plot
values_to_plot = filtered_data['Bestellmenge']

# Create the boxplot
plt.figure(figsize=(8, 6))
plt.boxplot(values_to_plot, vert=False, patch_artist=True, boxprops=dict(facecolor='lightblue'))
plt.title('Boxplot of Bestellmenge')
plt.xlabel('Bestellmenge [l]')
plt.grid(True)
plt.show()

# Extract the column to plot
filtered_data = data[(data['time_since_last_order'] > 0)]  # & (data['Bestellmenge'] > 0)
values_to_plot = filtered_data['time_since_last_order']
print(values_to_plot)
values_to_plot.dropna()


def plot():
    norm = plt.Normalize(vmin=price_data['Jahr'].min(), vmax=price_data['Jahr'].max())  # Normalize year values
    cmap = cm.viridis  # Use the 'viridis' colormap
    scatter = plt.scatter(price_data['Day_of_Year'], price_data['Preis'], c=price_data['Jahr'], cmap=cmap, norm=norm,
                          s=10)
    plt.colorbar(scatter, label='Year')

    plt.ylabel('Price [€ / 100 l]')

    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    filter = price_data[price_data['Total_Days'] > 2000]

    # Create the plot
    plt.figure(figsize=(12, 6))
    plt.plot(filter['Total_Days'], filter['Preis'], color='red', linewidth=2, label="Price Development")
    plt.fill_between(filter['Total_Days'], filter['Preis'], color='red', alpha=0.1)  # Shading under the line

    # Customizing the plot
    plt.title("Heizöl Preisentwicklung", fontsize=16, loc='left')
    plt.xlabel("")
    plt.ylabel("€ per 100 l")
    plt.ylim(70,100)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.grid(visible=True, which='major', linestyle='--', alpha=0.5)
    plt.tight_layout()

    # Show the plot
    plt.show()

    scatter = plt.scatter(price_data['Total_Days'], price_data['Preis'], c=price_data['Jahr'], cmap=cmap, norm=norm,
                          s=10)
    plt.colorbar(scatter, label='Year')

    plt.ylabel('Price [€ / 100 l]')
    plt.xlabel('Days from 2018 - 2025')
    plt.title('Development of Heating Oil Prices')

    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # Plot a pie chart
    plt.figure(figsize=(8, 6))
    articles.plot.pie(startangle=90, legend=False, cmap='viridis')

    # Enhance plot aesthetics
    plt.title('Ordered Products')
    plt.ylabel('')  # Remove default y-label
    # plt.legend(loc='upper left')
    plt.show()

    #####################################################
    plt.scatter(x=data.ID, y=data.previous_order_amount)
    plt.show()

    customers = data['ID'].value_counts()
    frequency = customers.value_counts()

    # Plot the result
    plt.figure(figsize=(8, 6))
    frequency.plot(kind='bar', color='skyblue')
    plt.title('Frequency of Customerstyle')
    plt.xlabel('how often customers bought oil at Rödl')
    plt.ylabel('Frequency')
    plt.xlim(0, 20)
    plt.xticks(rotation=0)  # Keep the x-axis labels horizontal
    plt.show()


def PCA():
    numeric_data = original_numeric_data.dropna()
    scaler = StandardScaler()
    numeric_data = scaler.fit_transform(numeric_data)
    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(numeric_data)
    pca_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])

    pca_df.sort_values(by='PC2', ascending=True, inplace=True)
    # pca_df = pca_df[pca_df['PC2'] >= -50]

    axis_1 = 'PC1'
    axis_2 = 'PC2'
    title = 'PCA Plot'

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(
        pca_df[axis_1],
        pca_df[axis_2],
        c=original_numeric_data['Preis'],  # Color by continuous values
        cmap='viridis',  # Choose a colormap
        s=50
    )
    plt.colorbar(scatter, label='Preis')
    plt.xlabel(axis_1)
    plt.ylabel(axis_2)
    plt.title(title)
    plt.show()

    print(pca_df)


def ml():
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_squared_error, r2_score
    import matplotlib.pyplot as plt

    ml_data = original_numeric_data.copy()
    ml_data.dropna()
    X = ml_data[['ID', 'ArtNr', 'Bestellmenge', 'Day_of_Year', 'Preis', 'VKP', 'previous_order_amount']]
    print(X)  # Features (independent variable)
    y = ml_data['time_since_last_order']  # Target (dependent variable)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)  # Mean Squared Error
    r2 = r2_score(y_test, y_pred)  # R^2 score
    print(f"Mean Squared Error: {mse}")
    print(f"R^2 Score: {r2}")
    plt.scatter(X_test, y_test, color='blue', label='Actual')
    plt.plot(X_test, y_pred, color='red', label='Predicted')
    plt.title('Linear Regression: Actual vs Predicted')
    plt.xlabel('Feature')
    plt.ylabel('Target')
    plt.legend()
    plt.show()

plot()