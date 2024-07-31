import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import linear_model
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error

# dataset read
df = pd.read_csv('train_data.csv')


# filters
df = df[['duration','days_left','stops','airline','class','price']]


# numeric-encoder: word -> numeric
def df_numeric_encoder(df, is_train=True,category_airline=None):
    if is_train:
        price_sum = df.groupby('airline')['price'].sum()
        days_left_sum = df.groupby('airline')['days_left'].sum()
        category_airline = (price_sum / days_left_sum).sort_values().index
    category_stops = ['zero', 'one', 'two_or_more']
    category_class = [ 'Economy', 'Business']

    encoder = OrdinalEncoder(categories=[category_airline, category_stops, category_class])
    mask = ['airline', 'stops', 'class']
    df[mask] = encoder.fit_transform(df[mask])
    
    if is_train:
        return df, category_airline
    return df

df,category_airline = df_numeric_encoder(df)


# split categories
def split_categories(df, categories, column=None, original_df=None):
    new_df = []

    if column is None:
        new_df = [df]

    if column:
        column_unique_values = np.sort(original_df[column].unique())
        for df_part in df:
            for col_val in column_unique_values:
                new_df.append(df_part[df_part[column] == col_val])

    if len(categories) >= 1:
        return split_categories(new_df, categories[1:], categories[0], original_df)
    
    return new_df


categories = ['class', 'airline', 'stops']
df = split_categories(df, categories, original_df=df)


# drop empty arrays
def drop_empty_arrays(df, is_train=True):
    non_empty_arrays = [df_part for df_part in df if len(df_part) > 0]
    MLR_models_index = [i for i, df_part in enumerate(df) if len(df_part) > 0]

    if is_train:
        return non_empty_arrays, MLR_models_index
    return non_empty_arrays


df,MLR_models_index = drop_empty_arrays(df)


# correlation
correlation = []
print('\n\n\nCORRELATION:')

for df_part in df: 
    correlation.append(df_part.corrwith(df_part['price']).abs().sort_values(ascending=False))
    print(f'{correlation[-1].index[1]} {correlation[-1].iloc[1]}')



# # GRAFIK 
# print('\n\n\GRAPHICS:')

# for i, df_part in enumerate(df):
#     plt.figure(figsize=(10,6))
#     sns.scatterplot(data=df_part, x=correlation[i].index[1], y='price', color='blue')
#     sns.regplot(data=df_part, x=correlation[i].index[1], y='price', scatter=False, color='red')
#     plt.show()


# MultiLinear
MLR_models = []
x_train = []
y_train = []
print("\n\n\nCOEFFICIENTS and THETA0")

for df_part in df:
    x_train.append(df_part[['duration','days_left','stops','airline','class']])
    y_train.append(df_part[['price']])

    MLR_models.append(linear_model.LinearRegression())
    MLR_models[-1].fit(x_train[-1], y_train[-1])
    # print(f'Coefficients:  {MLR_models[-1].coef_}')
    print(f'theta0: {MLR_models[-1].intercept_}') 






######################## RESULTS
# TRAIN: MAE, RMSE
print('\n\n\nTRAIN: MAE, RMSE')
MAE = []

for i, df_part in enumerate(df):
    try:
        y_pred = MLR_models[i].predict(x_train[i])

        MAE.append(mean_absolute_error(y_train[i], y_pred))
        RMSE = np.sqrt(mean_squared_error(y_train[i], y_pred))
        print(f"{MAE[-1]=}")
        # print(f"{RMSE=}")
    except:
        continue

print(f'\n\n\nMEAN={np.mean(MAE)}')
print([int(x) for x in MAE])
print([len(x) for x in df])


# Result Accuracy
accuracies = []

for i, x in enumerate(MAE):
    acc = 100 - ((x / (np.sum(df[i]['price']) / len(df[i]))) * 100)
    df_length = np.sum([len(x) for x in df])
    acc_length = len(df[i])
    acc = (acc_length / df_length) * acc
    accuracies.append(acc)

print(f'\n\n\nModel aniqligi: {np.sum(accuracies):.2f}%')




# SAVE
# test_data.csv -> predict -> submission.csv
df_test = pd.read_csv('test_data.csv')
df_test = df_numeric_encoder(df_test, False, category_airline)
df_test = split_categories(df_test, categories, original_df=df_test)
df_test = [df_test[i] for i in MLR_models_index]


x_test = []
for df_part in df_test:
    x_test.append(df_part[['duration','days_left','stops','airline','class']])


y_pred = []
for i, df_part in enumerate(df_test):
    y_pred.append(MLR_models[i].predict(x_test[i]))


data = {
    'id': [id for part in df_test for id in part['id']],
    'price': [price[0] for part in y_pred for price in part],
}


df_sample_solution = pd.DataFrame(data)
df_sample_solution = df_sample_solution.sort_values(by='id').reset_index(drop=True)
df_sample_solution.to_csv('submission.csv', index=False)