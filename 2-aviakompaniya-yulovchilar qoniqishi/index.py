#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import RocCurveDisplay, roc_auc_score, roc_curve, auc,confusion_matrix


# dataset read
df = pd.read_csv('train_dataset.csv')
print(df.info())


# FILL NULLS
print("\n\n\nFILL NULLS")
print(df.isnull().sum())
df['Arrival Delay in Minutes'] = df['Arrival Delay in Minutes'].fillna(df['Departure Delay in Minutes'])


# GRAPHIC
fig, axes = plt.subplots(4,1, figsize=(10,24))
x_axis = ['Gender','Customer Type','Type of Travel','Class']

for i,x in enumerate(x_axis):
    sns.countplot(x=x, hue='satisfaction', palette='viridis', data=df, ax=axes[i])
plt.show()


# numeric-encoder: word -> numeric
def df_numeric_encoder(df, category_gender=None,category_customer_type=None,category_type_of_trafel=None,category_class=None):
    category_gender = ['Male', 'Female']
    category_customer_type = ['disloyal Customer', 'Loyal Customer']
    category_type_of_trafel = ['Personal Travel', 'Business travel']
    category_class = ['Eco', 'Eco Plus', 'Business']

    encoder = OrdinalEncoder(categories=[category_gender, category_customer_type, category_type_of_trafel, category_class])
    mask = ['Gender', 'Customer Type', 'Type of Travel', 'Class']
    df[mask] = encoder.fit_transform(df[mask])
    
    return df

df = df_numeric_encoder(df)


# standard scaler
scaler_mask = [ 'Flight Distance', 'Inflight wifi service',
        'Departure/Arrival time convenient', 'Ease of Online booking',
            'Food and drink', 'Online boarding', 'Seat comfort',
        'Inflight entertainment', 'On-board service', 'Leg room service',
        'Baggage handling', 'Checkin service', 'Inflight service',
        'Cleanliness', 'Departure Delay in Minutes', 'Arrival Delay in Minutes']

def standard_scaler(df):
    scaler = StandardScaler()
    return scaler.fit_transform(df[scaler_mask])

df[scaler_mask] = standard_scaler(df)


# CORRELATION
print("\n\n\nCORRELATION")
print(df.corrwith(df['satisfaction']).abs().sort_values(ascending=False))


# LogisticRegression
print("\n\n\nLogisticRegression Predictions")

x_train = df.drop(columns=['id', 'satisfaction'])
y_train = df[['satisfaction']]

LR_model = LogisticRegression(max_iter=10000)
LR_model.fit(x_train, y_train)

# y_train_pred = LR_model.predict_proba(x_train)[:, 1]
y_train_pred = LR_model.predict(x_train)
roc_auc = roc_auc_score(y_train, y_train_pred)

# Compute ROC curve
fpr, tpr, thresholds = roc_curve(y_train, y_train_pred)
roc_auc = auc(fpr, tpr)

# ROC curve
display = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc, estimator_name='ROC curve')
display.plot()
plt.show()

# Create the confusion matrix
conf_mat = confusion_matrix(y_train, y_train_pred)

# Visualize the confusion matrix using seaborn
sns.heatmap(conf_mat, annot=True, fmt="d", cmap="Blues")
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

print(f"ROC AUC Score: {roc_auc:.2f}")


# SAVE
# test_data.csv -> predict -> submission.csv
df_test = pd.read_csv('test_dataset.csv')
df_test = df_numeric_encoder(df_test)
df_test['Arrival Delay in Minutes'] = df_test['Arrival Delay in Minutes'].fillna(df['Departure Delay in Minutes'])
df_test[scaler_mask] = standard_scaler(df_test)


x_test = df_test.drop(columns=['id'])
y_test_pred = LR_model.predict(x_test)


df_sample = pd.read_csv('sample_submission.csv')
df_sample['satisfaction'] = y_test_pred
df_sample.to_csv('submission.csv', index=False)
# %%
