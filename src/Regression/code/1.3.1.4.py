# Import iris dataset using seaborn
import seaborn as sns
data = sns.load_dataset('tips')

# One-Hot encode the `day` column
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
ohe = OneHotEncoder(sparse_output=False,categories=[['Thur', 'Fri', 'Sat', 'Sun']])
day_ohe = ohe.fit_transform(data[['day']])
day_ohe_df = pd.DataFrame(day_ohe, columns=[f'day_{day}' for day in ohe.categories[0]])
# Convert the one-hot encoded columns to integers
day_ohe_df = day_ohe_df.astype(int)
# Merge the one-hot encoded columns with the original dataframe
data = pd.concat([data, day_ohe_df], axis=1)

# LabelEncode the `sex` column
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
data['sex_encoded'] = le.fit_transform(data['sex'])
# Print the transformation of the `sex` column
print('sex encoding:')
print(f"Male: {le.transform(['Male'])} Female: {le.transform(['Female'])}\n")

# LabelEncode the `smoker` column
data['smoker_encoded'] = le.fit_transform(data['smoker'])
# Print the transformation of the `smoker` column
print('smoker encoding:')
print(f"Yes: {le.transform(['Yes'])} No: {le.transform(['No'])}\n")

# LabelEncode the `time` column
data['time_encoded'] = le.fit_transform(data['time'])
# Print the transformation of the `time` column
print('time encoding:')
print(f"Lunch: {le.transform(['Lunch'])} Dinner: {le.transform(['Dinner'])} \n")

features = ['total_bill','sex_encoded','smoker_encoded','time_encoded','day_Thur','day_Fri','day_Sat','day_Sun','size']
target = 'tip'

print(data[features + [target]].sample(5).to_latex())