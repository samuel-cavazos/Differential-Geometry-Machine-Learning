# One-Hot encode the `day` column
from sklearn.preprocessing import OneHotEncoder

ohe = OneHotEncoder(sparse_output=False,categories=[['Thur', 'Fri', 'Sat', 'Sun']])
day_ohe = ohe.fit_transform(data[['day']])
day_ohe_df = pd.DataFrame(day_ohe, columns=[f'day_{day}' for day in ohe.categories[0]])
# Merge the one-hot encoded columns with the original dataframe
data = pd.concat([data, day_ohe_df], axis=1)

print(data.sample(5).to_latex(index=False, float_format="%.2f"))