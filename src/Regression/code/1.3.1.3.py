# Ordinal encode the `day` column
from sklearn.preprocessing import OrdinalEncoder
oe = OrdinalEncoder(categories=[['Thur', 'Fri', 'Sat', 'Sun']])
data['day_OrdinalEncoded'] = oe.fit_transform(data[['day']])

# Save the encoder to a pickle file
with open('ordinal_encoder.pkl', 'wb') as f:
    pickle.dump(oe, f)

data.sample(5)