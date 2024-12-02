# Label Encode the `day` column
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
data['day_encoded'] = le.fit_transform(data['day'])

# Save the encoder to a pickle file 
import pickle
with open('label_encoder.pkl', 'wb') as f:
    pickle.dump(le, f)

data.sample(5)