# Mean Squared Error Loss
def mse_loss(y_pred, y_true):
    return np.mean((y_pred - y_true)**2)

# Compute predictions for the training data
y_pred_train = [linear_model(p, w, b) for p in x_train]

print(f'50th sample target: {y_train[50]}')
print(f'50th prediction: {y_pred_train[50]}')
print(f'Loss at 50th sample: {mse_loss(y_pred_train[50], y_train[50])}')

print('Total Loss over all samples:', mse_loss(np.array(y_pred_train), np.array(y_train)))