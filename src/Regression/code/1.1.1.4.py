# Linear model
def linear_model(x, w, b):
    return w * x + b

y_pred = [linear_model(p, w, b) for p in x]

plt.plot(x, y_data, 'o')
plt.plot(x, y_true, label='Best Fit')
plt.plot(x, y_pred, label='Prediction')
plt.legend()
plt.show()