# Linear model
def linear_model(x, w, b):
    return x * w + b

y_pred = [linear_model(p, w, b) for p in x]

plt.figure(figsize=(10, 5))
plt.plot(x, y_true, color='black', label='True function')
plt.scatter(x_train, y_train, color='darkred', label='Training data')
plt.scatter(x_test, y_test, color='blue', label='Test data')
plt.plot(x, y_pred, color='green', label='Initial model')
plt.legend()
plt.savefig('fig2.png')
plt.show()