# Activation function and its derivative
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_derivative(z):
    return sigmoid(z) * (1 - sigmoid(z))

# Plot 
z = np.linspace(-10, 10, 100)

plt.plot(z, sigmoid(z), label='sigmoid')
plt.plot(z, sigmoid_derivative(z), label='sigmoid derivative')
plt.legend()

plt.show()
