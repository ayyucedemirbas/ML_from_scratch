import numpy as np
import os
from urllib import request
import time

class MNISTDataLoader:
    def __init__(self, filename="mnist.npz"):
        self.filename = filename
        self.url = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz"

    def download(self):
        if not os.path.exists(self.filename):
            print(f"Downloading MNIST data from {self.url}...")
            request.urlretrieve(self.url, self.filename)
            print("Download complete!")
        else:
            print("MNIST data file already exists.")

    def one_hot_encode(self, y, num_classes=10):
        m = y.shape[0]
        one_hot = np.zeros((m, num_classes))
        for i in range(m):
            one_hot[i, y[i]] = 1
        return one_hot

    def load_and_process(self):
        self.download()
        with np.load(self.filename, allow_pickle=True) as f:
            x_train, y_train = f['x_train'], f['y_train']
            x_test, y_test = f['x_test'], f['y_test']

        flat_x_train = x_train.reshape(x_train.shape[0], -1)
        flat_x_test = x_test.reshape(x_test.shape[0], -1)

        norm_x_train = flat_x_train.astype(np.float32) / 255.0
        norm_x_test = flat_x_test.astype(np.float32) / 255.0

        enc_y_train = self.one_hot_encode(y_train)
        enc_y_test = self.one_hot_encode(y_test)

        return norm_x_train, enc_y_train, norm_x_test, enc_y_test


class MNISTClassifier:
    def __init__(self, input_size=784, hidden_size=128, output_size=10):
        # He Initialization (Best for ReLU)
        self.W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2. / input_size)
        self.b1 = np.zeros((1, hidden_size))
        
        self.W2 = np.random.randn(hidden_size, output_size) * np.sqrt(2. / hidden_size)
        self.b2 = np.zeros((1, output_size))
        
        # Adam Optimizer Parameters
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.epsilon = 1e-8
        self.lr = 0.001
        self.t = 0 # Time step counter
        
        # Momentum (m) and RMSProp (v) moving averages
        self.m = {'W1': 0, 'b1': 0, 'W2': 0, 'b2': 0}
        self.v = {'W1': 0, 'b1': 0, 'W2': 0, 'b2': 0}

    def relu(self, Z):
        return np.maximum(0, Z)

    def softmax(self, Z):
        exp_Z = np.exp(Z - np.max(Z, axis=1, keepdims=True))
        return exp_Z / np.sum(exp_Z, axis=1, keepdims=True)

    def forward(self, X):
        self.Z1 = np.dot(X, self.W1) + self.b1
        self.A1 = self.relu(self.Z1)
        
        self.Z2 = np.dot(self.A1, self.W2) + self.b2
        self.A2 = self.softmax(self.Z2)
        
        return self.A2

    def compute_loss(self, Y_pred, Y_true):
        m = Y_true.shape[0]
        Y_pred = np.clip(Y_pred, 1e-15, 1 - 1e-15)
        loss = -np.sum(Y_true * np.log(Y_pred)) / m
        return loss

    def backward(self, X, Y_true):
        m = X.shape[0]
        
        # Output Layer Gradients
        # Derivative of Softmax + CrossEntropy simplifies to (Prediction - Target)
        dZ2 = self.A2 - Y_true
        dW2 = (1 / m) * np.dot(self.A1.T, dZ2)
        db2 = (1 / m) * np.sum(dZ2, axis=0, keepdims=True)
        
        # Hidden Layer Gradients
        dA1 = np.dot(dZ2, self.W2.T)
        dZ1 = dA1 * (self.Z1 > 0) # Derivative of ReLU
        dW1 = (1 / m) * np.dot(X.T, dZ1)
        db1 = (1 / m) * np.sum(dZ1, axis=0, keepdims=True)
        
        return {'W1': dW1, 'b1': db1, 'W2': dW2, 'b2': db2}

    def update_parameters_adam(self, grads):
        self.t += 1
        params = {'W1': self.W1, 'b1': self.b1, 'W2': self.W2, 'b2': self.b2}
        
        for key in params.keys():
            g = grads[key]
            
            self.m[key] = self.beta1 * self.m[key] + (1 - self.beta1) * g

            self.v[key] = self.beta2 * self.v[key] + (1 - self.beta2) * (g ** 2)
            
            m_hat = self.m[key] / (1 - self.beta1 ** self.t)
            
            v_hat = self.v[key] / (1 - self.beta2 ** self.t)
            
            params[key] -= self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)

        self.W1, self.b1 = params['W1'], params['b1']
        self.W2, self.b2 = params['W2'], params['b2']


def create_mini_batches(X, Y, batch_size=64):
    m = X.shape[0]
    indices = np.arange(m)
    np.random.shuffle(indices)
    
    X_shuffled = X[indices]
    Y_shuffled = Y[indices]
    
    for i in range(0, m, batch_size):
        yield X_shuffled[i:i + batch_size], Y_shuffled[i:i + batch_size]

if __name__ == "__main__":
    loader = MNISTDataLoader()
    X_train, Y_train, X_test, Y_test = loader.load_and_process()
    
    input_dim = 784
    hidden_dim = 128
    output_dim = 10
    
    model = MNISTClassifier(input_dim, hidden_dim, output_dim)
    
    epochs = 10
    batch_size = 64
    learning_rate = 0.001
    model.lr = learning_rate

    start_time = time.time()

    for epoch in range(epochs):
        epoch_loss = 0
        num_batches = 0
        
        for X_batch, Y_batch in create_mini_batches(X_train, Y_train, batch_size):
            predictions = model.forward(X_batch)
            
            loss = model.compute_loss(predictions, Y_batch)
            epoch_loss += loss
            num_batches += 1
            
            grads = model.backward(X_batch, Y_batch)
            
            model.update_parameters_adam(grads)
            
        avg_loss = epoch_loss / num_batches
        print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f}")

    training_time = time.time() - start_time
    print(f"Time: {training_time:.2f} seconds")


    test_preds = model.forward(X_test)
    pred_labels = np.argmax(test_preds, axis=1)
    true_labels = np.argmax(Y_test, axis=1)
    
    accuracy = np.mean(pred_labels == true_labels)
    print(f"Final Test Accuracy: {accuracy:.2%}")
    
    print(f"Example -> Model Prediction: {pred_labels[0]}, True Label: {true_labels[0]}")
