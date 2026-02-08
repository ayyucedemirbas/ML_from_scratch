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
            print(f"Downloading MNIST data...")
            request.urlretrieve(self.url, self.filename)

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

        # Reshape to (Batch, Channel, H, W)
        x_train = x_train.reshape(-1, 1, 28, 28).astype(np.float32) / 255.0
        x_test = x_test.reshape(-1, 1, 28, 28).astype(np.float32) / 255.0

        y_train = self.one_hot_encode(y_train)
        y_test = self.one_hot_encode(y_test)
        
        return x_train, y_train, x_test, y_test


class Conv3x3:
    def __init__(self, num_filters, input_channels=1):
        self.num_filters = num_filters
        self.input_channels = input_channels
        self.filters = np.random.randn(num_filters, input_channels, 3, 3) * 0.1
        self.bias = np.zeros((num_filters, 1))
        self.last_input = None

    def forward(self, input_data):
        self.last_input = input_data
        batch, channels, h, w = input_data.shape
        h_out, w_out = h - 2, w - 2
        output = np.zeros((batch, self.num_filters, h_out, w_out))

        for i in range(h_out):
            for j in range(w_out):
                region = input_data[:, :, i:i+3, j:j+3]
                for f in range(self.num_filters):
                    output[:, f, i, j] = np.sum(region * self.filters[f], axis=(1, 2, 3))
        
        return output + self.bias.reshape(1, -1, 1, 1)

    def backward(self, d_L_d_out, learning_rate):
        # d_L_d_out shape: (Batch, Filters, H_out, W_out)
        h_out, w_out = d_L_d_out.shape[2], d_L_d_out.shape[3]
        
        d_L_d_filters = np.zeros(self.filters.shape)
        d_L_d_bias = np.sum(d_L_d_out, axis=(0, 2, 3)).reshape(-1, 1)
        
        for i in range(h_out):
            for j in range(w_out):
                region = self.last_input[:, :, i:i+3, j:j+3]
                for f in range(self.num_filters):
                    d_out_pixel = d_L_d_out[:, f, i, j][:, None, None, None]
                    d_L_d_filters[f] += np.sum(region * d_out_pixel, axis=0)

        # GRADIENT CLIPPING: Prevents gradients from becoming too large
        d_L_d_filters = np.clip(d_L_d_filters, -1.0, 1.0)
        d_L_d_bias = np.clip(d_L_d_bias, -1.0, 1.0)

        self.filters -= learning_rate * d_L_d_filters
        self.bias -= learning_rate * d_L_d_bias

class MaxPool2:
    def forward(self, input_data):
        self.last_input = input_data
        batch, filters, h, w = input_data.shape
        h_out, w_out = h // 2, w // 2
        output = np.zeros((batch, filters, h_out, w_out))
        
        for i in range(h_out):
            for j in range(w_out):
                region = input_data[:, :, i*2:(i*2)+2, j*2:(j*2)+2]
                output[:, :, i, j] = np.amax(region, axis=(2, 3))
        return output

    def backward(self, d_L_d_out):
        input_data = self.last_input
        batch, filters, h, w = input_data.shape
        d_L_d_input = np.zeros(input_data.shape)
        h_out, w_out = d_L_d_out.shape[2], d_L_d_out.shape[3]
        
        for i in range(h_out):
            for j in range(w_out):
                region = input_data[:, :, i*2:(i*2)+2, j*2:(j*2)+2]
                max_val = np.amax(region, axis=(2, 3), keepdims=True)
                mask = (region == max_val)
                d_L_d_input[:, :, i*2:(i*2)+2, j*2:(j*2)+2] += d_L_d_out[:, :, i:i+1, j:j+1] * mask
        return d_L_d_input

class CNN:
    def __init__(self):
        self.conv = Conv3x3(num_filters=8)
        self.pool = MaxPool2()
        
        # Output after Conv(26x26) -> Pool(13x13) -> 8 filters = 8*13*13 = 1352
        self.flat_size = 8 * 13 * 13
        
        # Dense Weights (He Initialization)
        self.W_fc = np.random.randn(self.flat_size, 10) * np.sqrt(2. / self.flat_size)
        self.b_fc = np.zeros((1, 10))

    def forward(self, X):

        out = self.conv.forward(X)
        out = np.maximum(0, out)

        out = self.pool.forward(out)
        
        self.pool_output_shape = out.shape

        out_flat = out.reshape(out.shape[0], -1)
        self.flat_input = out_flat
        
        z = np.dot(out_flat, self.W_fc) + self.b_fc
        
        z_shifted = z - np.max(z, axis=1, keepdims=True)
        exp_z = np.exp(z_shifted)
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def train_step(self, X, y_true, lr=0.005):
        m = X.shape[0]
        
        y_pred = self.forward(X)
        
        y_pred_clipped = np.clip(y_pred, 1e-15, 1 - 1e-15)
        loss = -np.sum(y_true * np.log(y_pred_clipped)) / m
        
        d_z = y_pred - y_true
        
        d_W_fc = (1/m) * np.dot(self.flat_input.T, d_z)
        d_b_fc = (1/m) * np.sum(d_z, axis=0, keepdims=True)
        
        d_W_fc = np.clip(d_W_fc, -1.0, 1.0)
        d_b_fc = np.clip(d_b_fc, -1.0, 1.0)
        
        d_flat = np.dot(d_z, self.W_fc.T)
        d_pool_out = d_flat.reshape(self.pool_output_shape)
        
        d_conv_out = self.pool.backward(d_pool_out)
        

        d_conv_out[d_conv_out == 0] = 0
        
        self.conv.backward(d_conv_out, lr)
        
        self.W_fc -= lr * d_W_fc
        self.b_fc -= lr * d_b_fc
        
        return loss

if __name__ == "__main__":
    loader = MNISTDataLoader()
    
    X_train, Y_train, X_test, Y_test = loader.load_and_process()
    
    print(f"Training Shape: {X_train.shape}")
    
    model = CNN()
    
    learning_rate = 0.005 
    batch_size = 32
    epochs = 5

    print(f"Starting Training (LR={learning_rate})...")
    
    for epoch in range(epochs):
        epoch_loss = 0
        num_batches = 0
        
        perm = np.random.permutation(X_train.shape[0])
        X_shuffled = X_train[perm]
        Y_shuffled = Y_train[perm]
        
        for i in range(0, X_train.shape[0], batch_size):
            X_batch = X_shuffled[i:i+batch_size]
            Y_batch = Y_shuffled[i:i+batch_size]
            
            loss = model.train_step(X_batch, Y_batch, lr=learning_rate)
            epoch_loss += loss
            num_batches += 1
            
            if num_batches % 5 == 0:
                print(f"Epoch {epoch+1} | Batch {num_batches} | Loss: {loss:.4f}", end="\r")
        
        print(f"\nEpoch {epoch+1} DONE. Average Loss: {epoch_loss/num_batches:.4f}")

    print("\nEvaluating...")
    correct = 0
    total = 0
    for i in range(0, X_test.shape[0], batch_size):
        X_batch = X_test[i:i+batch_size]
        Y_batch = Y_test[i:i+batch_size]
        preds = model.forward(X_batch)
        correct += np.sum(np.argmax(preds, axis=1) == np.argmax(Y_batch, axis=1))
        total += len(X_batch)
        
    print(f"Test Accuracy: {correct/total:.2%}")
