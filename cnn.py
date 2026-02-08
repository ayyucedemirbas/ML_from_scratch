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
            print("Download complete")

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

        # (Batch, Channels, Height, Width)
        # MNIST is grayscale, so Channels = 1
        x_train = x_train.reshape(x_train.shape[0], 1, 28, 28)
        x_test = x_test.reshape(x_test.shape[0], 1, 28, 28)

        # Normalize to 0-1
        x_train = x_train.astype(np.float32) / 255.0
        x_test = x_test.astype(np.float32) / 255.0

        # One-hot encoding
        y_train = self.one_hot_encode(y_train)
        y_test = self.one_hot_encode(y_test)

        return x_train, y_train, x_test, y_test

class Conv3x3:
    #Input Shape: (Batch, Channels, Height, Width)
    def __init__(self, num_filters, input_channels=1):
        self.num_filters = num_filters
        self.input_channels = input_channels
        self.filter_size = 3
        
        # Xavier/Glorot Initialization for Conv Filters
        # Shape: (num_filters, input_channels, 3, 3)
        self.filters = np.random.randn(num_filters, input_channels, 3, 3) / 9.0
        self.bias = np.zeros((num_filters, 1))

        self.last_input = None

    def iterate_regions(self, image):
        h, w = image.shape[1], image.shape[2]
        
        for i in range(h - 2):
            for j in range(w - 2):
                im_region = image[:, i:(i + 3), j:(j + 3)]
                yield im_region, i, j

    def forward(self, input_data):
        self.last_input = input_data
        batch_size, channels, h, w = input_data.shape
        
        # Output dimensions (H-2, W-2)
        h_out, w_out = h - 2, w - 2
        
        # Initialize output: (Batch, Num_Filters, H_out, W_out)
        output = np.zeros((batch_size, self.num_filters, h_out, w_out))

        for i in range(h_out):
            for j in range(w_out):
                # Region: (Batch, Channels, 3, 3)
                region = input_data[:, :, i:i+3, j:j+3]
                
                for f in range(self.num_filters):
                    output[:, f, i, j] = np.sum(region * self.filters[f], axis=(1, 2, 3))
        
        return output + self.bias.reshape(1, -1, 1, 1)

    def backward(self, d_L_d_out, learning_rate):
        batch_size, _, h_out, w_out = d_L_d_out.shape
        
        # Initialize gradients
        d_L_d_filters = np.zeros(self.filters.shape)
        d_L_d_bias = np.zeros(self.bias.shape)
        
        # Gradient for filters
        # dW = sum(d_out * input_region)
        for i in range(h_out):
            for j in range(w_out):
                region = self.last_input[:, :, i:i+3, j:j+3]
                
                for f in range(self.num_filters):
                    # d_out_pixel: (Batch,)
                    d_out_pixel = d_L_d_out[:, f, i, j]
                    
                    # region: (Batch, Channels, 3, 3)
                    # d_out_pixel: (Batch, 1, 1, 1)
                    d_L_d_filters[f] += np.sum(
                        region * d_out_pixel[:, None, None, None], axis=0
                    )

        # Gradient for bias: sum over batch, h, w
        # d_L_d_out: (Batch, Filters, H, W) -> sum over (0, 2, 3)
        d_L_d_bias = np.sum(d_L_d_out, axis=(0, 2, 3)).reshape(-1, 1)

        # Update weights
        self.filters -= learning_rate * d_L_d_filters
        self.bias -= learning_rate * d_L_d_bias
        
        return None 

class MaxPool2:
    def iterate_regions(self, image):
        h, w = image.shape[2], image.shape[3]
        new_h = h // 2
        new_w = w // 2
        
        for i in range(new_h):
            for j in range(new_w):
                im_region = image[:, :, (i*2):(i*2+2), (j*2):(j*2+2)]
                yield im_region, i, j

    def forward(self, input_data):
        self.last_input = input_data
        batch, filters, h, w = input_data.shape
        h_out, w_out = h // 2, w // 2
        
        output = np.zeros((batch, filters, h_out, w_out))
        
        for i in range(h_out):
            for j in range(w_out):
                region = input_data[:, :, (i*2):(i*2+2), (j*2):(j*2+2)]
                output[:, :, i, j] = np.amax(region, axis=(2, 3))
                
        return output

    def backward(self, d_L_d_out):
        input_data = self.last_input
        batch, filters, h, w = input_data.shape
        d_L_d_input = np.zeros(input_data.shape)
        
        h_out, w_out = d_L_d_out.shape[2], d_L_d_out.shape[3]
        
        for i in range(h_out):
            for j in range(w_out):
                region = input_data[:, :, (i*2):(i*2+2), (j*2):(j*2+2)]
                
                # Find max indices
                max_val = np.amax(region, axis=(2, 3), keepdims=True)
                mask = (region == max_val)
                
                # Distribute gradient
                d_L_d_input[:, :, (i*2):(i*2+2), (j*2):(j*2+2)] += \
                    d_L_d_out[:, :, i:i+1, j:j+1] * mask
                    
        return d_L_d_input

class SimpleCNN:
    def __init__(self):
        # Conv Layer: 8 filters of 3x3, input 1 channel
        self.conv = Conv3x3(num_filters=8, input_channels=1)
        
        # Pool Layer: 2x2
        self.pool = MaxPool2()
        
        # Input: 28x28 -> Conv(3x3, no pad) -> 26x26 -> Pool(2x2) -> 13x13
        # Total Flattened size: 8 filters * 13 * 13 = 1352
        self.flat_size = 8 * 13 * 13
        
        # Dense Layer (Softmax Output)
        self.W_fc = np.random.randn(self.flat_size, 10) / np.sqrt(self.flat_size)
        self.b_fc = np.zeros((1, 10))

    def softmax(self, Z):
        exp_Z = np.exp(Z - np.max(Z, axis=1, keepdims=True))
        return exp_Z / np.sum(exp_Z, axis=1, keepdims=True)

    def forward(self, X):
        # X: (Batch, 1, 28, 28)
        out = self.conv.forward(X)
        
        # ReLU (Activation)
        out = np.maximum(0, out) 
        
        # MaxPool
        out = self.pool.forward(out)
        
        self.pool_output_shape = out.shape # Store for backward
        # Flatten: (Batch, 8, 13, 13) -> (Batch, 1352)
        out_flat = out.reshape(out.shape[0], -1)
        self.flat_input = out_flat
        
        # Dense Softmax
        z = np.dot(out_flat, self.W_fc) + self.b_fc
        return self.softmax(z)

    def compute_loss(self, y_pred, y_true):
        # Clip to prevent log(0)
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        m = y_true.shape[0]
        loss = -np.sum(y_true * np.log(y_pred)) / m
        return loss

    def train_step(self, X, y_true, learning_rate=0.01):
        m = X.shape[0]
        
        y_pred = self.forward(X)
        loss = self.compute_loss(y_pred, y_true)
        
        
        # Gradient of Loss w.r.t Dense Output (Softmax + CE)
        d_z = y_pred - y_true
        
        # Gradient for Dense Weights
        d_W_fc = (1/m) * np.dot(self.flat_input.T, d_z)
        d_b_fc = (1/m) * np.sum(d_z, axis=0, keepdims=True)
        
        # Gradient w.r.t Flattened Input (Back to Pooling)
        d_flat = np.dot(d_z, self.W_fc.T)
        
        # Reshape back to (Batch, 8, 13, 13)
        d_pool_out = d_flat.reshape(self.pool_output_shape)
        
        # Backprop through MaxPool
        d_conv_out = self.pool.backward(d_pool_out)
        self.conv.backward(d_conv_out, learning_rate)
        
        self.W_fc -= learning_rate * d_W_fc
        self.b_fc -= learning_rate * d_b_fc
        
        return loss

if __name__ == "__main__":
    loader = MNISTDataLoader()
    X_train, Y_train, X_test, Y_test = loader.load_and_process()
    
    print(f"Training Data Shape: {X_train.shape}")
    
    model = SimpleCNN()
    
    epochs = 3
    batch_size = 32
    learning_rate = 0.1
    
    print(f"Starting Training for {epochs} epochs...")
    start_time = time.time()
    
    for epoch in range(epochs):
        epoch_loss = 0
        num_batches = 0
        
        permutation = np.random.permutation(X_train.shape[0])
        X_shuffled = X_train[permutation]
        Y_shuffled = Y_train[permutation]
        
        for i in range(0, X_train.shape[0], batch_size):
            X_batch = X_shuffled[i:i+batch_size]
            Y_batch = Y_shuffled[i:i+batch_size]
            
            loss = model.train_step(X_batch, Y_batch, learning_rate)
            epoch_loss += loss
            num_batches += 1
            
            if num_batches % 10 == 0:
                print(f"Epoch {epoch+1} | Batch {num_batches} | Loss: {loss:.4f}", end="\r")
        
        print(f"\nEpoch {epoch+1} Average Loss: {epoch_loss/num_batches:.4f}")
        
    total_time = time.time() - start_time
    print(f"\nTraining Finished in {total_time:.2f} seconds.")
    
    # Process test set in batches to avoid memory overflow
    correct = 0
    total = 0
    
    for i in range(0, X_test.shape[0], batch_size):
        X_batch = X_test[i:i+batch_size]
        Y_batch = Y_test[i:i+batch_size]
        
        preds = model.forward(X_batch)
        pred_labels = np.argmax(preds, axis=1)
        true_labels = np.argmax(Y_batch, axis=1)
        
        correct += np.sum(pred_labels == true_labels)
        total += len(true_labels)
        
    print(f"Test Accuracy: {correct/total:.2%}")
