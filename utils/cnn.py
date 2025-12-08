#!/usr/bin/env/ python3
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model, regularizers
from tensorflow.keras.layers import Input, Conv1D, BatchNormalization, ReLU, MaxPooling1D, Dropout, Flatten, Dense, GlobalAveragePooling1D

# Enable dynamic GPU memory allocation
try:
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
except Exception:
    pass    # Continue with CPU if no GPU is available

class CNN_trainer:
    """
    Key implementations:
        1. Data augmentation tailored for time-series (additive Gaussian noise)
        2. Early stopping with patience=5 to prevent overfitting
        3. Cosine annealing learning rate schedule for smooth convergence
    
    References: 
        1. https://apxml.com/courses/cnns-for-computer-vision/chapter-2-advanced-training-optimization/learning-rate-schedules
        2. https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/schedules/CosineDecay
    """
    def __init__(self, X_t, y_t, X_v, y_v, epochs=100, batch_size=128, lr=5e-4, l2=1e-5, dropout=0.3):
        """
        X_t: Training time-series
        y_t: Training labels
        X_v: Validation time-series
        y_v: Validation labels
        epochs: Number of training epochs
        batch_size: Batch size for training
        lr: Initial learning rate
        l2: L2 regularization coefficient
        dropout: Dropout rate for regularization
        """
        # Store data
        self.X_t = X_t
        self.y_t = y_t
        self.X_v = X_v
        self.y_v = y_v
        
        # Store hyperparameters
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.initial_lr = lr
        self.l2 = l2
        self.dropout = dropout
        
        std = float(np.std(self.X_t))
        self.aug_noise_std = 0.05 * std if std > 0 else 0.0
    
    # Initialize model
    def init_model(self):
        """
        Plain CNN architecture:
          Conv1D → BN → ReLU → Dropout
          Conv1D → BN → ReLU → Dropout
          GAP → Dense layers → Output
        """
        reg = regularizers.l2(self.l2)

        inputs = Input(shape=self.X_t.shape[1:])
        x = inputs

        # Block 1
        x = Conv1D(32, 5, padding="same", use_bias=False, kernel_regularizer=reg)(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        x = Dropout(0.1)(x)

        # Block 2
        x = Conv1D(64, 5, padding="same", use_bias=False, kernel_regularizer=reg)(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        x = Dropout(0.1)(x)

        # Global pooling
        x = GlobalAveragePooling1D()(x)

        # Dense head
        x = Dense(64, activation="relu", kernel_regularizer=reg)(x)
        x = Dropout(0.2)(x)

        x = Dense(32, activation="relu", kernel_regularizer=reg)(x)

        outputs = Dense(2, activation="linear")(x)

        self.model = Model(inputs, outputs)
    
    # Initialize optimizer
    def init_optimizer(self):
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr)
        
    # Initialize training metrics
    def init_metrics(self):
        self.train_loss = tf.keras.metrics.Mean()
        self.train_mae = tf.keras.metrics.MeanAbsoluteError()
        
    # NOTE: Apply random augmentations
    def augment_data(self, x):
        """
        Inject Gaussian noise: x' = x + noise
        """
        if self.aug_noise_std <= 0.0:
            return x
        noise = np.random.normal(loc=0.0, scale=self.aug_noise_std, size=x.shape)
        x_aug = x + noise

        return x_aug
    
    # Compile training step into a static computation graph for speed improvement
    @tf.function
    def train_step(self, x, y):
        with tf.GradientTape() as tape:
            # Forward pass
            predictions = self.model(x, training=True)
            
            # Compute cross-entropy loss (MSE)
            loss_fn = tf.keras.losses.MeanSquaredError()
            loss = loss_fn(y, predictions)
            loss = tf.reduce_mean(loss)
            
            # Add L2 regularization losses from model layers
            if self.model.losses:
                loss += tf.add_n(self.model.losses)
        
        # Compute and apply gradients
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        
        # Update metrics
        self.train_loss.update_state(loss)
        self.train_mae.update_state(y, predictions)
        
        return loss
    
    # Train for one complete epoch
    def train_epoch(self):
        self.train_loss.reset_states()
        self.train_mae.reset_states()
        
        # Shuffle training data
        train_idx = np.random.permutation(len(self.X_t))
        
        # Compute number of batches
        num_batches = (len(self.X_t) + self.batch_size - 1) // self.batch_size
        
        # Iterate over all batches
        for i in range(num_batches):
            # Get batch indices
            start_idx = i * self.batch_size
            end_idx = min(start_idx + self.batch_size, len(self.X_t))
            batch_idx = train_idx[start_idx:end_idx]
            
            x_batch = self.X_t[batch_idx].astype(np.float32)
            y_batch = self.y_t[batch_idx]
            
            # Apply augmentation to each image in batch
            x_batch = np.array([self.augment_data(img) for img in x_batch])
            
            # Perform training step
            self.train_step(x_batch, y_batch)
            
            # Print progress every 10 batches and at end
            if (i + 1) % 10 == 0 or (i + 1) == num_batches:
                print(f"\r{i+1}/{num_batches} - loss: {self.train_loss.result():.4f} - "
                      f"MAE: {self.train_mae.result():.4f}", end='', flush=True) 
    
    # Perform validation
    def validate(self, X_v):
        # Initialize validation metrics
        val_loss = tf.keras.metrics.Mean()
        val_mae = tf.keras.metrics.MeanAbsoluteError()
   
        # Compute number of batches
        num_batches = (len(X_v) + self.batch_size - 1) // self.batch_size
        
        # Iterate over all batches
        for i in range(num_batches):
            # Get batch indices
            start_idx = i * self.batch_size
            end_idx = min(start_idx + self.batch_size, len(X_v))
            
            x_batch = X_v[start_idx:end_idx]
            y_batch = self.y_v[start_idx:end_idx]
            
            # Forward pass
            predictions = self.model(x_batch, training=False)
            loss = tf.reduce_mean(tf.keras.losses.MSE(y_batch, predictions))

            val_loss.update_state(loss)
            val_mae.update_state(y_batch, predictions)
        
        return val_loss.result().numpy(), val_mae.result().numpy()
    
    # Update learning rate using cosine annealing schedule
    def update_learning_rate(self, epoch):        
        denom = max(1, self.epochs - 1)
        cosine_decay = 0.5 * (1 + np.cos(np.pi * epoch / denom))
        new_lr = self.initial_lr * cosine_decay
        self.optimizer.learning_rate.assign(new_lr)
        print(f"Learning rate: {new_lr:.6f}")
    
    # Start training with early stopping
    def run(self):
        self.init_model()
        self.init_optimizer()
        self.init_metrics()
        self.history = {"loss": [], "val_loss": [], "mae": [], "val_mae": []}
        
        # Normalize validation data
        X_v_normalized = self.X_v.astype(np.float32) 
        
        # Store the best validation MAE for early stopping
        best_val_mae = float('inf')
        # Define counter for consecutive epochs without improvement
        patience_counter = 0
        
        for epoch in range(self.epochs):
            print(f"\nTraining Epoch {epoch + 1}/{self.epochs}")
            self.train_epoch()
            
            # Validate
            val_loss, val_mae = self.validate(X_v_normalized)
            print(f" - val_loss: {val_loss:.4f} - val_MAE: {val_mae:.4f}")
            
            # Store metrics in history
            self.history["loss"].append(self.train_loss.result().numpy())
            self.history["val_loss"].append(val_loss)
            self.history["mae"].append(self.train_mae.result().numpy())
            self.history["val_mae"].append(val_mae)
            
            # Update learning rate
            self.update_learning_rate(epoch)
                    
            # Inside the training loop
            if val_mae < best_val_mae:
                best_val_mae = val_mae
                patience_counter = 0
                print(f"Validation MAE improved to {val_mae:.4f}")

                # Save the best model weights
                self.model.save_weights('best_model_weights.h5')
                print("Saved best model weights to 'best_model_weights.h5'")
            else:
                patience_counter += 1
                print(f"No improvement (patience: {patience_counter}/5)")

                if patience_counter >= 5:
                    print("Early stopping triggered")
                    # Load the previously saved best model weights
                    self.model.load_weights('best_model_weights.h5')
                    print("Loaded best model weights from 'best_model_weights.h5'")
                    break

                
        print(f"\nCNN training complete. Best validation MAE: {best_val_mae:.4f})")
