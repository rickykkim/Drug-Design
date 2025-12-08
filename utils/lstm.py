#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#!/usr/bin/env python3
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model, regularizers
from tensorflow.keras.layers import Input, LSTM, Bidirectional, Dense, Dropout, ReLU

# Enable dynamic GPU memory allocation
try:
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
except Exception:
    pass # Continue with CPU if no GPU is available


class LSTM_trainer:
    """
    Pure LSTM trainer for multivariate time series regression. We implement a bidirectional lstm to allow
    the time-series data to learn from both past and future context. Ideally this allows for our RNN to learn
    temporal relationships where later seqeunces in the manufacturing data give context for understanding
    earlier patterns. For instance, early instability (from speed noise) followed by a later period of tablet
    press speed of 0 suggests higher total waste due to clogs/jamming. A uni-directional LSTM would predict
    the waste from the clogging without understanding that the clog was responsible for the instability to
    begin with. However, bi-directional LSTMs can help the model interpretate the earlier anomalies and how
    they evolve.

    Differences vs CNN_trainer:
        1. Uses stacked (Bi)LSTM layers instead of Conv1D.
        2. Keeps the same training loop, augmentation, early stopping,
        and cosine learning rate schedule for a fair comparison.

    Inputs:
      X_t: training time series, shape (N_train, T, F)
      y_t: training targets, shape (N_train, 2) -> [total_waste, Total impurities]
      X_v: validation time series, shape (N_val, T, F)
      y_v: validation targets, shape (N_val, 2)
      
    References:
        1. https://www.geeksforgeeks.org/nlp/explanation-of-bert-model-nlp/
        2. https://www.tensorflow.org/api_docs/python/tf/keras/layers/Bidirectional
        3. https://medium.com/data-science-data-engineering/time-series-prediction-lstm-bi-lstm-gru-99334fc16d75
    """
    def __init__(self, X_t, y_t, X_v, y_v, epochs=100, batch_size=128, lr=5e-4, l2=1e-5, dropout=0.3):
        #Store data
        self.X_t = X_t
        self.y_t = y_t
        self.X_v = X_v
        self.y_v = y_v

        #Hyperparameters
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.initial_lr = lr
        self.l2 = l2
        self.dropout = dropout

        std = float(np.std(self.X_t))
        self.aug_noise_std = 0.05 * std if std > 0 else 0.0

    #Initialize Model: pure LSTM on time series
    def init_model(self):
        """
        LSTM architecture for 1D multivariate time series; consists of:
            1. Two stacked (Bi)LSTM layers:
                First: return_sequences=True to keep full sequence
                Second: return_sequences=False to output a single vector
            2. L2 regularization on recurrent weights
            3. Dense blocks with Dropout for regression head
            4. Final Dense(2) with linear activation: [total_waste, Total impurities]
        """
        reg = regularizers.l2(self.l2)

        inputs = Input(shape=self.X_t.shape[1:])
        x = inputs

        #First LSTM layer (sequence to sequence)
        x = Bidirectional(LSTM(64, return_sequences=True, kernel_regularizer=reg, recurrent_regularizer=reg,
        dropout=self.dropout, recurrent_dropout=0.0))(x)

        #Second LSTM layer (sequence to vector)
        x = Bidirectional(LSTM(32, return_sequences=False, kernel_regularizer=reg, recurrent_regularizer=reg,
        dropout=self.dropout, recurrent_dropout=0.0))(x)

        #Dense layers
        x = Dense(64, use_bias=False, kernel_regularizer=reg)(x)
        x = ReLU()(x)
        x = Dropout(0.2)(x)

        x = Dense(32, use_bias=False, kernel_regularizer=reg)(x)
        x = ReLU()(x)
#         x = Dropout(0.2)(x)

        outputs = Dense(2, activation="linear")(x)

        self.model = Model(inputs=inputs, outputs=outputs)

    #Initialize optimizer and training metrics
    def init_optimizer(self):
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr)

    def init_metrics(self):
        self.train_loss = tf.keras.metrics.Mean()
        self.train_mae = tf.keras.metrics.MeanAbsoluteError()

    #Data augmentation. Here we add Gaussian noise to the time-series data
    def augment_data(self, x):
        if self.aug_noise_std <= 0.0:
            return x
        noise = np.random.normal(loc=0.0, scale=self.aug_noise_std, size=x.shape)
        x_aug = x + noise
        return x_aug

    #Add training step
    @tf.function
    def train_step(self, x, y):
        with tf.GradientTape() as tape:
            #Forward pass
            predictions = self.model(x, training=True)
            
            #Compute loss (here we use mean squared error)
            loss_fn = tf.keras.losses.MeanSquaredError()
            loss = loss_fn(y, predictions)
            loss = tf.reduce_mean(loss)
            
            #L2 regularization
            if self.model.losses:
                loss += tf.add_n(self.model.losses)
        
        #Compute and apply gradients 
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        
        #Update
        self.train_loss.update_state(loss)
        self.train_mae.update_state(y, predictions)

        return loss

    # Train for one complete epoch
    def train_epoch(self, epoch):
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

            # Apply augmentation to each image in the batch
            x_batch = np.array([self.augment_data(seq) for seq in x_batch])

            # Perform training step
            self.train_step(x_batch, y_batch)
            
            # Print progress every 10 batches and at end
            if (i + 1) % 10 == 0 or (i + 1) == num_batches:
                print(
                    f"\r{i+1}/{num_batches} - loss: {self.train_loss.result():.4f} "
                    f"- MAE: {self.train_mae.result():.4f}",
                    end="", flush=True
                )

    # Perform validation
    def validate(self, X_v):
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

        best_val_mae = float('inf')
        # Define counter for consecutive epochs without improvement
        patience_counter = 0

        for epoch in range(self.epochs):
            print(f"\nTraining Epoch {epoch + 1}/{self.epochs}")
            self.train_epoch(epoch)

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

            # Update the best validation MAE and reset patience (if applicable)
            if val_mae < best_val_mae:
                best_val_mae = val_mae
                patience_counter = 0
                print(f"Validation MAE improved to {val_mae:.4f}")
                self.model.save_weights('best_lstm_model_weights.h5')
                print("Updated the best LSTM model")
            # Allow up to 5 consecutive non-improving epochs before stopping
            else:
                patience_counter += 1
                print(f"No improvement (patience: {patience_counter}/5)")
                if patience_counter >= 5:
                    print("Early stopping triggered")
                    self.model.load_weights('best_lstm_model_weights.h5')
                    break

        print(f"\nLSTM training complete (Best validation MAE: {best_val_mae:.4f})")

