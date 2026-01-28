import tensorflow as tf
import matplotlib.pyplot as plt
import h5py
from keras.models import load_model

# Load the model and its training history from the .h5 file
model = load_model(r'C:\Users\david\Desktop\Python\AI_cnn_model.h5')

# Load training history
with h5py.File(model, 'r') as hf:
    history = hf['history'][:]

# Extract loss and batch size data from the history
loss = history['loss']
batch_size = history['batch_size']

# Create a chart to display the loss and batch size
epochs = range(1, len(loss) + 1)

# Create subplots
fig, ax1 = plt.subplots()

# Plot the training loss
ax1.set_xlabel('Epochs')
ax1.set_ylabel('Loss', color='tab:blue')
ax1.plot(epochs, loss, color='tab:blue')
ax1.tick_params(axis='y', labelcolor='tab:blue')
ax1.set_ylim(0, max(loss) + 0.1)

# Create a second y-axis for batch size
ax2 = ax1.twinx()
ax2.set_ylabel('Batch Size', color='tab:red')
ax2.plot(epochs, batch_size, color='tab:red')
ax2.tick_params(axis='y', labelcolor='tab:red')

# Set the title and labels
plt.title('Training Loss and Batch Size Over Epochs')
plt.xlabel('Epochs')

# Display the plot
plt.show()