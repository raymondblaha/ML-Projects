import pickle
import os
import matplotlib.pyplot as plt

configs = ["base", "swish", "batchnorm", "schedule", "adamw", "dropout"]
colors = {"base": "b", "swish": "g", "batchnorm": "black", "schedule": "orange", "adamw": "r", "dropout": "magenta"}

for config in configs:
    with open(f'history_{config}.pkl', 'rb') as f:
        history = pickle.load(f)
    plt.plot(history['loss'], color=colors[config], label=f'{config} train')
    plt.plot(history['val_loss'], color=colors[config], linestyle='dashed', label=f'{config} valid')

plt.legend()
plt.show()
plt.savefig('Loss.png')
