import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('results.csv', header=None, names=['config', 'loss', 'accuracy', 'time'])

# Add labels to which model is which
df.loc[df['config'] == 'base', 'label'] = 'Base'
df.loc[df['config'] == 'swish', 'label'] = 'Swish'
df.loc[df['config'] == 'batchnorm', 'label'] = 'BatchNorm'
df.loc[df['config'] == 'schedule', 'label'] = 'Schedule'
df.loc[df['config'] == 'adamw', 'label'] = 'AdamW'
df.loc[df['config'] == 'dropout', 'label'] = 'Dropout'


#Make a scatter plot of the time spent training vs the accuracy on the test data. Save it as acc time.png
plt.scatter(df['time'], df['accuracy'])
plt.xlabel('Time')
plt.ylabel('Accuracy')
plt.show()
plt.savefig('Acc_time.png')
