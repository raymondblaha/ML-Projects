import matplotlib.pyplot as plt
import pickle

with open("k.pkl", "rb") as f:
    accuracies, ks, times, explained_variances = pickle.load(f)

# plot Time for training
fig, axs = plt.subplots(nrows=3, ncols=1, figsize=(12, 12))
axs[0].plot(ks, times)
axs[0].set_xlabel("Dimensions Used for Classification")
axs[0].set_ylabel("Time (s)") 
axs[0].set_title("Time for training")

# plot the explained variance ratio
axs[1].plot(ks, explained_variances)
axs[1].set_xlabel("Dimensions Used for Classification")
axs[1].set_ylabel("Explained Variance Ratio")
axs[1].set_title("Explained Variance Ratio")

# plot the accuracy of the classifier on the training data
axs[2].plot(ks, accuracies)
axs[2].set_xlabel("Dimensions Used for Classification")
axs[2].set_ylabel("Accuracy")
axs[2].set_title("Accuracy of the classifier on the training data")

plt.tight_layout()
plt.savefig("k_plots.png")

