import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    # Load training data
    train_data = np.loadtxt("train_data.csv", delimiter=",")
    
    # Load training expected values
    train_expected = np.loadtxt("train_expected.csv", delimiter=",")

    # Plot training data in subplot
    plt.subplot(1, 2, 1)
    plt.scatter(train_data, train_expected, color='blue', label='Training Data')
    plt.xlabel('Input x')
    plt.ylabel('y')
    plt.title('Training Data')
    plt.legend()
    plt.grid(True)

    # Load validation data
    valid_data = np.loadtxt("valid_data.csv", delimiter=",")

    # Load predictions and expected values
    predictions = np.loadtxt("predictions.csv", delimiter=",")
    expected = np.loadtxt("expected.csv", delimiter=",")

    # Plot predictions
    plt.subplot(1, 2, 2)
    plt.scatter(valid_data, predictions, color='green', label='Predictions')
    plt.scatter(valid_data, expected, color='red', label='Expected')
    plt.xlabel('Input x')
    plt.ylabel('y')
    plt.title('Predictions vs Expected')
    plt.legend()
    plt.grid(True)
    plt.savefig("IS_Lab_2_Results.png")
    plt.show()