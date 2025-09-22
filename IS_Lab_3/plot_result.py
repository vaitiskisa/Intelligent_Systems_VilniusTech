import numpy as np
import matplotlib.pyplot as plt

def plot_results_1D():
    # Load training data
    train_data = np.loadtxt("train_data.csv", delimiter=",")
    
    # Load training expected values
    train_expected = np.loadtxt("train_expected.csv", delimiter=",")

    # Plot training data in subplot
    plt.subplot(1, 2, 1)
    plt.plot(train_data, train_expected, color='blue', label='Training Data')
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

    order = np.argsort(valid_data)
    valid_sorted = valid_data[order]
    predictions_sorted = predictions[order]
    expected_sorted = expected[order]

    # Plot predictions
    plt.subplot(1, 2, 2)
    plt.plot(valid_sorted, expected_sorted, color='red',  linewidth=5, label='Expected (sorted)')
    plt.plot(valid_sorted, predictions_sorted, color='blue', linewidth=2, label='Predictions (sorted)')
    plt.xlabel('Input x')
    plt.ylabel('y')
    plt.title('Predictions vs Expected')
    plt.legend()
    plt.grid(True)
    plt.savefig("IS_Lab_2_Results.png")
    plt.show()

if __name__ == "__main__":
    plot_results_1D()
