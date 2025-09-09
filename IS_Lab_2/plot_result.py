import numpy as np
import matplotlib.pyplot as plt

def plot_results_1D():
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

def plot_results_2D():
    # Load training data
    train_data = np.loadtxt("train_data_2D.csv", delimiter=",")
    print(train_data)
    
    # Load training expected values
    train_expected = np.loadtxt("train_expected_2D.csv", delimiter=",")

    # Plot training data in subplot
    fig = plt.figure(figsize=(12, 6))

    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    ax1.scatter(train_data[:, 0], train_data[:, 1], train_expected, color='blue', label='Training Data')
    ax1.set_xlabel('Input x1')
    ax1.set_ylabel('Input x2')
    ax1.set_zlabel('y')
    ax1.set_title('Training Data')
    ax1.legend()

    # Load validation data
    valid_data = np.loadtxt("valid_data_2D.csv", delimiter=",")

    # Load predictions and expected values
    predictions = np.loadtxt("predictions_2D.csv", delimiter=",")
    expected = np.loadtxt("expected_2D.csv", delimiter=",")

    # Plot predictions
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    ax2.scatter(valid_data[:, 0], valid_data[:, 1], predictions, color='green', label='Predictions')
    ax2.scatter(valid_data[:, 0], valid_data[:, 1], expected, color='red', label='Expected')
    ax2.set_xlabel('Input x1')
    ax2.set_ylabel('Input x2')
    ax2.set_zlabel('y')
    ax2.set_title('Predictions vs Expected')
    ax2.legend()

    plt.savefig("IS_Lab_2_Results_2D.png")
    plt.show()

if __name__ == "__main__":
    # plot_results_1D()
    plot_results_2D()