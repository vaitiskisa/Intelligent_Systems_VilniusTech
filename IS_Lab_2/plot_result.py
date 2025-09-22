import numpy as np
import matplotlib.pyplot as plt

def plot_results_1D():
    # Load training data
    train_data = np.loadtxt("train_data_1D.csv", delimiter=",")
    
    # Load training expected values
    train_expected = np.loadtxt("train_expected_1D.csv", delimiter=",")

    # Plot training data in subplot
    plt.subplot(1, 2, 1)
    plt.scatter(train_data, train_expected, color='blue', label='Training Data')
    plt.xlabel('Input x')
    plt.ylabel('y')
    plt.title('Training Data')
    plt.legend()
    plt.grid(True)

    # Load validation data
    valid_data = np.loadtxt("valid_data_1D.csv", delimiter=",")

    # Load predictions and expected values
    predictions = np.loadtxt("predictions_1D.csv", delimiter=",")
    expected = np.loadtxt("expected_1D.csv", delimiter=",")

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
    
    # Load training expected values
    train_expected = np.loadtxt("train_expected_2D.csv", delimiter=",")

    # Plot training data in subplot
    fig = plt.figure(figsize=(12, 6))

    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    x_vals = np.unique(train_data[:, 0])
    y_vals = np.unique(train_data[:, 1])
    X, Y = np.meshgrid(x_vals, y_vals, indexing='ij')
    Z = train_expected.reshape(len(x_vals), len(y_vals))
    surf1 = ax1.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none', alpha=0.9)
    ax1.set_xlabel('Input x1')
    ax1.set_ylabel('Input x2')
    ax1.set_zlabel('y')
    ax1.set_title('Training Data')
    fig.colorbar(surf1, ax=ax1, shrink=0.6, aspect=10)

    # Load validation data
    valid_data = np.loadtxt("valid_data_2D.csv", delimiter=",")

    # Load predictions and expected values
    predictions = np.loadtxt("predictions_2D.csv", delimiter=",")
    expected = np.loadtxt("expected_2D.csv", delimiter=",")

    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    surf2 = ax2.plot_trisurf(valid_data[:, 0], valid_data[:, 1], predictions,
                              cmap='viridis', linewidth=0.2, antialiased=True, alpha=0.9)
    surf3 = ax2.plot_trisurf(valid_data[:, 0], valid_data[:, 1], expected,
                              cmap='hot', linewidth=0.2, antialiased=True, alpha=0.9)
    ax2.set_xlabel('Input x1')
    ax2.set_ylabel('Input x2')
    ax2.set_zlabel('y')
    ax2.set_title('Predictions')
    fig.colorbar(surf2, ax=ax2, shrink=0.6, aspect=10)
    fig.colorbar(surf3, ax=ax2, shrink=0.6, aspect=10)

    # # Plot predictions
    # ax3 = fig.add_subplot(2, 2, 3, projection='3d')
    # surf3 = ax3.plot_trisurf(valid_data[:, 0], valid_data[:, 1], expected,
    #                           cmap='viridis', linewidth=0.2, antialiased=True, alpha=0.9)
    # ax3.set_xlabel('Input x1')
    # ax3.set_ylabel('Input x2')
    # ax3.set_zlabel('y')
    # ax3.set_title('Expected')
    # fig.colorbar(surf3, ax=ax3, shrink=0.6, aspect=10)

    plt.savefig("IS_Lab_2_Results_2D.png")
    plt.show()

if __name__ == "__main__":
    # plot_results_1D()
    plot_results_2D()
