import matplotlib as plt

def plot_predicted_vs_actual(y_test, y_test_hat):
    plt.scatter(y_test_hat, y_test, c='b', alpha=0.5, marker='.', label='Real')
    plt.grid(color='#D3D3D3', linestyle='solid')
    plt.legend(loc='lower right')
    plt.show()
