import numpy as np
import nibabel as nib
from scipy.io import loadmat
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

def load_nii(filename):
    data = nib.load(filename)
    data = data.get_fdata()
    data = np.array(data)
    return data

def load_rve_encoding(ve_parameter):
    gx1 = ve_parameter["gy1"]
    gy1 = ve_parameter["gx1"]
    gx2 = ve_parameter["gy2"]
    gy2 = ve_parameter["gx2"]
    gx1 = torch.tensor(gx1)
    gy1 = torch.tensor(gy1)
    gx2 = torch.tensor(gx2)
    gy2 = torch.tensor(gy2)

    return torch.cat([gx1, gy1, gx2, gy2], dim=1)


def alpha_func_pytorch(phase):
    """
    Returns calculated tagging efficiency for unipolar pcasl as a function of the phase rotation per TR.

    Parameters:
        phase (float): Phase rotation per TR in radians.

    Returns:
        a (float): Tagging efficiency for unipolar pcasl.
    """
    # Coefficients derived from FT of simulated response across v=5:5:40
    a = (-1 / 36 / 1.8239) * (75.33 * torch.cos(phase) - 11.6 * torch.cos(3 * phase) + 1.93 * torch.cos(5 * phase))

    return a


def compute_labeling_efficiencies_pytorch(pred, rve_encoding):
    """
        :params x  (ns)
        :params y  (ns)
        :params df (ns)
    """
    x = pred[:, 0]
    y = pred[:, 1]
    df = pred[:, 2]
    x = x[:,None]
    y = y[:,None]
    df = df[:,None]
    gx1 = rve_encoding[:, 0].to(x.device)
    gy1 = rve_encoding[:, 1].to(x.device)
    gx2 = rve_encoding[:, 2].to(x.device)
    gy2 = rve_encoding[:, 3].to(x.device)
    gx1 = gx1[None,:]
    gy1 = gy1[None,:]
    gx2 = gx2[None,:]
    gy2 = gy2[None,:]

    phase = ((x - gx1) * (gx2 - gx1) + (y - gy1) * (gy2 - gy1)) / ((gx1 - gx2) ** 2 + (gy1 - gy2) ** 2) * torch.pi + df
    labeling_efficiencies = alpha_func_pytorch(phase)

    return labeling_efficiencies


# Define the Neural Network
class PINN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(PINN, self).__init__()

        self.ve_parameter = loadmat("./dictionary/ve_param.mat")
        self.ve_parameter = load_rve_encoding(self.ve_parameter)

        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, output_dim)

        self.relu = nn.ReLU()
        self.A = nn.Parameter(torch.randn(input_dim, output_dim))  # (62, 3)

    def forward(self, x):
        x = x.float()
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        y_pred = self.fc4(x)  # (batch_size, 3)
        return y_pred

    def constraint_loss(self, y_pred, x):
        labeling_efficiencies = compute_labeling_efficiencies_pytorch(y_pred, self.ve_parameter)
        ns, ne = labeling_efficiencies.shape
        labeling_efficiencies = torch.cat([torch.ones(ns, 2).to(labeling_efficiencies.device), labeling_efficiencies],
                                          dim=1)
        return torch.mean((x - labeling_efficiencies) ** 2)

# Loss functions
def supervised_loss(y_pred, y_true):
    return torch.mean((y_pred - y_true) ** 2)

# Define the plot function for 2D histograms
def plot_2d_histogram(x, y, xlabel, ylabel, title, bins=100):
    plt.figure(figsize=(8, 6))
    plt.hist2d(x, y, bins=bins, range=[[-64, 64], [-64, 64]] if xlabel != 'f' and ylabel != 'f' else [[-64, 64], [-2, 2]])
    plt.colorbar(label='Density')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.show()


if __name__ == "__main__":

    #config_path = "./configs/config_PINN_rVE_ASL.yaml"
    #with open(config_path, "r") as f:
    #config = yaml.load(f, Loader=yaml.FullLoader)

    x_values = np.arange(64, -65, -2)  # From -64 to 64 with step 2
    y_values = np.arange(64, -65, -2)  # From -64 to 64 with step 2
    f_values = np.arange(-2, 2.1, 0.1)
    dictionary = []
    for f in f_values:  # Iterate over f
        for x in x_values:  # Iterate over x
            for y in y_values:  # Iterate over y
                dictionary.append([x, y, f])

    x = load_nii("./data/1/edMs_rVEASL.nii")
    x = np.reshape(x, [64 * 64 * 12, 62])
    y0 = load_nii("./data/1/indices_ccmax.nii")
    y = []
    for i in range(49152):
        y.append(dictionary[int(y0[0, i])-1])
    y = np.array(y)

    input_dim = 62
    output_dim = 3
    batch_size = 2048
    num_epochs = 5

    # Generate Dataset
    X = torch.tensor(x)
    Y = torch.tensor(y)

    # Use PyTorch DataLoader for batching
    dataset = TensorDataset(X, Y)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Initialize Model, Optimizer
    model = PINN(input_dim, output_dim)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training Loop with Adaptive Lambda
    for epoch in range(num_epochs):
        total_loss = 0
        total_Ls = 0
        total_Lc = 0

        for x_batch, y_batch in data_loader:
            optimizer.zero_grad()

            y_pred = model(x_batch)  # Forward pass
            Ls = supervised_loss(y_pred, y_batch)
            Lc = model.constraint_loss(y_pred, x_batch)

            # Dynamically adjust lambda_2
            lambda_2 = (5 * Ls.item() / Lc.item()) if Lc.item() > 0 else 1.0

            loss = Ls + lambda_2 * Lc
            loss.backward()  # Backpropagation
            optimizer.step()  # Update weights

            total_loss += loss.item()
            total_Ls += Ls.item()
            total_Lc += Lc.item()

        avg_Ls = total_Ls / len(data_loader)
        avg_Lc = total_Lc / len(data_loader)
        print(
            f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss / len(data_loader)}, Ls: {avg_Ls}, Lc: {avg_Lc}, Lambda_2: {lambda_2:.5f}")

    model.eval()  # Set model to evaluation mode

    # Get final predicted outputs on the entire dataset
    with torch.no_grad():  # No gradients needed for inference
        predictions = model(X).numpy()

        # Extract x, y, f values
    x_pred = predictions[:, 0]  # First column corresponds to x
    y_pred = predictions[:, 1]  # Second column corresponds to y
    f_pred = predictions[:, 2]  # Third column corresponds to f

    # Generate 2D histograms
    plot_2d_histogram(x_pred, y_pred, 'x', 'y', '2D Distribution of x and y')
    plot_2d_histogram(x_pred, f_pred, 'x', 'f', '2D Distribution of x and f')
    plot_2d_histogram(y_pred, f_pred, 'y', 'f', '2D Distribution of y and f')

    x_pred = y[:, 0]  # First column corresponds to x
    y_pred = y[:, 1]  # Second column corresponds to y
    f_pred = y[:, 2]  # Third column corresponds to f
    # Generate 2D histograms
    plot_2d_histogram(x_pred, y_pred, 'x', 'y', '2D Distribution of x and y')
    plot_2d_histogram(x_pred, f_pred, 'x', 'f', '2D Distribution of x and f')
    plot_2d_histogram(y_pred, f_pred, 'y', 'f', '2D Distribution of y and f')
