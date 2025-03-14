import numpy as np
import nibabel as nib
from scipy.io import loadmat, savemat
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torchvision.models as models

def load_nii(filename):
    data = nib.load(filename)
    data = data.get_fdata()
    data = np.array(data)
    return data

def save_nii(filename,data):
    # data = torch.squeeze(data)
    # data = data.detach().clone().cpu()
    data = nib.Nifti1Image(data,np.eye(4),dtype=np.uint8)
    nib.save(data,filename)

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

def one_hot_encode(labels, num_classes):
    """
    Converts class indices into one-hot encoded vectors efficiently.

    Parameters:
    - labels: Tensor of shape (batch_size,) containing class indices.
    - num_classes: Total number of classes.

    Returns:
    - one_hot: A sparse representation of one-hot encoded vectors.
    """
    batch_size = labels.shape[0]
    one_hot = torch.zeros((batch_size, num_classes), dtype=torch.float32, device=labels.device)
    one_hot.scatter_(1, labels.unsqueeze(1), 1)  # Efficiently set correct index to 1
    return one_hot


class MobileNetV2Classifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(MobileNetV2Classifier, self).__init__()
        self.dictionary = load_nii("./data/1/theoretical_efficiency.nii")
        self.dictionary = torch.LongTensor(self.dictionary).to(device)
        self.ve_parameter = loadmat("./dictionary/ve_param.mat")
        self.ve_parameter = load_rve_encoding(self.ve_parameter).to(device)
        self.softmax = nn.Softmax(dim=1)  # Convert logits to probabilities

        # Load Pretrained MobileNetV2
        self.mobilenet = models.mobilenet_v2(pretrained=True)

        # Modify the first convolutional layer to accept 62 features
        self.mobilenet.features[0][0] = nn.Conv2d(1, 32, kernel_size=(3, 1), stride=2, padding=(1, 0), bias=False)

        # Modify the classifier layer for 173,225 classes
        self.mobilenet.classifier[1] = nn.Linear(1280, num_classes)

    def forward(self, x):
        x = x.unsqueeze(1).unsqueeze(1)  # Reshape to (batch_size, 1, 62, 1) for Conv2d
        return self.mobilenet(x)
    def compute_constraint_loss(self, logits, x):
        """
        Computes the constraint loss:
        1. Convert logits to probabilities.
        2. Find the predicted class index (highest probability).
        3. Use `constraint_function` to map the class index to 3-size y.
        4. Compute MSE loss with target y.
        """
        probabilities = self.softmax(logits)  # Convert logits to probabilities
        predicted_indices = torch.argmax(probabilities, dim=1)  # Get predicted class indices

        y_pred_3 = self.dictionary[predicted_indices].float()
        # y_pred_3 = y_pred_3[:,:22].to(device)
        # Compute the mean of each column
        mean1 = x.mean(dim=1, keepdim=True)
        mean2 = y_pred_3.mean(dim=1, keepdim=True)

        # Compute the standard deviation of each column
        std1 = x.std(dim=1, unbiased=False, keepdim=True)
        std2 = y_pred_3.std(dim=1, unbiased=False, keepdim=True)

        # Compute covariance
        covariance = ((x - mean1) * (y_pred_3 - mean2)).mean(dim=1)
        correlation = covariance / (std1.squeeze() * std2.squeeze())

        # Filter values greater than 0.6
        filtered_corr = correlation[correlation > 0.6]

        # Compute the average of the remaining correlations
        if len(filtered_corr) > 0:
            avg_correlation = filtered_corr.mean().item()
        else:
            avg_correlation = 0

        return 1-avg_correlation

    def constraint_loss(self, logits, x):

        probabilities = self.softmax(logits)  # Convert logits to probabilities
        predicted_indices = torch.argmax(probabilities, dim=1)  # Get predicted class indices
        y_pred_4 = dictionary[predicted_indices].float()

        labeling_efficiencies = compute_labeling_efficiencies_pytorch(y_pred_4, self.ve_parameter)
        ns, ne = labeling_efficiencies.shape
        labeling_efficiencies = torch.cat([torch.ones(ns, 2).to(labeling_efficiencies.device), labeling_efficiencies],
                                          dim=1)
        return torch.mean((x - labeling_efficiencies) ** 2)


class ResNetClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(ResNetClassifier, self).__init__()
        self.dictionary = load_nii("./data/1/theoretical_efficiency.nii")
        self.dictionary = torch.LongTensor(self.dictionary).to(device)
        self.ve_parameter = loadmat("./dictionary/ve_param.mat")
        self.ve_parameter = load_rve_encoding(self.ve_parameter).to(device)
        self.softmax = nn.Softmax(dim=1)  # Convert logits to probabilities

        # Load Pretrained ResNet
        self.resnet = models.resnet18(pretrained=True)  # Use ResNet-18 for efficiency

        # Modify the first layer to accept 62 features instead of images
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 1), stride=2, padding=(3, 0), bias=False)

        # Modify the fully connected layer for 173,225 classes
        self.resnet.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = x.unsqueeze(1).unsqueeze(1)  # Reshape to (batch_size, 1, 62, 1) for Conv2d
        return self.resnet(x)

    def compute_constraint_loss(self, logits, x):
        """
        Computes the constraint loss:
        1. Convert logits to probabilities.
        2. Find the predicted class index (highest probability).
        3. Use `constraint_function` to map the class index to 3-size y.
        4. Compute MSE loss with target y.
        """
        probabilities = self.softmax(logits)  # Convert logits to probabilities
        predicted_indices = torch.argmax(probabilities, dim=1)  # Get predicted class indices

        y_pred_3 = self.dictionary[predicted_indices].float()
        # y_pred_3 = y_pred_3[:,:22].to(device)
        # Compute the mean of each column
        mean1 = x.mean(dim=1, keepdim=True)
        mean2 = y_pred_3.mean(dim=1, keepdim=True)

        # Compute the standard deviation of each column
        std1 = x.std(dim=1, unbiased=False, keepdim=True)
        std2 = y_pred_3.std(dim=1, unbiased=False, keepdim=True)

        # Compute covariance
        covariance = ((x - mean1) * (y_pred_3 - mean2)).mean(dim=1)
        correlation = covariance / (std1.squeeze() * std2.squeeze())

        # Filter values greater than 0.6
        filtered_corr = correlation[correlation > 0.6]

        # Compute the average of the remaining correlations
        if len(filtered_corr) > 0:
            avg_correlation = filtered_corr.mean().item()
        else:
            avg_correlation = 0

        return 1-avg_correlation

    def constraint_loss(self, logits, x):

        probabilities = self.softmax(logits)  # Convert logits to probabilities
        predicted_indices = torch.argmax(probabilities, dim=1)  # Get predicted class indices
        y_pred_4 = dictionary[predicted_indices].float()

        labeling_efficiencies = compute_labeling_efficiencies_pytorch(y_pred_4, self.ve_parameter)
        ns, ne = labeling_efficiencies.shape
        labeling_efficiencies = torch.cat([torch.ones(ns, 2).to(labeling_efficiencies.device), labeling_efficiencies],
                                          dim=1)
        return torch.mean((x - labeling_efficiencies) ** 2)

# CNN Model for Multi-Class Classification
class CNNClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(CNNClassifier, self).__init__()

        self.dictionary = load_nii("./data/1/theoretical_efficiency.nii")
        self.dictionary = torch.LongTensor(self.dictionary).to(device)
        self.ve_parameter = loadmat("./dictionary/ve_param.mat")
        self.ve_parameter = load_rve_encoding(self.ve_parameter).to(device)

        # 1D CNN Layers
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1)

        # Fully Connected Layers
        self.fc1 = nn.Linear(128 * input_dim, 512)
        self.fc2 = nn.Linear(512, 1024)
        self.fc3 = nn.Linear(1024, num_classes)  # 173225 classes

        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)  # Convert logits to probabilities

    def forward(self, x):
        """
        x shape: (batch_size, 62) → Reshape to (batch_size, 1, 62) for CNN
        """
        x = x.unsqueeze(1)  # Add channel dimension → (batch_size, 1, 62)
        x = x.to(device)

        x = self.relu(self.conv1(x))  # (batch_size, 32, 62)
        x = self.relu(self.conv2(x))  # (batch_size, 64, 62)
        x = self.relu(self.conv3(x))  # (batch_size, 128, 62)

        x = x.view(x.shape[0], -1)  # Flatten → (batch_size, 128 * 62)

        x = self.relu(self.fc1(x))  # (batch_size, 512)
        x = self.relu(self.fc2(x))  # (batch_size, 1024)
        logits = self.fc3(x) # (batch_size, 173225)

        return logits  # Output raw class scores for cross-entropy loss

    def compute_constraint_loss(self, logits, x):
        """
        Computes the constraint loss:
        1. Convert logits to probabilities.
        2. Find the predicted class index (highest probability).
        3. Use `constraint_function` to map the class index to 3-size y.
        4. Compute MSE loss with target y.
        """
        probabilities = self.softmax(logits)  # Convert logits to probabilities
        predicted_indices = torch.argmax(probabilities, dim=1)  # Get predicted class indices

        y_pred_3 = self.dictionary[predicted_indices].float()
        # y_pred_3 = y_pred_3[:,:22].to(device)
        # Compute the mean of each column
        mean1 = x.mean(dim=1, keepdim=True)
        mean2 = y_pred_3.mean(dim=1, keepdim=True)

        # Compute the standard deviation of each column
        std1 = x.std(dim=1, unbiased=False, keepdim=True)
        std2 = y_pred_3.std(dim=1, unbiased=False, keepdim=True)

        # Compute covariance
        covariance = ((x - mean1) * (y_pred_3 - mean2)).mean(dim=1)
        correlation = covariance / (std1.squeeze() * std2.squeeze())

        # Filter values greater than 0.6
        filtered_corr = correlation[correlation > 0.6]

        # Compute the average of the remaining correlations
        if len(filtered_corr) > 0:
            avg_correlation = filtered_corr.mean().item()
        else:
            avg_correlation = 0

        return 1-avg_correlation

    def constraint_loss(self, logits, x):

        probabilities = self.softmax(logits)  # Convert logits to probabilities
        predicted_indices = torch.argmax(probabilities, dim=1)  # Get predicted class indices
        y_pred_4 = dictionary[predicted_indices].float()

        labeling_efficiencies = compute_labeling_efficiencies_pytorch(y_pred_4, self.ve_parameter)
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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    x_values = np.arange(64, -65, -2)  # From -64 to 64 with step 2
    y_values = np.arange(64, -65, -2)  # From -64 to 64 with step 2
    f_values = np.arange(-2, 2.1, 0.1)
    dictionary = []
    for f in f_values:  # Iterate over f
        for x in x_values:  # Iterate over x
            for y in y_values:  # Iterate over y
                dictionary.append([x, y, f])

    x = load_nii("./data/1/dMs_vectors.nii")
    # Encoding Steps
    # x = x[:,:22]

    y = load_nii("./data/1/indices_ccmax.nii")
    y = np.squeeze(y)

    y0 = []
    for i in range(49152):
        y0.append(dictionary[int(y[i])-1])
    y0 = np.array(y0)
    dictionary = torch.LongTensor(dictionary).to(device)

    input_dim = 62
    num_classes = 173225
    batch_size = 1024  # Reduce for efficiency
    num_epochs = 50
    lambda_constraint = 0.4  # Weight for constraint loss
    model_save_path = "./model/CNN_CCconstraint_60_2.pth"
    # Generate Fake Dataset
    X = torch.tensor(x).float().to(device)
    Y = torch.tensor(y).long().to(device)
    #
    # # Use DataLoader
    dataset = TensorDataset(X, Y)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Initialize Model & Optimizer
    # model = CNNClassifier(input_dim, num_classes).to(device)
    # model = ResNetClassifier(input_dim, num_classes).to(device)
    model = MobileNetV2Classifier(input_dim, num_classes).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    cross_entropy_loss = nn.CrossEntropyLoss()

    # Training Loop
    for epoch in range(num_epochs):
        total_loss = 0
        total_cls_loss = 0
        total_constraint_loss = 0

        for x_batch, y_batch in data_loader:
            optimizer.zero_grad()

            logits = model(x_batch)  # Forward pass
            cls_loss = cross_entropy_loss(logits, y_batch)  # Classification loss
            # constraint_loss = model.compute_constraint_loss(logits, x_batch)  # Constraint loss
            constraint_loss = model.constraint_loss(logits, x_batch)

            # Total loss with balance factor
            loss = cls_loss + 0.5 * constraint_loss
            # loss = cls_loss
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_cls_loss += cls_loss.item()
            total_constraint_loss += constraint_loss

        avg_cls_loss = total_cls_loss / len(data_loader)
        avg_constraint_loss = total_loss / len(data_loader) - avg_cls_loss

        print(
            f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss / len(data_loader)}, Cls Loss: {avg_cls_loss}, Constraint Loss: {avg_constraint_loss}")

    # Save trained model
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")

    # Load the trained model for inference

    model.load_state_dict(torch.load(model_save_path, map_location=device))
    model.eval()  # Set model to evaluation mode

    # Define mini-batch size for evaluation (to prevent memory overflow)
    eval_batch_size = 512  # Adjust based on available memory

    # Create DataLoader for evaluation
    eval_dataset = TensorDataset(X)
    eval_loader = DataLoader(eval_dataset, batch_size=eval_batch_size, shuffle=False)

    predicted_indices_list = []
    predicted_indices_list2 = []

    # Get final predicted outputs on the entire dataset
    # Perform mini-batch inference
    with torch.no_grad():  # No gradient computation to save memory
        for batch in eval_loader:
            x_batch = batch[0].to(device)  # Move batch to GPU
            logits = model(x_batch)  # Forward pass

            # Apply softmax to get probabilities
            probabilities = torch.softmax(logits, dim=1)

            # Get the highest probability index (predicted class)
            predicted_indices = torch.argmax(probabilities, dim=1)

            y_predictions = dictionary[predicted_indices]
            predicted_indices_list.append(y_predictions.cpu())
            predicted_indices_list2.append(predicted_indices.cpu())

    # Concatenate all predictions
    final_predictions = torch.cat(predicted_indices_list).numpy()
    final_predictions2 = torch.cat(predicted_indices_list2).numpy()

    savemat("./results/60_2.mat", {"distribution_XYF_DL": final_predictions2})

    # probabilities = nn.Softmax(predictions)  # Convert logits to probabilities
    # predicted_indices = torch.argmax(probabilities, dim=1)  # Get predicted class indices
    # y_predictions = dictionary[predicted_indices].numpy()
        # Extract x, y, f values
    x_pred = final_predictions[:, 0]  # First column corresponds to x
    y_pred = final_predictions[:, 1]  # Second column corresponds to y
    f_pred = final_predictions[:, 2]  # Third column corresponds to f

    # Generate 2D histograms
    plot_2d_histogram(x_pred, y_pred, 'x', 'y', '2D Distribution of x and y')
    plot_2d_histogram(x_pred, f_pred, 'x', 'f', '2D Distribution of x and f')
    plot_2d_histogram(y_pred, f_pred, 'y', 'f', '2D Distribution of y and f')

    x_pred = y0[:, 0]  # First column corresponds to x
    y_pred = y0[:, 1]  # Second column corresponds to y
    f_pred = y0[:, 2]  # Third column corresponds to f
    # Generate 2D histograms
    plot_2d_histogram(x_pred, y_pred, 'x', 'y', '2D Distribution of x and y')
    plot_2d_histogram(x_pred, f_pred, 'x', 'f', '2D Distribution of x and f')
    plot_2d_histogram(y_pred, f_pred, 'y', 'f', '2D Distribution of y and f')
