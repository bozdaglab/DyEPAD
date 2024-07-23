"author: Cagri Ozdemir, Ph.D."
import numpy as np
from sklearn.neighbors import kneighbors_graph
from scipy.sparse import lil_matrix
from torch_geometric.utils import from_scipy_sparse_matrix
import scipy
import scipy.sparse as sp
from sklearn.neighbors import NearestNeighbors
import torch.nn.functional as F
import torch
from torch_geometric.nn import GCNConv
import torch.nn as nn
import numpy as np
import pywt
from scipy.fftpack import dct, idct
from termcolor import colored


def z_normalize(A):
    """
    Apply z-normalization to a feature matrix A along axis 1 (per sample).

    Parameters:
    A (ndarray): The feature matrix where rows are samples and columns are features.

    Returns:
    ndarray: The normalized feature matrix.
    """
    # Compute the mean and standard deviation for each sample (row)
    means = np.mean(A, axis=1, keepdims=True)
    stds = np.std(A, axis=1, keepdims=True)

    # Apply z-normalization
    A_norm = (A - means) / stds

    return A_norm

def create_undirected_knn_graph(A, k):
    """
    Creates an undirected k-NN graph including self-connections.

    Parameters:
    A (numpy.ndarray): The feature matrix.
    k (int): The number of nearest neighbors.

    Returns:
    numpy.ndarray: The adjacency matrix of the undirected k-NN graph.
    """
    # Compute the k-NN graph for the entire feature matrix A
    knn_graph = kneighbors_graph(A, k, mode='connectivity', include_self=True, metric='cosine')  # include self-connections

    # Convert knn_graph to lil_matrix for efficient modification
    knn_graph = lil_matrix(knn_graph)

    # Ensure self-connections are explicitly set to 1
    knn_graph.setdiag(np.ones(A.shape[0]))

    # Ensure adjacency matrix is symmetric
    adjacency_matrix = knn_graph.maximum(knn_graph.T).toarray()
    if not np.allclose(adjacency_matrix, adjacency_matrix.T, atol=1e-7):
        print("\033[91mAdjacency matrix is not symmetric.\033[0m")
    adj_torch = from_scipy_sparse_matrix(scipy.sparse.coo_matrix(adjacency_matrix))[0]

    return adj_torch

def create_label_based_knn_graph(A: np.ndarray, labels: np.ndarray, k: int) -> torch.Tensor:
    """
    Creates a k-NN graph with edges only between nodes with the same label.

    Parameters:
    A (numpy.ndarray): The feature matrix.
    labels (np.ndarray): The array of labels for each node.
    k (int): The number of nearest neighbors.

    Returns:
    torch.Tensor: The adjacency matrix of the k-NN graph.
    """
    n_samples = A.shape[0]
    adjacency_matrix = sp.lil_matrix((n_samples, n_samples))

    for label in np.unique(labels):
        # Get indices of nodes with the current label
        label_indices = np.where(labels == label)[0]
        if len(label_indices) > 1:
            # Create k-NN graph for nodes with the same label
            knn = NearestNeighbors(n_neighbors=min(k, len(label_indices)), metric='cosine')
            knn.fit(A[label_indices])
            distances, indices = knn.kneighbors(A[label_indices])

            for i, neighbors in enumerate(indices):
                for neighbor in neighbors:
                    adjacency_matrix[label_indices[i], label_indices[neighbor]] = 1

    # Ensure self-connections
    adjacency_matrix.setdiag(np.ones(n_samples))

    # Ensure the adjacency matrix is symmetric
    adjacency_matrix = adjacency_matrix.maximum(adjacency_matrix.T)

    # Convert to torch tensor
    adj_torch = from_scipy_sparse_matrix(sp.coo_matrix(adjacency_matrix))[0]

    return adj_torch

class CustomGRU(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(CustomGRU, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        # Parameters for reset gate
        self.W_ir = nn.Linear(input_size, hidden_size)
        self.W_hr = nn.Linear(hidden_size, hidden_size)
        self.b_ir = nn.Parameter(torch.zeros(hidden_size))
        self.b_hr = nn.Parameter(torch.zeros(hidden_size))

        # Parameters for update gate
        self.W_iz = nn.Linear(input_size, hidden_size)
        self.W_hz = nn.Linear(hidden_size, hidden_size)
        self.b_iz = nn.Parameter(torch.zeros(hidden_size))
        self.b_hz = nn.Parameter(torch.zeros(hidden_size))

        # Parameters for new state
        self.W_in = nn.Linear(input_size, hidden_size)
        self.W_hn = nn.Linear(hidden_size, hidden_size)
        self.b_in = nn.Parameter(torch.zeros(hidden_size))
        self.b_hn = nn.Parameter(torch.zeros(hidden_size))

    def forward(self, A, B):
        # A, B: (batch_size, input_size)

        # Reset gate
        r_t = torch.sigmoid(self.W_ir(A) + self.b_ir + self.W_hr(B) + self.b_hr)

        # Update gate
        z_t = torch.sigmoid(self.W_iz(A) + self.b_iz + self.W_hz(B) + self.b_hz)

        # New state
        n_t = torch.tanh(self.W_in(A) + self.b_in + r_t * (self.W_hn(B) + self.b_hn))

        # Updated A based on B
        A_updated = (1 - z_t) * n_t + z_t * B

        return A_updated


class DyIGCN(torch.nn.Module):
    def __init__(self, in_size0=16, hid_size0=8, out_size=2):
        super(DyIGCN, self).__init__()
        self.conv0 = GCNConv(in_size0, hid_size0 * 2)
        self.fc1 = nn.Linear(hid_size0 * 2, hid_size0)
        self.fc2 = nn.Linear(hid_size0, out_size)
        self.gru_model = CustomGRU(hid_size0 * 2, hid_size0 * 2)

    def forward(self, data0, Zt, first):
        if first == True:
            x0, edge_index0 = data0.x, data0.edge_index

            x_emb0 = self.conv0(x0, edge_index0)
            x_emb1 = F.relu(x_emb0)
            x_emb1 = F.dropout(x_emb1, self.training)

            zbar = x_emb1

        if first == False:
            x0, edge_index0 = data0.x, data0.edge_index

            x_emb0 = self.conv0(x0, edge_index0)
            x_emb1 = F.relu(x_emb0)
            x_emb1 = F.dropout(x_emb1, self.training)
            zbar = self.gru_model(x_emb1, Zt)

        out = F.dropout(F.relu(self.fc1(zbar)), self.training)
        out_final = self.fc2(out)

        return out_final, zbar


def polynomial_kernel(X_train, X_test, degree, alpha, c):
    """
    Compute the polynomial kernel matrix between training and test data.

    Parameters:
    X_train (numpy.ndarray): Training data, shape (n_train_samples, n_features)
    X_test (numpy.ndarray): Test data, shape (n_test_samples, n_features)
    degree (int): Degree of the polynomial
    alpha (float): Scale factor
    c (float): Constant term

    Returns:
    numpy.ndarray: Polynomial kernel matrix, shape (n_train_samples, n_test_samples)
    """
    # Compute the dot product between training and test data
    dot_product = np.dot(X_train, X_test.T)

    # Compute the polynomial kernel using the formula: K(x_train, x_test) = (alpha * dot_product + c)^degree
    K = (alpha * dot_product + c) ** degree

    return K


def t_poly_kernel(A_tr, A_tst, degree, alpha, c, transform=""):
    (a, b0, c0) = A_tr.shape
    (a, b1, c1) = A_tst.shape
    if transform != "haar" and transform != "dft" and transform != "dct" and transform != "hartley" and transform != "identity":
        print(colored('Warning, transform is not applicable! Usable transforms: haar, dft, dct, hartley or identity.',
                      'red'))
        return
    if transform == "haar":
        C = np.zeros((a, b0, b1))
        z1 = int(a)
        z2 = int(a / 2)
        coeffsA = pywt.dwt(A_tr, 'haar', axis=0)
        cA, cD = coeffsA
        D = np.concatenate((cA, cD))
        coeffsA_tst = pywt.dwt(A_tst, 'haar', axis=0)
        cAt, cDt = coeffsA_tst
        D_tst = np.concatenate((cAt, cDt))
        for j in range(a):
            C[j, :, :] = polynomial_kernel(D[j, :, :], D_tst[j, :, :], degree, alpha, c)
    if transform == "dct":
        C = np.zeros((a, b0, b1))
        D = dct(A_tr, axis=0, norm='ortho')
        D_tst = dct(A_tst, axis=0, norm='ortho')
        for j in range(a):
            C[j, :, :] = polynomial_kernel(D[j, :, :], D_tst[j, :, :], degree, alpha, c)
    if transform == "dft":
        C = np.zeros((a, b0, b1), dtype='complex')
        D = np.fft.fft(A_tr, axis=0)
        D_tst = np.fft.fft(A_tst, axis=0)
        for j in range(a):
            C[j, :, :] = polynomial_kernel(D[j, :, :], D_tst[j, :, :], degree, alpha, c)
            C = C.real
    if transform == "hartley":
        C = np.zeros((a, b0, b1))
        D = np.fft.fft(A_tr, axis=0)
        D_r = np.real(D)
        D_im = np.imag(D)
        D1 = D_r + D_im

        Dt = np.fft.fft(A_tst, axis=0)
        D_rt = np.real(Dt)
        D_imt = np.imag(Dt)
        D2 = D_rt + D_imt
        for j in range(a):
            C[j, :, :] = polynomial_kernel(D1[j, :, :], D2[j, :, :], degree, alpha, c)

    if transform == "identity":
        C = np.zeros((a, b0, b1))
        for i in range(a):
            C[i, :, :] = polynomial_kernel(A_tr[i, :, :], A_tst[i, :, :], degree, alpha, c)
    return C