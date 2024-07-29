import pickle
import numpy as np
import os
import torch
from torch_geometric.data import Data
import gc
from sklearn.model_selection import train_test_split
import utils
from MKLpy.algorithms import EasyMKL
current_directory = os.getcwd()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

""" 
longitudinal_NACC_data.pkl represents the time series data and is an array with shape of
 (# of samples, # of time points, # of features).
"""

with open(current_directory + '/longitudinal_NACC_data.pkl', 'rb') as f:
  long_data = pickle.load(f)
#z normalization
long_data_norm = np.zeros((long_data.shape[0],long_data.shape[1],long_data.shape[2]))
for i in range(long_data.shape[1]):
    long_data_norm[:, i, :] = utils.z_normalize(long_data[:, i, :])
"""

label_NACC.pkl represents the label data and is an array with shape of (# of samples, # of time points, 1) (1414, 7, 1).
The label values are 0 which represents MCI and 1 which represents AD.
"""
with open(current_directory + '/label_NACC.pkl','rb') as f:
    labels = pickle.load(f)
labels = labels.astype(np.int64)

DATA = {}
for i in range(long_data.shape[1]):
    DATA[i] = Data(x=torch.tensor(long_data_norm[:, i, :],dtype=torch.float32), edge_index=utils.create_undirected_knn_graph(long_data_norm[:, i, :],k=5),
                   y=torch.tensor(labels[:,i,0]).long())


Zt = torch.zeros(())
def train():
    model.train()
    optimizer.zero_grad()
    out0,Ztbar = model(data0.to(device),Zt.to(device),first)
    loss1 = criterion(out0[train_mask].to(device), labels_train[train_mask].to(device))
    loss1.backward()
    optimizer.step()
    return loss1
def validate():
    model.eval()
    with torch.no_grad():
        out0, Ztbar = model(data0.to(device),Zt.to(device),first)
    return out0, Ztbar


max_epochs = 500
insize0 = long_data.shape[2]
sample_size = long_data.shape[0]
hdsize0 = 512
outsize = 2
learning_rate = 0.003
train_snapshots = 6
test_time_point = 7
ac = list()
f1 = list()
aucroc = list()
y_visit7 = DATA[test_time_point-1].y

model = utils.DyIGCN(in_size0=insize0, hid_size0=hdsize0, out_size=outsize)
model.to(device)

for i in [61,76,3,7,120,20,16]:
    train_idx, test_idx = train_test_split(np.arange(len(y_visit7)),
                                           test_size=0.2, shuffle=True, stratify=y_visit7, random_state=i)
    train_mask = np.array([i in set(train_idx) for i in range(sample_size)])
    Emb_tensor = torch.zeros((train_snapshots, long_data.shape[0], hdsize0 * 2))
    for j in range(train_snapshots):
        print('running snapshot #:', j)
        if j == 0:
            first = True
            custom_gru_params = None
        else:
            first = False
        data0 = DATA[j]
        model = utils.DyIGCN(in_size0=insize0, hid_size0=hdsize0, out_size=outsize)
        model.to(device)
        # If it's not the first snapshot, load the saved CustomGRU parameters
        # if custom_gru_params is not None:
        #     model.gru_model.load_state_dict(custom_gru_params)
        labels_train = data0.y
        weights = torch.tensor([1, 2], dtype=torch.float).to(device)
        criterion = torch.nn.CrossEntropyLoss(weight=weights)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        train_loss = []

        for epoch in range(max_epochs):
            print('epoch:', epoch)
            loss0 = train()
            print("train_loss:", loss0)
            train_loss.append(loss0)
        out0, Ztbar = validate()
        Zt = Ztbar.detach()
        Emb_tensor[j, :, :] = Zt

        del model
        gc.collect()
        torch.cuda.empty_cache()


    emb_tensor_train = np.array(Emb_tensor[:,train_idx,:])
    emb_tensor_test = np.array(Emb_tensor[:,test_idx,:])
    gram_tensor_train = utils.t_poly_kernel(emb_tensor_train,emb_tensor_train,degree=3, alpha=1, c=0,transform="hartley")
    gram_tensor_test = utils.t_poly_kernel(emb_tensor_train,emb_tensor_test,degree=3, alpha=1, c=0,transform="hartley")
    # Initialize and train the MKL algorithm
    mkl = EasyMKL(lam=1)  # You can tune the lambda parameter
    # Use list comprehension to create K_train
    K_train = [gram_tensor_train[ell, :, :] for ell in range(train_snapshots)]
    K_test = [gram_tensor_test[ell, :, :].T for ell in range(train_snapshots)]

    mkl.fit(K_train, y_visit7[train_idx])
    # Predict and evaluate
    y_pred = mkl.predict(K_test)
    from sklearn.metrics import accuracy_score, classification_report, f1_score, roc_auc_score

    accuracy = accuracy_score(y_visit7[test_idx], y_pred)
    f1score = f1_score(y_visit7[test_idx], y_pred, average='macro')
    auc = roc_auc_score(y_visit7[test_idx], y_pred)
    print('run #:', i, 'accuarcy:', accuracy, 'F1 score:', f1score, 'AUC:', auc)
    ac.append(accuracy)
    f1.append(f1score)
    aucroc.append(auc)
print('accuracy', 'mean:', np.mean(ac), 'std:', np.std(ac))
print('F1 score', 'mean:', np.mean(f1), 'std:', np.std(f1))
print('AUC score', 'mean:', np.mean(aucroc), 'std:', np.std(aucroc))

from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Compute confusion matrix
cm = confusion_matrix(y_visit7[test_idx], y_pred)
# Define class labels (if your classes are not binary, adjust accordingly)
classes = ['MCI', 'AD']
# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, cmap='Blues', fmt='g', xticklabels=classes, yticklabels=classes)
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix')
plt.show()
