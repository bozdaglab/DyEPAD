# DyEPAD
Dynamic GNN framekwork to predict progression of Alzheimer's Disease

![DyMEAP__](https://github.com/user-attachments/assets/6c1ff3e3-d0ec-4e85-8fba-775ca589c416)
Graphical illustration of DyEPAD. (a) DyEPAD utilizes GCN layers to derive node embeddings from graph-structured EHR data, and subsequently employs GRU layers to capture evolutionary patterns. (b) The derived embeddings are then processed through tensor algebraic operations for frequency domain analysis, capturing the complete range of evolutionary patterns.
# Run DyEPAD
After having all files in the directory, run `DyEPAD.py`.
### Inputs
National Alzheimer’s Coordinating Center (NACC) datasets  dataset was employed for the training and testing of DyEPAD. 
   * `label_NACC.pkl`: The ground truth labels. (# of samples, # of time points, 1) (1414, 7, 1)
   * `longitudinal_NACC_data.pkl`: The time series EHR data is organized in a format where the dimensions correspond to the number of samples, the number of time points, and the number of features.

We note that users can adjust the training and testing visits in `DyEPAD.py`. 
   * Line 62: the number of visits to train.
   * Line 63: the visit to make predictions.
