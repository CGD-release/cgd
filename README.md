# Confined Gradient Descent

### Prerequisites
1. Pytorch 1.9.0
2. numpy 1.21.2
3. scikit-learn 1.2.0
4. pandas 1.3.2

### Passive scripts
Unzip "pickled_mnist.pkl.zip" and run the dataset name.
- mnist.py

### Active scripts
1. CGD experiment scripts: Start with _cgd_, followed by '_', then the dataset name.
 - cgd_cifar.py
2. FedAvg experiment scripts: Start with _fedavg_, followed by '_', then the dataset name.
 - fedavg_cifar.py

 ### Outputs
  Collected in _./output_, columns including
  epoch	test_acc	test_loss	train_acc	train_loss	mia_acc	idv_acc
  The first 5 columns are semantically self-explained.
  The _mia_acc_ column shows the membership inference attack prediction
  accuracy (attack accuracy).
  The _idv_acc_ shows the individual prediction accuracy for different CGD members. This column
  is not meaningful for FedAvg trainings.
