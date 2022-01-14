import numpy as np
from CGD_BDP_torch import *

data = np.load("./mnist_train.npz")
train_imgs = data['arr_0']
train_labels = data['arr_1']
data = np.load("./mnist_test.npz")
test_imgs = data['arr_0']
test_labels = data['arr_1']

train_imgs = torch.tensor(train_imgs, dtype=torch.float)
test_imgs = torch.tensor(test_imgs, dtype=torch.float)
train_labels = torch.tensor(train_labels, dtype=torch.long)
test_labels = torch.tensor(test_labels, dtype=torch.long)

print(f"Data loaded, training images: {train_imgs.size(0)}, testing images: {test_imgs.size(0)}")

print("Initializing...")
num_iter = 100
Pv = 2
Ph = 4
hidden = 256
init_lambda = 0.01
q = 0.1
batch = 10
max_norm = 1
sigma = 0.01
cgd = CGD_torch(
    num_iter=num_iter,
    train_imgs=train_imgs,
    train_labels=train_labels,
    test_imgs=test_imgs,
    test_labels=test_labels,
    Pv=Pv,
    Ph=Ph,
    n_H=hidden,
    init_lambda=init_lambda,
    batch=batch,
    dataset="MNIST",
    sampling_prob=0.1,
    max_grad_norm=max_norm,
    sigma=sigma,
    stride=10
)
cgd.shuffle_data()
cgd.confined_init()

print("Start training...")
cgd.eq_train()
print("Training complete")