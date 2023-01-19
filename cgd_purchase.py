import numpy as np
from CGD_active import *
from bayesian_privacy_accountant import BayesianPrivacyAccountant
import time

data = np.load("./purchase.npz")

imgs = data['arr_0']
labels = data['arr_1']
labels -= 1
train_imgs = imgs[:180000]
train_labels = labels[:180000]
test_imgs = imgs[180000:]
test_labels = labels[180000:]

train_imgs = torch.tensor(train_imgs, dtype=torch.float)
test_imgs = torch.tensor(test_imgs, dtype=torch.float)
train_labels = torch.tensor(train_labels, dtype=torch.long)
test_labels = torch.tensor(test_labels, dtype=torch.long)

print(train_imgs.size(1))

print("Initializing...")
num_iter = 201
Pv = 4
Ph = 100
hidden = 1024
init_lambda = 0.01
q = 0.1
batch = 20
max_norm = 1
sigma = 0.01
cgd = CgdTorch(
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
    dataset="PURCHASE",
    sigma=sigma
)
cgd.shuffle_data()
cgd.confined_init()
total_steps = num_iter * batch
t1 = time.time()
cgd.eq_train()
t2 = time.time()
print(f"Time consumed {t2-t1}s")