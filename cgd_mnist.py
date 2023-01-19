import numpy as np
from CGD_active import *
import time

print("Loading data...")
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

print("Initializing...")
num_iter = 2001
Pv = 4
Ph = 8
hidden = 1024
init_lambda = 0.01
q = 0.1
batch = 60
max_norm = 1
sigma = 0.1
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
    dataset="MNIST",
    sigma=sigma,
    verify_lrr=False
)
cgd.shuffle_data()
cgd.confined_init()
total_steps = num_iter * batch
t1 = time.perf_counter()
cgd.eq_train(attack_strength=4)
t2 = time.perf_counter()
print(f"Time consumed {t2-t1}s")