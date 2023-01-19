import time
from CGD_active import *

data = np.load("./cifar_resnet56.npz")

train_imgs = data['arr_0']
train_labels = data['arr_1']
test_imgs = data['arr_3']
test_labels = data['arr_4']

train_imgs = torch.tensor(train_imgs, dtype=torch.float)
test_imgs = torch.tensor(test_imgs, dtype=torch.float)
train_labels = torch.tensor(train_labels, dtype=torch.long)
test_labels = torch.tensor(test_labels, dtype=torch.long)

print(f"Data loaded, training images: {train_imgs.size(0)}, testing images: {test_imgs.size(0)}")

print("Initializing...")
num_iter = 10001
Pv = 4
Ph = 8
hidden = 1024
init_lambda = 0.1
batch = 10
cgd = FedAvg(
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
    dataset="CIFAR_FedAvg"
)
cgd.shuffle_data()
cgd.init_fed_avg()
t1 = time.perf_counter()
cgd.eq_train(threshold=0.60, attack_strength=3)
t2 = time.perf_counter()
print(f"Time consumed {t2-t1}s")