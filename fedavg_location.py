from CGD_active import *
import time

data = np.load("./location.npz")

imgs = data['arr_0']
labels = data['arr_1']
train_imgs = imgs[:4000]
train_labels = labels[:4000]
test_imgs = imgs[4000:]
test_labels = labels[4000:]

train_imgs = torch.tensor(train_imgs, dtype=torch.float)
test_imgs = torch.tensor(test_imgs, dtype=torch.float)
train_labels = torch.tensor(train_labels, dtype=torch.long)
test_labels = torch.tensor(test_labels, dtype=torch.long)

print("Initializing...")
num_iter = 2001
Pv = 4
Ph = 8
hidden = 1024
init_lambda = 0.1
batch = 10
max_norm = 1
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
    dataset="LOCATION_FedAvg"
)
cgd.shuffle_data()
cgd.init_fed_avg()
t1 = time.perf_counter()
cgd.eq_train(threshold=0.65, attack_strength=0.2)
t2 = time.perf_counter()
print(f"Time consumed {t2-t1}s")