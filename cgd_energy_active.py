import numpy as np
from CGD_active_energy import *
import time

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

train_path = r"./sent140_train_wo_length.npz"
test_path = r"./sent140_test_wo_length.npz"
max_length = 200
vocab_size = 34393


class SequenceDataset(Dataset):
    def __init__(self, dataframe, sequence_length=5):
        self.sequence_length = sequence_length
        # print(dataframe.values)
        # self.X = torch.tensor(dataframe.values).float()
        self.X = dataframe.values
        # print('self.X', self.X)
        # print(type(self.X))

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, i): 
        if i >= self.sequence_length:
            i_start = i - self.sequence_length
            x = self.X[i_start:i, :]
        else:
            p = []
            p.append(self.X[0])
            padding = np.repeat(p, self.sequence_length - i, axis=0)
            # padding = self.X[0].repeat(self.sequence_length - i, axis=1)
            # padding.reshape(-1,1)
            # print(type(padding),padding.shape)
            x = self.X[0:i, :]
            # x = torch.cat((padding, x), 0)
            # print(type(x),x.shape)
            # print(x)
            x = np.concatenate((padding,x),0)

        return x, self.X[i][0]

def load_data(data_path,sequence_length,test_start="2015-12"):

    df = pd.read_csv(data_path, parse_dates=True, index_col="Datetime")
    df = df.sort_index(axis=0)

    for c in df.columns:
        mean = df[c].mean()
        stdev = df[c].std()

        df[c] = (df[c] - mean) / stdev

    df_train = df.loc[:test_start].copy()
    df_test = df.loc[test_start:].copy()

    train_dataset = SequenceDataset(
        df_train,
        sequence_length=sequence_length
    )
    test_dataset = SequenceDataset(
        df_test,
        sequence_length=sequence_length
    )

    return train_dataset, test_dataset


sequence_length = 100

print('loading data...')
data_path = r'./DAYTON_hourly.csv'
train_dataset, test_dataset = load_data(data_path,sequence_length)
print('Train: ', len(train_dataset), ' Test: ', len(test_dataset))
# outfile = 'train_energy.npz' # not used by any reference

train_x = []
train_y = []
for sample in train_dataset:
    train_x.append(sample[0])
    train_y.append(sample[1])

print('train set split done')

test_x = []
test_y = []

for sample in test_dataset:
    # print(sample)
    test_x.append(sample[0])
    test_y.append(sample[1])

print('train set split done')

train_x = torch.tensor(train_x, dtype=torch.float)
test_x = torch.tensor(test_x, dtype=torch.float)
train_y = torch.tensor(train_y, dtype=torch.float)
test_y = torch.tensor(test_y, dtype=torch.float)


print("Initializing...")
num_iter = 2001
Pv = 4
Ph = 8
hidden = 16
init_lambda = 0.1
batch = 20
max_norm = 1

sigma = 0.01
stride = 1

cgd = CgdTorch(
    num_iter=num_iter,
    train_imgs=train_x,
    train_labels=train_y,
    test_imgs=test_x,
    test_labels=test_y,
    Pv=Pv,
    Ph=Ph,
    n_H=hidden,
    init_lambda=init_lambda,
    batch=batch,
    dataset="ENERGY",
    verify_lrr=True,
    sigma=sigma,
    loss=torch.nn.MSELoss(),
    num_labels=1
)
cgd.shuffle_data()
cgd.confined_init()
t1 = time.perf_counter()
cgd.eq_train(threshold=0.65, attack_strength=0.5)
t2 = time.perf_counter()
print(f"Time consumed {t2-t1}s")