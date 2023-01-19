import torch
import os
import pandas as pd
import numpy as np
from constants import *


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class CGD_model_fc(torch.nn.Module):
    def __init__(self, Pv:int, n_H:int, in_length=28*28, out_length=10):
        super().__init__()
        self.Pv = Pv
        self.n_H = n_H
        self.v_part = torch.nn.Sequential(
            torch.nn.Linear(in_length//Pv, n_H),
            torch.nn.ReLU()
        )
        self.out_class = out_length
        self.h_part = torch.nn.Linear(n_H*Pv, out_length)
        self.optimizer = torch.optim.Adam(self.parameters())
        self.local_data = None
        self.local_label = None
        self.loss_func = None
        self.maliciousness = False
        self.attacked_member = None
        self.attacked_non_member = None
        self.loss_history = []
        self.last_loss = None
        self.member_map = None
        self.mislead_label = None

    def forward(self, x):
        x = x.view(x.size(0), self.Pv, -1)
        out = self.v_part(x)
        out = out.view(out.size(0), -1)
        return self.h_part(out)
    
    def step(self):
        self.optimizer.step()

    def get_flatten_parameters(self):
        """
        Return the flatten parameter of the current model
        :return: the flatten parameters as tensor
        """
        out = torch.zeros(0)
        with torch.no_grad():
            for parameter in self.parameters():
                out = torch.cat([out, parameter.flatten()])
        return out

    def load_parameters(self, parameters: torch.Tensor):
        """
        Load parameters to the current model using the given flatten parameters
        :param parameters: The flatten parameter to load
        :return: None
        """
        start_index = 0
        for param in self.parameters():
            with torch.no_grad():
                length = len(param.flatten())
                to_load = parameters[start_index: start_index + length]
                to_load = to_load.reshape(param.size())
                param.copy_(to_load)
                start_index += length

    def back_prop(self, batch_size=16, attack=False):
        """
        Perform back propagation on each participant
        """
        acc = 0
        loss = 0
        idx = 0
        self.zero_grad()
        while idx < len(self.local_data):
            cur_batch_data = self.local_data[idx: idx+batch_size]
            cur_batch_label = self.local_label[idx: idx+batch_size]
            idx += batch_size
            out = self.forward(cur_batch_data)
            y_pred = torch.max(out, dim=1).indices
            batch_loss = self.loss_func(out, cur_batch_label)
            batch_loss.backward()
            acc += (y_pred == cur_batch_label).sum()
            loss += batch_loss.item()
        acc = acc / self.local_data.size(0)
        self.last_loss = loss
        """
        Malicious participant conduct attack
        """
        if attack and self.maliciousness:
            idx = 0
            """
            First inverse local gradient 
            """
            self.inverse_grad()
            while idx < len(self.local_data):
                """
                Conduct gradient descent using the flipped label
                """
                cur_batch_data = self.local_data[idx: idx + batch_size]
                cur_batch_label = self.mislead_label[idx: idx + batch_size]
                idx += batch_size
                out = self.forward(cur_batch_data)
                batch_loss = self.loss_func(out, cur_batch_label)
                batch_loss.backward()
        return acc, loss

    def apply_grad(self, grad, verify_lrr=False):
        """
        Apply the given global gradient - if consider self attacked (See self.verify_lrr()), then reverse the update
        """
        grad_iter = iter(grad)
        cache = []
        for param in self.parameters():
            cache.append(param.clone().detach())
            grad = next(grad_iter)
            if param.grad is not None:
                param.grad.copy_(grad)
            else:
                param.grad = grad
        self.step()
        if verify_lrr and not self.maliciousness and not self.verify_lrr():
            param_iter = iter(cache)
            with torch.no_grad():
                for param in self.parameters():
                    param.data = next(param_iter)

    def verify_lrr(self):
        """
        The Defensive mechanism locally implemented on each benign participant
        It first collects a loss value history (hard coded to 15) then compare the difference of last loss value with
        the mean value of the memoried loss value. If loss value is larger than 1 * stdev,
        then recognize self as attacked, return True; else return False
        """
        if len(self.loss_history) >= 15:
            stdev = np.std(self.loss_history)
            mean = np.mean(self.loss_history)
            if self.last_loss - mean >= stdev:
                if torch.randint(low=1, high=100, size=(1,)).item() > 50:
                    self.loss_history.pop(0)
                    self.loss_history.append(self.last_loss)
                return False
            self.loss_history.pop(0)
            self.loss_history.append(self.last_loss)
        else:
            if self.last_loss is not None:
                self.loss_history.append(self.last_loss)
        return True

    def grant_malicious_set(self, samples):
        """
        Do label flipping (generate a random label) and mask the member samples
        """
        self.local_data = torch.vstack([self.attacked_member[0], self.attacked_non_member[0]])
        self.local_label = torch.hstack([self.attacked_member[1], self.attacked_non_member[1]])
        self.member_map = ([True] * (samples//2) + [False] * (samples//2))
        self.mislead_label = torch.randint(0, self.out_class, size=self.local_label.size())

    def inverse_grad(self):
        for param in self.parameters():
            param.grad = -param.grad


class CgdTorch:
    def __init__(self,
                 num_iter,
                 train_imgs,
                 train_labels,
                 test_imgs,
                 test_labels,
                 Ph,
                 Pv,
                 n_H,
                 init_lambda,
                 batch,
                 dataset,
                 sigma=0,
                 malicious_factor=0.25,
                 update_fraction=0.3,
                 verify_lrr=False,
                 active_attack=True,
                 stride=10):
        self.num_iter = num_iter
        self.train_imgs = train_imgs
        self.train_labels = train_labels.flatten()
        self.test_imgs = test_imgs
        self.test_labels = test_labels.flatten()
        self.Ph = Ph
        self.Pv = Pv
        self.n_H = n_H
        self.patch_data()
        self.init_lambda = init_lambda
        self.batch_size = train_imgs.size(0) // batch
        self.batch = batch
        self.dataset = dataset
        self.sigma = sigma
        self.stride = stride
        self.models = []
        self.glb_model = None
        self.loss = torch.nn.CrossEntropyLoss()
        self.sum_grad = None
        self.grad_length = 0
        self.label_classes = self.train_labels.unique().size(0)
        self.malicious_count = int(self.Ph * malicious_factor)
        self.update_participants_count = int(update_fraction * (self.Ph + self.malicious_count))
        self.verify_lrr = verify_lrr
        self.active_attack = active_attack
        self.attack_strength = 0

    def patch_data(self):
        diff = self.train_imgs.size(1) - (self.train_imgs.size(1) // self.Pv) * self.Pv
        train_patch = torch.zeros(self.train_imgs.size(0), diff)
        self.train_imgs = torch.hstack([self.train_imgs, train_patch])
        test_patch = torch.zeros(self.test_imgs.size(0), diff)
        self.test_imgs = torch.hstack([self.test_imgs, test_patch])

    """
    Initialize CGD leaning according to given lambda
    """
    def confined_init(self, random_lambda=False):
        sample_per_cap = self.train_imgs.size(0) // self.Ph
        # Initiate benign users
        for i in range(self.Ph):
            model = CGD_model_fc(self.Pv, self.n_H, in_length=self.train_imgs.size(1), out_length=self.label_classes)
            bound = self.init_lambda
            if random_lambda:
                bound = self.init_lambda * torch.randn(1).item()
            param = torch.rand(model.get_flatten_parameters().size()) * bound
            model.load_parameters(param)
            model.loss_func = self.loss
            model.local_data = self.train_imgs[i*sample_per_cap:(i+1)*sample_per_cap]
            model.local_label = self.train_labels[i*sample_per_cap:(i+1)*sample_per_cap]
            self.models.append(model)
            if self.grad_length == 0:
                self.grad_length = model.get_flatten_parameters().size(0)
        # Initiate malicious users, malicious_count = self.Ph * malicious_factor
        for i in range(self.malicious_count):
            model = CGD_model_fc(self.Pv, self.n_H, in_length=self.train_imgs.size(1), out_length=self.label_classes)
            bound = self.init_lambda
            if random_lambda:
                bound = self.init_lambda * torch.randn(1).item()
            param = torch.rand(model.get_flatten_parameters().size()) * bound
            model.load_parameters(param)
            attacked_member_idx = torch.randperm(self.train_imgs.size(0))[:sample_per_cap // 2]
            model.attacked_member = self.train_imgs[attacked_member_idx], self.train_labels[attacked_member_idx]
            attacked_non_member_idx = torch.randperm(self.test_imgs.size(0))[:sample_per_cap//2]
            model.attacked_non_member = self.test_imgs[attacked_non_member_idx], self.test_labels[attacked_non_member_idx]
            model.maliciousness = True
            model.grant_malicious_set(sample_per_cap)
            model.loss_func = self.loss
            self.models.append(model)
        self.test_labels = self.test_labels[self.malicious_count * sample_per_cap // 2:]
        self.test_imgs = self.test_imgs[self.malicious_count * sample_per_cap // 2:]
        print(f"Experiment initialized, {self.Ph}x{self.Pv} honest participants, {self.malicious_count}x{self.Pv} malicious participants")

    def shuffle_data(self):
        shuffled_index = torch.randperm(self.train_imgs.size(0))
        self.train_imgs = self.train_imgs[shuffled_index]
        self.train_labels = self.train_labels[shuffled_index]
        shuffled_index = torch.randperm(self.test_imgs.size(0))
        self.test_imgs = self.test_imgs[shuffled_index]
        self.test_labels = self.test_labels[shuffled_index]

    def grad_reset(self):
        if self.sum_grad is None:
            sum_grad = []
            for param in self.models[0].parameters():
                g_l = torch.zeros(param.size())
                sum_grad.append(g_l)
            self.sum_grad = sum_grad
        else:
            for item in self.sum_grad:
                item.zero_()

    def back_prop(self, idx, step=False, attack=False):
        self.grad_reset()
        sum_acc = 0
        sum_loss = 0
        for i in idx:
            model = self.models[i]
            acc_i, loss_i = model.back_prop(self.batch_size, attack)
            self.collect_grad(model)
            sum_acc += acc_i
            sum_loss += loss_i
            if step:
                model.step()
        sum_acc = sum_acc / self.Ph
        sum_loss = sum_loss / self.Ph
        return sum_acc, sum_loss

    def collect_grad(self, model, add_noise=False):
        grad_iter = iter(self.sum_grad)
        for param in model.parameters():
            grad = param.grad
            if add_noise:
                grad += torch.randn(grad.size()) * self.sigma
            collector = next(grad_iter, None)
            if not model.maliciousness:
                collector += grad
            else:
                collector += grad * self.attack_strength

    def apply_grad(self, idx, grad_mean=True):
        if grad_mean:
            for grad in self.sum_grad:
                grad /= self.Ph
        if idx is None:
            idx = range(len(self.models))
        for i in idx:
            model = self.models[i]
            model.apply_grad(self.sum_grad, self.verify_lrr)

    def evaluate(self):
        outs = torch.zeros(self.test_labels.size(0), self.label_classes)
        loss = 0
        test_x = self.test_imgs
        test_y = self.test_labels.flatten()
        acc_idv = []
        for i in range(self.Ph):
            model = self.models[i]
            with torch.no_grad():
                out = model(test_x)
            outs += out
            pred_y = torch.max(out, dim=1).indices
            acc = torch.sum(pred_y == test_y, dtype=torch.float)
            acc = acc / self.test_labels.size(0)
            acc_idv.append(acc)
            loss_i = self.loss(out, test_y)
            loss += loss_i.item()
        pred_y = torch.max(outs, dim=1).indices
        loss = loss / self.Ph
        g_acc = torch.sum(pred_y == test_y, dtype=torch.float)
        g_acc = g_acc/self.test_labels.size(0)
        return g_acc, acc_idv, loss

    def mia_evaluate(self):
        tp = 0
        fp = 0
        tn = 0
        fn = 0
        for i in range(self.Ph, self.Ph+self.malicious_count):
            malicious_participant = self.models[i]
            with torch.no_grad():
                out = malicious_participant(malicious_participant.local_data)
            pred_y = torch.max(out, dim=1).indices
            for j in range(len(pred_y)):
                pred = pred_y[j]
                pred_correct = pred == malicious_participant.local_label[j]
                is_member = malicious_participant.member_map[j]
                if pred_correct and is_member:
                    tp += 1
                elif not pred_correct and not is_member:
                    fn += 1
                elif not pred_correct and is_member:
                    fp += 1
                else:
                    tn += 1
        # if tp + tn > fn + fp and self.attack_strength < 3:
        #     self.attack_strength += 0.1
        # elif tp + tn < fn + fp and self.attack_strength > 0:
        #     self.attack_strength -= 0.1
        return tp, fp, tn, fn

    def flatten(self, grads):
        out = None
        for g in grads:
            if out is None:
                out = g.flatten()
            else:
                out = torch.cat([out, g.flatten()])
        return out

    def eq_train(self, threshold=0.85, attack_strength=1, selective_apply_grad=False):
        active_attack = False
        self.attack_strength = attack_strength
        epoch_col = []
        train_acc_col = []
        train_loss_col = []
        test_acc_col = []
        test_loss_col = []
        mia_acc_col= []
        idv_acc_col = []
        attack_started_round = 0
        if not os.path.exists(RECORDING_PATH):
            os.makedirs(RECORDING_PATH)
            print("Created a dir", RECORDING_PATH)
        else:
            print("Path found", RECORDING_PATH)
        
        for epoch in range(self.num_iter):
            idx = torch.randperm(self.Ph + self.malicious_count)[:self.update_participants_count]
            acc, loss = self.back_prop(idx=idx, attack=active_attack)
            if epoch % self.stride == 0:
                test_acc, acc_idv, test_loss = self.evaluate()
                epoch_col.append(epoch)
                test_acc_col.append(test_acc.item())
                test_loss_col.append(test_loss)
                train_acc_col.append(acc.item())
                train_loss_col.append(loss)
                idv_acc_col.append(acc_idv)
                tp, fp, tn, fn = self.mia_evaluate()
                mia_acc = (tp + fn) / (tp + fp + tn + fn)
                mia_acc_col.append(mia_acc)
                print(f'Epoch {epoch} - test acc {test_acc:6.4f}, test loss: {test_loss:6.4f}, mia_acc{mia_acc:6.4f}, tp={tp}, fn={fn}, fp={fp}, tn={tn}')
                if test_acc >= threshold and not active_attack:
                    active_attack = True
                    attack_started_round = epoch
                    print(f"Attack starts at round {attack_started_round}")
            if not selective_apply_grad:
                idx = None
            self.apply_grad(idx)
        recorder = pd.DataFrame({"epoch":epoch_col, "test_acc":test_acc_col, "test_loss":test_loss_col, "train_acc":train_acc_col,
                                 "train_loss":train_loss_col, "mia_acc":mia_acc_col, "idv_acc": idv_acc_col})
        recorder.to_csv(RECORDING_PATH+f"{self.dataset}_epoch_{self.num_iter}_Pv_{self.Pv}_Ph_{self.Ph}_nH_{self.n_H}_starts_"
                                       f"{attack_started_round}_lambda_{self.init_lambda}_lrr_{self.verify_lrr}"+time_str+".csv")


class FedAvg(CgdTorch):
    def __init__(self, num_iter, train_imgs, train_labels, test_imgs, test_labels, Ph, Pv, n_H, init_lambda, batch,
                 dataset):
        super(FedAvg, self).__init__(num_iter, train_imgs, train_labels, test_imgs, test_labels, Ph, Pv, n_H, init_lambda, batch,
                         dataset, verify_lrr=False)
        self.global_model = CGD_model_fc(Pv=Pv, n_H=n_H, in_length=self.train_imgs.size(1), out_length=self.label_classes)

    def init_fed_avg(self):
        self.confined_init()
        param = self.global_model.get_flatten_parameters()
        for model in self.models:
            model.load_parameters(param)

    def evaluate(self):
        test_x = self.test_imgs
        test_y = self.test_labels.flatten()
        model = self.global_model
        with torch.no_grad():
            out = model(test_x)
            loss = self.loss(out, test_y)
        pred_y = torch.max(out, dim=1).indices
        loss = loss.item() / self.Ph
        g_acc = torch.sum(pred_y == test_y, dtype=torch.float)
        g_acc = g_acc / self.test_labels.size(0)
        acc_idv = []
        for i in range(self.Ph):
            model = self.models[i]
            with torch.no_grad():
                out = model(test_x)
            pred_y = torch.max(out, dim=1).indices
            acc = torch.sum(pred_y == test_y, dtype=torch.float)
            acc = acc / self.test_labels.size(0)
            acc_idv.append(acc)
        return g_acc, acc_idv, loss

    def apply_grad(self, idx, grad_mean=True):
        super().apply_grad(idx, grad_mean)
        self.global_model.apply_grad(self.sum_grad, self.verify_lrr)
    
    def eq_train(self, threshold=0.85, attack_strength=1, selective_apply_grad=False):
        super(FedAvg, self).eq_train(threshold, attack_strength, True)

