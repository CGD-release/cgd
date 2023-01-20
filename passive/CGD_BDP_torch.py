import itertools
import torch
import pandas as pd
from constants import *


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class CGD_model_cnn(torch.nn.Module):
    def __init__(self, Pv, out_channel, reso=28):
        super().__init__()
        self.Pv = Pv
        self.reso = reso
        self.out_channel = out_channel
        self.input = torch.nn.Sequential(
            torch.nn.Conv2d(Pv ** 2, out_channel, 3, 1, 1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2)
        )
        self.hidden = torch.nn.Sequential(
            torch.nn.Linear((reso//(Pv*2))**2 * out_channel, 10)
        )
        self.optimizer = torch.optim.Adam(self.parameters())

    def forward(self, x):
        x = x.view(x.size(0), self.Pv ** 2, self.reso // self.Pv, -1)
        h_input = self.input(x)
        h_input = h_input.view(x.size(0), -1)
        out = self.hidden(h_input)
        return out

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


class CGD_model_fc(torch.nn.Module):
    def __init__(self, Pv:int, n_H:int, in_length=28*28, out_length=10):
        super().__init__()
        self.Pv = Pv
        self.n_H = n_H
        self.v_parts = []
        for i in range(Pv):
            self.v_parts.append(torch.nn.Sequential(
                torch.nn.Linear(in_length // Pv, n_H),
                torch.nn.ReLU()
            ))
        self.h_parts = torch.nn.Sequential(
            torch.nn.Linear(n_H, out_length)
        )
        p_chain = []
        for v in self.v_parts:
            for p in v.parameters():
                p_chain.append(p)
        for p in self.h_parts.parameters():
            p_chain.append(p)
        self.p_collection = p_chain
        self.optimizer = torch.optim.Adam(iter(p_chain))

    def parameters(self, recurse: bool = True):
        return iter(self.p_collection)
    
    def forward(self, x):
        x = x.view(x.size(0), self.Pv, -1)
        outs = [self.v_parts[i](x[:, i]) for i in range(self.Pv)]
        h_in = outs[0]
        for i in range(1, self.Pv):
            h_in += outs[i]
        return self.h_parts(h_in)
    
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
                

class CGD_torch:
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
                 sampling_prob,
                 max_grad_norm,
                 sigma,
                 sub_batch=8,
                 stride=10):
        self.num_iter = num_iter
        self.train_imgs = train_imgs
        self.train_labels = train_labels
        self.test_imgs = test_imgs
        self.test_labels = test_labels
        self.Ph = Ph
        self.Pv = Pv
        self.n_H = n_H
        self.init_lambda = init_lambda
        self.batch_size = train_imgs.size(0) // batch
        self.batch = batch
        self.dataset = dataset
        self.sampling_prob = sampling_prob
        self.max_grad_norm = max_grad_norm
        self.sigma = sigma
        self.stride = stride
        self.sub_batch = sub_batch
        self.models = []
        self.glb_model = None
        self.loss = torch.nn.CrossEntropyLoss()
        self.sum_grad = None
        self.grad_length = 0
        self.label_classes = 10

    def confined_init(self, random_lambda=False):
        self.label_classes = self.train_labels.unique().size(0)
        for i in range(self.Ph):
            model = CGD_model_fc(self.Pv, self.n_H, in_length=self.train_imgs.size(1), out_length=self.label_classes)
#           model = CGD_model_cnn(self.Pv, self.n_H)
            bound = self.init_lambda
            if random_lambda:
                bound = self.init_lambda * torch.randn(1).item()
            param = torch.rand(model.get_flatten_parameters().size()) * bound
            model.load_parameters(param)
            self.models.append(model)
            if self.grad_length == 0:
                self.grad_length = model.get_flatten_parameters().size(0)

    def cnn_init(self):
        for i in range(self.Ph):
            model = CGD_model_cnn(self.Pv, self.n_H)
            bound = self.init_lambda
            param = torch.rand(model.get_flatten_parameters().size()) * bound
            model.load_parameters(param)
            self.models.append(model)
            if self.grad_length == 0:
                self.grad_length = model.get_flatten_parameters().size(0)

    def avg_init(self):
        self.glb_model = CGD_model_fc(self.Pv, self.n_H, in_length=self.train_imgs.size(1))
        glb_param = self.glb_model.get_flatten_parameters()
        self.grad_length = self.glb_model.get_flatten_parameters().size(0)
        for i in range(self.Ph):
            model = CGD_model_fc(self.Pv, self.n_H, in_length=self.train_imgs.size(1))
            model.load_parameters(glb_param)
            self.models.append(model)

    def local_init(self):
        for i in range(self.Ph):
            model = CGD_model_fc(self.Pv, self.n_H, in_length=self.train_imgs.size(1))
            self.models.append(model)

    def shuffle_data(self):
        shuffled_index = torch.randperm(self.train_imgs.size(0))
        self.train_imgs = self.train_imgs[shuffled_index]
        self.train_labels = self.train_labels[shuffled_index]

    def shuffle_non_iid(self):
        batch_size = self.batch_size // self.Ph
        labels = self.train_labels.unique()
        img_org = torch.zeros_like(self.train_imgs)
        label_org = torch.zeros_like(self.train_labels)
        img_pool = {}
        cur_idx = {}
        for l in labels:
            idx = self.train_labels.flatten() == l
            img_pool[l.item()] = self.train_imgs[idx]
            cur_idx[l.item()] = 0
        glb_idx = 0
        cur_label = 0
        while glb_idx < self.train_imgs.size(0):
            cur_label = (cur_label + 1) % 10
            start_idx = glb_idx
            glb_idx += batch_size
            label_idx = cur_idx[cur_label]
            label_start = label_idx
            label_idx += batch_size
            if label_idx < img_pool[cur_label].size(0):
                block = img_pool[cur_label][label_start: label_idx]
                cur_idx[cur_label] = label_idx
                img_org[start_idx: glb_idx] = block
                label_org[start_idx: glb_idx] = cur_label
            else:
                break
        glb_idx -= batch_size
        res = None
        for l in img_pool:
            res_len = img_pool[l][cur_idx[l]:].size(0)
            label_org[glb_idx: glb_idx + res_len] = l
            glb_idx += res_len
            if res is None:
                res = img_pool[l][cur_idx[l]:]
            else:
                res = torch.cat([res, img_pool[l][cur_idx[l]:]], dim=0)
        img_org[-res.size(0):] = res
        self.train_imgs = img_org
        self.train_labels = label_org

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

    def back_prop(self, X, Y, step=False):
        self.grad_reset()
        sum_acc = 0
        sum_loss = 0
        sample_per_participant = X.size(0) // self.Ph
        for i in range(self.Ph):
            model = self.models[i]
            lower = sample_per_participant * i
            upper = sample_per_participant * (i + 1)
            x_i = X[lower: upper]
            y_i = Y[lower: upper]
            y_i = y_i.flatten()
            acc_i = 0.0
            loss_i = 0
            out = model(x_i)
            y_pred = torch.max(out, dim=1).indices
            j_loss = self.loss(out, y_i)
            model.zero_grad()
            j_loss.backward()
            self.collect_grad(model)
            acc_i += (y_pred == y_i).sum()
            loss_i += j_loss.item()
            acc_i = acc_i / y_i.size(0)
            # loss_i = loss_i / y_i.size(0)
            sum_acc += acc_i
            sum_loss += loss_i
            if step:
                model.step()
        sum_acc = sum_acc / self.Ph
        sum_loss = sum_loss / self.Ph
        return sum_acc, sum_loss

    def collect_grad(self, model, norm_clip=True, add_noise=True):
        grad_iter = iter(self.sum_grad)
        for param in model.parameters():
            grad = param.grad
            if add_noise:
                grad += torch.randn(grad.size()) * self.sigma
            if norm_clip and grad.norm() > self.max_grad_norm:
                grad = grad * self.max_grad_norm / grad.norm()
            collector = next(grad_iter, None)
            collector += grad

    def apply_grad(self, grad_mean=True, add_noise=False):
        for grad in self.sum_grad:
            if grad_mean:
                grad /= self.Ph
            if add_noise:
                noise = torch.randn(grad.size()) * self.sigma
#                 max_norm = torch.sqrt(torch.tensor(grad.flatten().size(0) / self.grad_length)) * self.max_grad_norm
                max_norm = self.max_grad_norm
                if noise.norm() > max_norm:
                    noise = noise * max_norm / noise.norm()
                grad += noise
        for i in range(self.Ph):
            model = self.models[i]
            grad_iter = iter(self.sum_grad)
            for param in model.parameters():
                grad = next(grad_iter)
                param.grad.copy_(grad)
            model.step()

    def apply_avg(self):
        for grad in self.sum_grad:
            grad /= self.Ph
        model = self.glb_model
        grad_iter = iter(self.sum_grad)
        for param in model.parameters():
            grad = next(grad_iter)
            param.grad = grad
        model.step()
        glb_param = model.get_flatten_parameters()
        for i in range(self.Ph):
            model = self.models[i]
            model.load_parameters(glb_param)

    def sparsify_update(self, gradients: list, p=None):
        if p is None:
            p = self.sampling_prob
        for g in gradients:
            sampling_idx = torch.zeros(g.size(), dtype=torch.bool)
            sampling_idx.bernoulli_(1 - p)
            g[sampling_idx] = 0
        return gradients

    def evaluate(self):
        outs = torch.zeros(self.test_labels.size(0), self.label_classes)
        loss = 0
        test_x = self.test_imgs
        test_y = self.test_labels.flatten()
        for i in range(self.Ph):
            model = self.models[i]
            with torch.no_grad():
                out = model(test_x)
            outs += out
            loss_i = self.loss(out, test_y)
            loss += loss_i.item()
        pred_y = torch.max(outs, dim=1).indices
        loss = loss / self.Ph
        acc = torch.sum(pred_y == test_y, dtype=torch.float)
        acc = acc/self.test_labels.size(0)
        return acc, loss

    def eval_worst(self):
        test_x = self.test_imgs
        test_y = self.test_labels.flatten()
        worst_acc = 1
        avg_acc = 0
        for i in range(self.Ph):
            model = self.models[i]
            with torch.no_grad():
                out = model(test_x)
            pred_y = torch.max(out, dim=1).indices
            acc = (pred_y == test_y).sum().item()
            acc = acc / self.test_labels.size(0)
            avg_acc += acc
            if acc < worst_acc:
                worst_acc = acc
        avg_acc /= self.Ph
        return worst_acc, avg_acc

    def accumulate(self, accountant, sample_grad):
        pairs = list(zip(*itertools.combinations(sample_grad, 2)))
        sigma = torch.sqrt(self.Ph*(self.init_lambda+torch.square(torch.tensor(self.sigma)))).item()
        accountant.accumulate(
            ldistr=(torch.stack(pairs[0]), sigma * self.max_grad_norm),
            rdistr=(torch.stack(pairs[1]), sigma * self.max_grad_norm),
            q=1/self.batch,
            steps=self.stride * self.batch,
        )

    def flatten(self, grads):
        out = None
        for g in grads:
            if out is None:
                out = g.flatten()
            else:
                out = torch.cat([out, g.flatten()])
        return out

    def count_privacy(self, accountant, num_subbatch=8):
        drop_samples = torch.randint(0, self.train_imgs.size(0), (num_subbatch, ))
        grad_samples = []
        for i in range(num_subbatch):
            drop_idx = torch.ones(self.train_imgs.size(0), dtype=torch.bool)
            drop_idx[drop_samples[i]] = 0
            dropped_x = self.train_imgs[drop_idx]
            dropped_y = self.train_labels[drop_idx]
            self.back_prop(dropped_x, dropped_y)
            grad_est = self.flatten(self.sum_grad)
            grad_est /= self.Ph
            grad_samples.append(grad_est)
        grad_samples = self.sparsify_update(grad_samples)
        self.accumulate(accountant=accountant, sample_grad=grad_samples)

    def eq_train(self, accountant=None):
        epoch_col = []
        train_acc_col = []
        train_loss_col = []
        test_acc_col = []
        test_loss_col = []
        eps_col = []
        for epoch in range(self.num_iter):
            batch_idx = 0
            acc = 0
            loss = 0
            while batch_idx * self.batch_size < self.train_imgs.size(0):
                lower = batch_idx * self.batch_size
                upper = (batch_idx+1) * self.batch_size
                batch_X = self.train_imgs[lower: upper]
                batch_Y = self.train_labels[lower: upper]
                batch_idx += 1
                batch_acc, batch_loss = self.back_prop(batch_X, batch_Y)
                acc += batch_acc
                loss += batch_loss
                self.apply_grad()
            acc /= batch_idx
            loss /= batch_idx
#             print(f'Epoch {epoch} - train acc: {acc:6.4f}, train loss: {loss:6.4f}')
            if epoch % self.stride == 0:
                test_acc, test_loss = self.evaluate()
                epoch_col.append(epoch)
                test_acc_col.append(test_acc.item())
                test_loss_col.append(test_loss)
                train_acc_col.append(acc.item())
                train_loss_col.append(loss)
                running_eps = (0, 0)
                if accountant:
                    if epoch > 1:
                        self.count_privacy(accountant)
                        running_eps = accountant.get_privacy(target_delta=1e-5)
                    print(f'Epoch {epoch} - test acc {test_acc:6.4f}, test loss: {test_loss:6.4f}, Privacy (𝜀,𝛿): '
                          f'{running_eps}')
                    eps_col.append(running_eps[0])
                else:
                    print(f'Epoch {epoch} - test acc {test_acc:6.4f}, test loss: {test_loss:6.4f}')
                    eps_col.append(0)
        recorder = pd.DataFrame({"epoch":epoch_col, "test_acc":test_acc_col, "test_loss":test_loss_col, "train_acc":train_acc_col,
                                 "train_loss":train_loss_col, "epsilon":eps_col})
        recorder.to_csv(RECORDING_PATH+f"CGD_BDP_{self.dataset}_epoch_{self.num_iter}_Pv_{self.Pv}_Ph_{self.Ph}_nH_{self.n_H}_Sigma_"
                                       f"{self.sigma}_lambda_{self.init_lambda}_"+time_str+".csv")

    def fed_avg(self):
        epoch_col = []
        train_acc_col = []
        train_loss_col = []
        test_acc_col = []
        test_loss_col = []
        for epoch in range(self.num_iter):
            batch_idx = 0
            acc = 0
            loss = 0
            while batch_idx * self.batch_size < self.train_imgs.size(0):
                lower = batch_idx * self.batch_size
                upper = (batch_idx + 1) * self.batch_size
                batch_X = self.train_imgs[lower: upper]
                batch_Y = self.train_labels[lower: upper]
                batch_idx += 1
                batch_acc, batch_loss = self.back_prop(batch_X, batch_Y)
                acc += batch_acc
                loss += batch_loss
                self.apply_avg()
            acc /= batch_idx
            loss /= batch_idx
            if epoch % self.stride == 0:
                test_acc, test_loss = self.evaluate()
                print(f'Epoch {epoch} - train acc: {acc:6.4f}, test acc: {test_acc:6.4f}')
                epoch_col.append(epoch)
                test_acc_col.append(test_acc.item())
                test_loss_col.append(test_loss)
                train_acc_col.append(acc.item())
                train_loss_col.append(loss)
        recorder = pd.DataFrame(
            {"epoch": epoch_col, "test_acc": test_acc_col, "test_loss": test_loss_col, "train_acc": train_acc_col,
             "train_loss": train_loss_col})
        recorder.to_csv(
            RECORDING_PATH + f"FedAvg_{self.dataset}_epoch_{self.num_iter}_Pv_{self.Pv}_Ph_{self.Ph}_nH_{self.n_H}_Sigma_"
                             f"{self.sigma}_lambda_{self.init_lambda}_" + time_str + ".csv")

    def local_train(self):
        epoch_col = []
        worst_col = []
        avg_col = []
        for epoch in range(self.num_iter):
            batch_idx = 0
            acc = 0
            loss = 0
            while batch_idx * self.batch_size < self.train_imgs.size(0):
                lower = batch_idx * self.batch_size
                upper = (batch_idx + 1) * self.batch_size
                batch_X = self.train_imgs[lower: upper]
                batch_Y = self.train_labels[lower: upper]
                batch_idx += 1
                batch_acc, batch_loss = self.back_prop(batch_X, batch_Y, step=True)
                acc += batch_acc
                loss += batch_loss
            acc /= batch_idx
            loss /= batch_idx
            if epoch % self.stride == 0:
                worst_acc, avg_acc = self.eval_worst()
                print(f'Epoch {epoch} - avg acc: {avg_acc:6.4f}, worst acc: {worst_acc:6.4f}')
                epoch_col.append(epoch)
                worst_col.append(worst_acc)
                avg_col.append(avg_acc)
        recorder = pd.DataFrame(
            {"epoch": epoch_col, "worst_acc": worst_col, "avg_acc": avg_col})
        recorder.to_csv(
            RECORDING_PATH + f"Local_{self.dataset}_epoch_{self.num_iter}_Pv_{self.Pv}_Ph_{self.Ph}_nH_{self.n_H}_Sigma_"
                             f"{self.sigma}_lambda_{self.init_lambda}_" + time_str + ".csv")

    def cgd_worst(self):
        epoch_col = []
        worst_col = []
        avg_col = []
        for epoch in range(self.num_iter):
            batch_idx = 0
            acc = 0
            loss = 0
            while batch_idx * self.batch_size < self.train_imgs.size(0):
                lower = batch_idx * self.batch_size
                upper = (batch_idx + 1) * self.batch_size
                batch_X = self.train_imgs[lower: upper]
                batch_Y = self.train_labels[lower: upper]
                batch_idx += 1
                batch_acc, batch_loss = self.back_prop(batch_X, batch_Y)
                acc += batch_acc
                loss += batch_loss
                self.apply_grad()
            acc /= batch_idx
            loss /= batch_idx
            if epoch % self.stride == 0:
                worst_acc, avg_acc = self.eval_worst()
                epoch_col.append(epoch)
                worst_col.append(worst_acc)
                avg_col.append(avg_acc)
                print(f'Epoch {epoch} - avg acc: {avg_acc:6.4f}, worst acc: {worst_acc:6.4f}')
        recorder = pd.DataFrame(
            {"epoch": epoch_col, "worst_acc": worst_col, "avg_acc": avg_col})
        recorder.to_csv(
            RECORDING_PATH + f"CGD_worst_{self.dataset}_epoch_{self.num_iter}_Pv_{self.Pv}_Ph_{self.Ph}_nH_{self.n_H}_Sigma_"
                             f"{self.sigma}_lambda_{self.init_lambda}_" + time_str + ".csv")
