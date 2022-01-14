import itertools
import torch
import pandas as pd
import datetime


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
time_str = datetime.datetime.now().strftime("%Y_%m_%d_%H")


class CGD_model_fc(torch.nn.Module):
    def __init__(self, Pv:int, n_H:int, in_length=28*28):
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
            torch.nn.Linear(n_H, 10)
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
                 stride=10,
                 output_path="./output/"):
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
        self.output_path = output_path
        self.models = []
        self.loss = torch.nn.CrossEntropyLoss()
        self.sum_grad = None
        self.grad_length = 0

    def confined_init(self, random_lambda=False):
        for i in range(self.Ph):
            model = CGD_model_fc(self.Pv, self.n_H, in_length=self.train_imgs.size(1))
            bound = self.init_lambda
            if random_lambda:
                bound = self.init_lambda * torch.randn(1).item()
            param = torch.rand(model.get_flatten_parameters().size()) * bound
            model.load_parameters(param)
            self.models.append(model)
            if self.grad_length == 0:
                self.grad_length = model.get_flatten_parameters().size(0)

    def shuffle_data(self):
        shuffled_index = torch.randperm(self.train_imgs.size(0))
        self.train_imgs = self.train_imgs[shuffled_index]
        self.train_labels = self.train_labels[shuffled_index]

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

    def back_prop(self, X, Y):
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
            acc_i = 0
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

    def sparsify_update(self, gradients: list, p=None):
        if p is None:
            p = self.sampling_prob
        sampling_idx = torch.zeros(gradients[0].size(), dtype=torch.bool)
        sampling_idx.bernoulli_(1-p)
        for g in gradients:
            g[sampling_idx] = 0
        return gradients

    def evaluate(self):
        outs = torch.zeros(self.test_labels.size(0), 10)
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
        acc = (pred_y == test_y).sum()
        acc = acc/self.test_labels.size(0)
        return acc, loss

    def flatten(self, grads):
        out = None
        for g in grads:
            if out is None:
                out = g.flatten()
            else:
                out = torch.cat([out, g.flatten()])
        return out

    def eq_train(self):
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
            if epoch % self.stride == 0:
                test_acc, test_loss = self.evaluate()
                epoch_col.append(epoch)
                test_acc_col.append(test_acc.item())
                test_loss_col.append(test_loss)
                train_acc_col.append(acc.item())
                train_loss_col.append(loss)
                print(f'Epoch {epoch} - test acc {test_acc:6.4f}, test loss: {test_loss:6.4f}, train acc {acc:6.4f}, '
                      f'train loss {loss:6.4f}')
        recorder = pd.DataFrame({"epoch":epoch_col, "test_acc":test_acc_col, "test_loss":test_loss_col,
                                 "train_acc":train_acc_col, "train_loss":train_loss_col})
        recorder.to_csv(self.output_path+f"CGD_{self.dataset}_epoch_{self.num_iter}_Pv_{self.Pv}_Ph_{self.Ph}_nH_{self.n_H}_Sigma_"
                                       f"{self.sigma}_lambda_{self.init_lambda}_"+time_str+".csv")
