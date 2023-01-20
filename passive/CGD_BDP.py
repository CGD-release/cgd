import numpy as np
import pandas as pd
import datetime
import torch
from numpy import linalg as LA
import itertools
import random


class CGD:

    def __init__(self,
                 num_iter,
                 train_imgs,
                 train_labels,
                 train_labels_one_hot,
                 test_imgs,
                 test_labels,
                 test_labels_one_hot,
                 Ph,
                 Pv,
                 n_H,
                 delta,
                 dataset,
                 sampling_prob,
                 max_grad_norm,
                 sigma,
                 batch_num):
        self.num_iter = num_iter
        self.train_imgs = train_imgs
        self.train_labels = train_labels
        self.train_labels_one_hot = train_labels_one_hot
        self.test_imgs = test_imgs
        self.test_labels = test_labels
        self.test_labels_one_hot = test_labels_one_hot
        self.Ph = Ph
        self.Pv = Pv
        self.n_H = n_H
        self.delta = delta
        self.dataset = dataset
        self.sampling_prob = sampling_prob
        self.max_grad_norm = max_grad_norm
        self.sigma = sigma
        self.batch_num = batch_num

    def relu(self, x):
        x[x < 0] = 0
        return x

    def h(self, X, W, b, Pv, Ph, n_H):
        '''
        Hypothesis function: simple FNN with 1 hidden layer
        Layer 1: input
        Layer 2: hidden layer, with a size implied by the arguments W[0], b
        Layer 3: output layer, with a size implied by the arguments W[1]
        '''
        # layer 1 = input layer
        K = 10
        N = X.shape[0]
        D = X.shape[1]
        X_lk = []
        z1_lk = []
        z1_l = []
        a2_l = []
        z2_l = []
        s_l = []
        total_l = []
        sigma_l = []

        ### Step 1:
        # layer 1 = input layer
        for i in range(Ph):
            sigma_z1_lk = np.zeros((int(N / Ph), n_H))
            for j in range(Pv):
                X_lk.append(X[i * int(N / Ph):(i + 1) * int(N / Ph), j * int(D / Pv):(j + 1) * int(D / Pv)])
                z1_lk.append(np.matmul(X_lk[i * Pv + j], W[i * Pv + j]))
                sigma_z1_lk += z1_lk[i * Pv + j]  # \sigma_{k=1}^Pv X^{lk}W_1^{lk}
            # layer 1 (input layer) -> layer 2 (hidden layer)
            z1_l.append(sigma_z1_lk + b[0])  # z1_l[i] \in R^{m^l*n_H}
            # layer 2 activation
            a2_l.append(self.relu(z1_l[i]))
            # layer 2 (hidden layer) -> layer 3 (output layer)
            z2_l.append(np.matmul(a2_l[i], W[Ph * Pv + i]))  # z2_l = a2_l*W2_l
            s_l.append(np.exp(z2_l[i]))
            total_l.append(np.sum(s_l[i], axis=1).reshape(-1, 1))
            sigma_l.append(s_l[i] / total_l[i])

        sigma = np.concatenate(sigma_l, axis=0)

        # the output is a probability for each sample
        return sigma

    def loss(self, y_pred, y_true):
        '''
        Loss function: cross entropy with an L^2 regularization
        y_true: ground truth, of shape (N, )
        y_pred: prediction made by the model, of shape (N, K)
        N: number of samples in the batch
        K: global variable, number of classes
        '''
        global K
        K = 10
        N = len(y_true)
        # loss_sample stores the cross entropy for each sample in X
        # convert y_true from labels to one-hot-vector encoding
        # y_true_one_hot_vec = (y_true[:,np.newaxis] == np.arange(K))
        y_true_one_hot_vec = y_true
        loss_sample = (np.log(y_pred) * y_true_one_hot_vec).sum(axis=1)
        # loss_sample is a dimension (N,) array
        # for the final loss, we need take the average
        return -np.mean(loss_sample)

    def backprop(self, W, b, X, y, Pv, Ph, n_H, accountant=None, alpha=1e-4):
        '''
        Step 1: explicit forward pass h(X;W,b, Pv, Ph,n_H)
        Step 2: backpropagation for dW and db
        '''
        K = 10
        N = X.shape[0]
        D = X.shape[1]
        X_lk = []
        z1_lk = []
        z1_l = []
        a2_l = []
        z2_l = []
        s_l = []
        total_l = []
        sigma_l = []

        ### Step 1:
        # layer 1 = input layer
        for i in range(Ph):
            sigma_z1_lk = np.zeros((int(N / Ph), n_H))
            for j in range(Pv):
                X_lk.append(X[i * int(N / Ph):(i + 1) * int(N / Ph), j * int(D / Pv):(j + 1) * int(D / Pv)])
                z1_lk.append(np.matmul(X_lk[i * Pv + j], W[i * Pv + j]))
                sigma_z1_lk += z1_lk[i * Pv + j]  # \sigma_{k=1}^Pv X^{lk}W_1^{lk}
            # layer 1 (input layer) -> layer 2 (hidden layer)
            z1_l.append(sigma_z1_lk + b[0])  # z1_l[i] \in R^{m^l*n_H}
            # layer 2 activation
            a2_l.append(self.relu(z1_l[i]))
            # layer 2 (hidden layer) -> layer 3 (output layer)
            z2_l.append(np.matmul(a2_l[i], W[Ph * Pv + i]))  # z2_l = a2_l*W2_l
            s_l.append(np.exp(z2_l[i]))
            total_l.append(np.sum(s_l[i], axis=1).reshape(-1, 1))
            sigma_l.append(s_l[i] / total_l[i])

        ### Step 2:

        # layer 2->layer 3 weights' derivative
        # delta2 is \partial L/partial z2, of shape (N,K)
        # y_one_hot_vec = (y[:, np.newaxis] == np.arange(K))

        y_l = []
        delta2_l = []
        grad_W2_l = []
        delta1_l = []
        grad_W1_k = []
        dW = []
        if accountant:
            for i in range(Ph):
                y_l.append(y[i * int(N / Ph):(i + 1) * int(N / Ph), ])
                delta2_l.append(sigma_l[i] - y_l[i])  # delta2_l \in R^{m^l*10}
                grad_W2_l_tilde = np.matmul(a2_l[i].T, delta2_l[i])
                #grad_W2_l_tilde /= max(1.0, LA.norm(grad_W2_l_tilde) / self.max_grad_norm)
                grad_W2_l_tilde += np.random.randn(grad_W2_l_tilde.shape[0], grad_W2_l_tilde.shape[1]) * (self.sigma * LA.norm(grad_W2_l_tilde))
                grad_W2_l.append(grad_W2_l_tilde)  # a2_l \in R^{m^l*n_H}, grad_W2_l \in R{n_H*10}
                # layer 1->layer 2 weights' derivative
                # delta1 is \partial a2/partial z1
                # layer 2 activation's (weak) derivative is 1*(z1>0)
                delta1_l.append(np.matmul(delta2_l[i], W[Pv * Ph + i].T) * (
                        z1_l[i] > 0))  # delta1_l: R^{m^l*n_H}, delta2_l: R^{m^l*10}, W2_l: R{n_H*10}
            grad_W2 = sum(grad_W2_l)  # global g_W2: need further investigation, sum? or other operations?

            # delta1 = np.concatenate(delta1_l, axis=0)

            for i in range(Pv):
                sigma_g_W1_lk = np.zeros((int(D / Pv), n_H))
                grad_W1_lk = []
                for j in range(Ph):
                    grad_W1_lk.append(np.matmul(X_lk[j * Pv + i].T, delta1_l[j]))  # grad_W1_lk: R^{(64/Pv)*n_H}
                    sigma_g_W1_lk += grad_W1_lk[j]
                sigma_g_W1_lk_tilde = sigma_g_W1_lk
                #sigma_g_W1_lk_tilde /= max(1.0, LA.norm(sigma_g_W1_lk_tilde) / self.max_grad_norm)
                sigma_g_W1_lk_tilde += np.random.randn(sigma_g_W1_lk_tilde.shape[0], sigma_g_W1_lk_tilde.shape[1]) * (self.sigma * LA.norm(sigma_g_W1_lk_tilde))
                grad_W1_k.append(sigma_g_W1_lk_tilde)

        else:
            for i in range(Ph):
                y_l.append(y[i * int(N / Ph):(i + 1) * int(N / Ph), ])
                delta2_l.append(sigma_l[i] - y_l[i])  # delta2_l \in R^{m^l*10}
                grad_W2_l.append(np.matmul(a2_l[i].T, delta2_l[i]))  # a2_l \in R^{m^l*n_H}, grad_W2_l \in R{n_H*10}
                # layer 1->layer 2 weights' derivative
                # delta1 is \partial a2/partial z1
                # layer 2 activation's (weak) derivative is 1*(z1>0)
                delta1_l.append(np.matmul(delta2_l[i], W[Pv * Ph + i].T) * (
                        z1_l[i] > 0))  # delta1_l: R^{m^l*n_H}, delta2_l: R^{m^l*10}, W2_l: R{n_H*10}
            grad_W2 = sum(grad_W2_l)  # global g_W2: need further investigation, sum? or other operations?

            # delta1 = np.concatenate(delta1_l, axis=0)

            for i in range(Pv):
                sigma_g_W1_lk = np.zeros((int(D / Pv), n_H))
                grad_W1_lk = []
                for j in range(Ph):
                    grad_W1_lk.append(np.matmul(X_lk[j * Pv + i].T, delta1_l[j]))  # grad_W1_lk: R^{(64/Pv)*n_H}
                    sigma_g_W1_lk += grad_W1_lk[j]
                grad_W1_k.append(sigma_g_W1_lk)

        for i in range(Ph):
            for j in range(Pv):
                # the alpha part is the derivative for the regularization
                # regularization = 0.5*alpha*(np.sum(W[1]**2) + np.sum(W[0]**2))
                dW.append(grad_W1_k[j] / N + alpha * W[i * Pv + j])

        for i in range(Ph):
            dW.append(grad_W2 / N + alpha * W[Pv * Ph + i])  # global
            # dW.append(grad_W2_l[i] / N + alpha * W[Pv+i]) #local

        delta1 = np.concatenate(delta1_l, axis=0)
        db = [np.mean(delta1, axis=0)]
        return dW, db

    def sparsify_update(self, dW, p, estimate=True):
        init = True
        if estimate:
            for i in range(len(dW)):
                param = torch.from_numpy(dW[i])
                if init:
                    idx = torch.zeros_like(param, dtype=torch.bool)
                    idx.bernoulli_(1 - p)
                init = False
                dW[i][idx.numpy()] = 0
        else:
            for i in range(len(dW)):
                if i == self.Pv * self.Ph:
                    init = True
                param = torch.from_numpy(dW[i])
                if init:
                    idx = torch.zeros_like(param, dtype=torch.bool)
                    idx.bernoulli_(1 - p)
                init = False
                dW[i][idx.numpy()] = 0

        return dW

    def sigma_gaussian(self):
        init = 0
        if self.delta == "random":
            for i in range(self.Pv * self.Ph):
                init += random.uniform(0.001, 0.1)
            init /= self.Pv * self.Ph
        else:
            init = self.delta
        return np.sqrt(self.Ph*(init+np.square(self.sigma)))

    def privacy_accounting(self, accountant, W, b, alpha):
        grads_est = []
        num_subbatch = 8
        sample = np.random.randint(0, self.train_imgs.shape[0], num_subbatch)
        for j in range(num_subbatch):
            grad_sample, db_sample = self.backprop(W, b,
                                                   np.delete(self.train_imgs, sample[j], 0),
                                                   np.delete(self.train_labels_one_hot, sample[j], 0),
                                                   self.Pv, self.Ph, self.n_H, accountant,
                                                   alpha)
            grad_sample = np.append(grad_sample[0], grad_sample[self.Ph * self.Pv])
            grad_sample /= max(1.0, LA.norm(grad_sample) / self.max_grad_norm)
            grads_est.append(grad_sample)
        grads_est = self.sparsify_update(grads_est, p=self.sampling_prob)

        # q = batch_size / (self.train_imgs.shape[0] * self.Ph)
        q = 1 / self.batch_num

        # NOTE:
        # Using combinations within a set of gradients (like below)
        #  does not actually produce samples from the correct distribution
        #  (for that, we need to sample pairs of gradients independently).
        #  However, the difference is not significant, and it speeds up computations.
        grads_est = torch.tensor(grads_est)
        pairs = list(zip(*itertools.combinations(grads_est, 2)))
        sigma_g = self.sigma_gaussian()
        accountant.accumulate(
            ldistr=(torch.stack(pairs[0]), sigma_g * self.max_grad_norm),
            rdistr=(torch.stack(pairs[1]), sigma_g * self.max_grad_norm),
            q=q,
            steps=self.batch_num * 20,
        )
        running_eps = accountant.get_privacy(target_delta=1e-5)
        return running_eps

    def train(self, accountant=None):
        eta = 5e-1
        alpha = 1e-6  # regularization
        gamma = 0.9  # RMSprop
        eps = 1e-3  # RMSprop
        n = self.train_imgs.shape[1]  # number of pixels in an image
        K = 10
        loss_col = []  # loss on train
        acc_col = []  # acc on train
        loss_val = []  # loss on test
        acc_val = []  # acc on test
        iter = []
        eps_col = []
        batch_size = int(self.train_imgs.shape[0]/(self.Ph * self.batch_num))  # trainig batch size per participant

        # initialization
        np.random.seed(1127)
        W = []
        # W1 = 1e-1*np.random.randn(int(n/Pv), n_H)
        for i in range(self.Pv * self.Ph):
            # W.append(W1)
            if self.delta == "random":
                init = random.uniform(0.001, 0.1)
                W.append(init * np.random.randn(int(n / self.Pv), self.n_H))
            else:
                W.append(self.delta * np.random.randn(int(n / self.Pv), self.n_H))

        for i in range(self.Ph):
            if self.delta == "random":
                init = random.uniform(0.001, 0.1)
                W.append(init * np.random.randn(self.n_H, K))
            else:
                W.append(self.delta * np.random.randn(self.n_H, K))
        b = [np.random.randn(self.n_H)]
        gW = []
        for i in range(self.Pv * self.Ph + self.Ph):
            gW.append(1)
        gb0 = 1
        etaW = []

        for i in range(self.num_iter):
            for batch in range(int(self.train_imgs.shape[0] / (batch_size * self.Ph))):
                inputs = self.train_imgs[batch*(batch_size * self.Ph):(batch+1)*(batch_size * self.Ph),:]
                label = self.train_labels_one_hot[batch*(batch_size * self.Ph):(batch+1)*(batch_size * self.Ph),:]
                dW, db = self.backprop(W, b, inputs,label,self.Pv, self.Ph, self.n_H, accountant, alpha)

                #if accountant:
                #    for j in range(self.Pv * self.Ph + self.Ph):
                #        dW[j] /= max(1.0, LA.norm(dW[j]) / self.max_grad_norm)
                #        dW[j] += np.random.randn(dW[j].shape[0], dW[j].shape[1]) * (self.sigma * self.max_grad_norm)
                #    dW = self.sparsify_update(dW, p=self.sampling_prob,estimate=False)

                # update
                for j in range(self.Pv * self.Ph + self.Ph):
                    gW[j] = gamma * gW[j] + (1 - gamma) * np.sum(dW[j] ** 2)
                    etaW.append(eta / np.sqrt(gW[j] + eps))
                    W[j] = W[j] - etaW[j] * dW[j]

                gb0 = gamma * gb0 + (1 - gamma) * np.sum(db[0] ** 2)
                etab0 = eta / np.sqrt(gb0 + eps)
                b[0] -= etab0 * db[0]



            if i % 20 == 0:  #

                running_eps = self.privacy_accounting(accountant, W, b, alpha) if (accountant and i>1) else None

                print("Step: %d/%d.  Privacy (𝜀,𝛿): %s" %
                          (i*self.batch_num + 1, self.num_iter*self.batch_num, running_eps))
                eps_col.append(running_eps)

                # sanity check 1
                y_pred = self.h(self.train_imgs, W, b, self.Pv, self.Ph, self.n_H)
                los = self.loss(y_pred, self.train_labels_one_hot)
                acc = np.mean(np.argmax(y_pred, axis=1) == np.reshape(self.train_labels, self.train_labels.shape[0]))
                print("Cross-entropy loss after", i + 1, "iterations is {:.8}".format(
                    los))
                print("Training accuracy after", i + 1, "iterations is {:.4%}".format(
                    acc))
                # print("W1_11={:.4f} W1_12={:.4f} W1_21={:.4f} W1_22={:.4f} W2_1={:.4f} W2_2={:.4f}"
                #      .format(W[0].sum(), W[1].sum(), W[2].sum(), W[3].sum(), W[4].sum(), W[5].sum()))

                loss_col.append(los)
                acc_col.append(acc)

                y_pred_test = self.h(self.test_imgs, W, b, self.Pv, self.Ph, self.n_H)
                print("Testing cross-entropy loss is {:.8}".format(self.loss(y_pred_test, self.test_labels_one_hot)))
                print("Testing accuracy is {:.4%}".format(
                    np.mean(np.argmax(y_pred_test, axis=1) == np.reshape(self.test_labels, self.test_labels.shape[0]))))
                loss_val.append(self.loss(y_pred_test, self.test_labels_one_hot))
                acc_val.append(
                    np.mean(np.argmax(y_pred_test, axis=1) == np.reshape(self.test_labels, self.test_labels.shape[0])))

                iter.append(i + 1)

                # reset RMSprop
                for t in range(self.Pv * self.Ph + self.Ph):
                    gW[t] = 1
                gb0 = 1

        y_pred_final = self.h(self.train_imgs, W, b, self.Pv, self.Ph, self.n_H)
        print("Final cross-entropy loss is {:.8}".format(self.loss(y_pred_final, self.train_labels_one_hot)))
        print("Final training accuracy is {:.4%}".format(
            np.mean(np.argmax(y_pred_final, axis=1) == np.reshape(self.train_labels, self.train_labels.shape[0]))))
        loss_col.append(self.loss(y_pred_final, self.train_labels_one_hot))
        acc_col.append(
            np.mean(np.argmax(y_pred_final, axis=1) == np.reshape(self.train_labels, self.train_labels.shape[0])))
        iter.append(self.num_iter)  # train final

        y_pred_test = self.h(self.test_imgs, W, b, self.Pv, self.Ph, self.n_H)
        print("Final testing cross-entropy loss is {:.8}".format(self.loss(y_pred_test, self.test_labels_one_hot)))
        print("Final testing accuracy is {:.4%}".format(
            np.mean(np.argmax(y_pred_test, axis=1) == np.reshape(self.test_labels, self.test_labels.shape[0]))))
        loss_val.append(self.loss(y_pred_test, self.test_labels_one_hot))
        acc_val.append(
            np.mean(np.argmax(y_pred_test, axis=1) == np.reshape(self.test_labels, self.test_labels.shape[0])))

        final_eps = self.privacy_accounting(accountant, W, b, alpha) if accountant else None
        print("Final  Privacy ", final_eps)
        eps_col.append(final_eps)

        df = pd.DataFrame(
            {'iter': iter, 'loss_train': loss_col, 'acc_train': acc_col, 'loss_test': loss_val, 'acc_test': acc_val,
             'eps': eps_col})
        dt = datetime.datetime.now()

        df.to_csv(
            str(self.dataset) + "_Sigma=" + str(self.sigma) + "_Init=" + str(self.delta) +
            "_Ph=" + str(self.Ph) + "_Pv=" + str(
                self.Pv) + "_T=" + str(self.num_iter) + "_nH=" + str(
                self.n_H) + "_" + str(dt) + ".csv", index=False)

        return (y_pred_final, y_pred_test)
