import numpy as np
import copy

from utils import *


class TuringNetwork:
    def __init__(self, n, net, u0, v0, tps=0.01, sigma=1.0, epsilon=0.1):
        assert net.shape[0] == net.shape[1] == n == len(u0) == len(v0)

        self.n = n
        self.net = net
        self.tps = tps
        self.sigma = sigma
        self.epsilon = epsilon
        self.u = u0.reshape(self.n)
        self.v = v0.reshape(self.n)
        print("u0 shape: {0}\tmax: {1:.4e}\tmin: {2:.4e}\tavg: {3:.4e}\tstd: {4:.4e}".format(self.u.shape, np.max(self.u), np.min(self.u), np.mean(self.u), np.std(self.u)))
        print("v0 shape: {0}\tmax: {1:.4e}\tmin: {2:.4e}\tavg: {3:.4e}\tstd: {4:.4e}".format(self.v.shape, np.max(self.v), np.min(self.v), np.mean(self.v), np.std(self.v)))
        print("Network shape: {}".format(self.net.shape))
        print("Network non-zeros: {} / {}".format(n * n - np.count_nonzero(network == 0), n * n))

    @staticmethod
    def f(u, v):
        # return 0.1 - 1 * u + 1 * (u ** 2) * v
        return 1 * (3 * u ** 2 / (v * (1 + 0.5 * u ** 2)) - u) + 0.5

    @staticmethod
    def g(u, v):
        # return 0.9 - 1 * (u ** 2) * v
        return 1 * (2 * u ** 2 - v) + 0.3

    def step(self):
        dui_dt = np.zeros(self.n)
        dvi_dt = np.zeros(self.n)
        for i in range(self.n):
            dui_dt[i] = self.f(self.u[i], self.v[i]) + self.epsilon * np.sum(self.net[i].reshape(self.n) * self.u)  # sum([self.net[i][j] * self.u[j] for j in range(self.n)])
            dvi_dt[i] = self.g(self.u[i], self.v[i]) + self.epsilon * self.sigma * np.sum(self.net[i].reshape(self.n) * self.v)  # self.sigma * sum([self.net[i][j] * self.v[j] for j in range(self.n)])
            # print(self.f(self.u[i], self.v[i]), np.sum(self.net[i].reshape(self.n) * self.u))
        self.u = self.u + dui_dt * self.tps
        self.v = self.v + dvi_dt * self.tps

    def run(self, epoch, epoch_step=100):
        for i in range(1, epoch + 1):
            u_old, v_old = copy.deepcopy(self.u), copy.deepcopy(self.v)
            self.step()
            if i % epoch_step == 0 or i == epoch:
                print("[{0:05d}/{1}]\tu_diff: {2:.4e}\tv_diff: {3:.4e}\tu_avg: {4:.4e}\tv_avg: {5:.4e}\tu_std: {6:.4e}\tv_std: {7:.4e}".format(
                    i,
                    epoch,
                    np.sum(np.abs(self.u - u_old)),
                    np.sum(np.abs(self.v - v_old)),
                    np.mean(self.u),
                    np.mean(self.v),
                    np.std(self.u),
                    np.std(self.u),
                ))
        print("u_last: {}".format(list(self.u)))
        print("v_last: {}".format(list(self.v)))


if __name__ == "__main__":
    network = np.load("data/average_network/network_laplacian_MCI.npy")
    u = np.load("data/average_node/A_MCI.npy")
    # u = u / np.max(u)
    v = np.load("data/average_node/T_MCI.npy")
    # v = v / np.max(v)
    # u = np.random.rand(160)
    # v = np.random.rand(160)
    tn = TuringNetwork(160, network, u, v, 0.01, 0.125, 0.02)

    tn.run(10000, 1000)
