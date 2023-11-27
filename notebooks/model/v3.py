import time
import argparse

import jax
import matplotlib.pyplot as plt
import optax
import matfree
import tree_math as tm
from flax import linen as nn
from jax import nn as jnn
from jax import numpy as jnp
from jax import random, jit
import pickle
import torch
from pathlib import Path
import torch
import torch.nn.functional as F
import torchvision as tv
import numpy as np
import urllib.request
import tarfile
import cv2
import numpy as np
from torchvision.transforms.functional import normalize


def select_num_samples(dataset, n_samples, cls_to_idx):
    idxs = []
    for key,_ in cls_to_idx.items():
        indices = np.where(dataset.targets == key)[0]
        idxs.append(np.random.choice(indices, n_samples, replace=False))
    idxs = np.concatenate(idxs)
    dataset.data = dataset.data[idxs]
    dataset.targets = dataset.targets[idxs]
    return dataset


def select_classes(dataset, classes):
    idxs = []
    for i in classes:
        indices = np.where(dataset.targets == i)[0]
        idxs.append(indices)
    idxs = np.concatenate(idxs).astype(int)
    dataset.data = dataset.data[idxs]
    dataset.targets = dataset.targets[idxs]
    return dataset


def numpy_collate_fn(batch):
    data, target = zip(*batch)
    data = np.stack(data)
    target = np.stack(target)
    return {"image": data, "label": target}


def channel_normalization(tensor, mean, std):
    tensor = torch.from_numpy(tensor).float().transpose(1, 3)
    tensor = normalize(tensor, mean, std)
    return tensor


class MNIST(torch.utils.data.Dataset):
    def __init__(
        self,
        path_root="/data/",
        train: bool = True,
        transform = None,
        n_samples: int = None,
        cls: list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        download=True,
    ):
        self.path = Path(path_root)
        if train:
            self.dataset = tv.datasets.MNIST(root=self.path, train=True, download=download)
        else:
            self.dataset = tv.datasets.MNIST(root=self.path, train=False, download=download)
        self.transfrm = transform

        clas_to_index = { c : i for i, c in enumerate(cls)}
        if len(cls)<10:
                self.dataset = select_classes(self.dataset, cls)
        if n_samples is not None:
            self.dataset = select_num_samples(self.dataset, n_samples, clas_to_index)

        self.dataset.targets = torch.tensor([clas_to_index[clas.item()] for clas in self.dataset.targets])

        self.data, self.targets = (self.dataset.data.float().unsqueeze(-1) / 255.0).transpose(1, 3).numpy(), F.one_hot(
            self.dataset.targets, len(cls)
        ).numpy()

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        if self.transfrm is not None:
            img = self.transfrm(torch.from_numpy(img)).numpy()
        return img, target

    def __len__(self):
        return len(self.data)

train_samples = 1000
classes_train = [0,1,2,3,4,5,6,7,8,9]
n_classes = 10
batch_size = 256
test_batch_size = 256

data_train = MNIST(path_root= "data/",
            train=True, n_samples=train_samples if train_samples > 0 else None, cls=classes_train
        )
data_test = MNIST(path_root = "/data/", train=False, cls=classes_train)

if train_samples > 0:
    N = train_samples * n_classes
else:
    N = len(data_train)
N_test = len(data_test)
if test_batch_size > 0:
    test_batch_size = test_batch_size
else:
    test_batch_size = len(data_test)

n_test_batches = int(N_test / test_batch_size)
n_batches = int(N / batch_size)

train_loader = torch.utils.data.DataLoader(
    data_train, batch_size=batch_size, shuffle=True, collate_fn=numpy_collate_fn, drop_last=True,
)

valid_loader = torch.utils.data.DataLoader(
    data_test, batch_size=test_batch_size, shuffle=True, collate_fn=numpy_collate_fn, drop_last=True,
)


class ConvNet(nn.Module):
    output_dim: int = 10

    @nn.compact
    def __call__(self, x):
        if len(x.shape) != 4:
            x = jnp.expand_dims(x, 0)
        x = jnp.transpose(x, (0, 2, 3, 1))
        x = nn.Conv(features=4, kernel_size=(3, 3), strides=(2, 2), padding=1)(x)
        x = nn.tanh(x)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = nn.Conv(features=4, kernel_size=(3, 3), strides=(2, 2), padding=1)(x)
        x = nn.tanh(x)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = x.reshape((x.shape[0], -1))
        return nn.Dense(features=self.output_dim)(x)

def compute_num_params(pytree):
    return sum(x.size if hasattr(x, "size") else 0 for x in jax.tree_util.tree_leaves(pytree))


model = ConvNet()
batch = next(iter(train_loader))
x_init, y_init = batch["image"], batch["label"]
output_dim = y_init.shape[-1]
key, split_key = random.split(jax.random.PRNGKey(0))
params = model.init(key, x_init)
alpha = 1.
optim = optax.chain(
        optax.clip(1.),
        getattr(optax, "adam")(1e-2),
    )
opt_state = optim.init(params)
n_params = compute_num_params(params)
n_epochs = 100



def cross_entropy_loss(preds, y, rho=1.0):
    """
    preds: (n_samples, n_classes) (logits)
    y: (n_samples, n_classes) (one-hot labels)
    """
    preds = preds * rho
    preds = jax.nn.log_softmax(preds, axis=-1)
    return -jnp.sum(jnp.sum(preds * y, axis=-1))

def accuracy(params, model, batch_x, batch_y):
    preds = model.apply(params, batch_x)
    return jnp.sum(preds.argmax(axis=-1) == batch_y.argmax(axis=-1))


def map_loss(
    params,
    model,
    x_batch,
    y_batch,
    alpha,
    n_params: int,
    N_datapoints_max: int,
):
    # define dict for logging purposes
    B = x_batch.shape[0]
    O = y_batch.shape[-1]
    D = n_params
    N = N_datapoints_max

    # hessian_scaler = 1

    vparams = tm.Vector(params)

    rho = 1.
    nll = lambda x, y, rho: 1/B * cross_entropy_loss(x, y, rho)

    y_pred = model.apply(params, x_batch)

    loglike_loss = nll(y_pred, y_batch, rho) #* hessian_scaler

    log_prior_term = -D / 2 * jnp.log(2 * jnp.pi) - (1 / 2) * alpha * (vparams @ vparams) + D / 2 * jnp.log(alpha)
    # log_det_term = 0
    loss = loglike_loss - 0. * log_prior_term

    return loss

def make_step(params, alpha, opt_state, x, y):
    grad_fn = jax.value_and_grad(map_loss, argnums=0, has_aux=False)
    loss, grads = grad_fn(params, model, x, y, alpha, n_params, N)
    param_updates, opt_state = optim.update(grads, opt_state)
    params = optax.apply_updates(params, param_updates)
    return loss, params, opt_state

jit_make_step = jit(make_step)

for epoch in range(1, n_epochs + 1):
    epoch_loss = 0
    epoch_accuracy = 0
    start_time = time.time()
    for _, batch in zip(range(n_batches), train_loader):
        X = batch["image"]
        y = batch["label"]
        B = X.shape[0]
        train_key, split_key = random.split(split_key)

        loss, params, opt_state = jit_make_step(params, alpha, opt_state, X, y)
        loss = loss
        epoch_loss += loss.item()

        epoch_accuracy += accuracy(params, model, X, y).item()

    epoch_accuracy /= (n_batches * B)
    epoch_time = time.time() - start_time
    print(
        f"epoch={epoch}, loss={epoch_loss:.3f}, , accuracy={epoch_accuracy:.2f}, alpha={alpha:.2f}, time={epoch_time:.3f}s"
    )


sampling_train_loader = torch.utils.data.DataLoader(
    data_train, batch_size=N, shuffle=True, collate_fn=numpy_collate_fn, drop_last=True,
)
data = next(iter(sampling_train_loader))
x_train = jnp.array(data["image"])
y_train = jnp.array(data["label"])
sample_key = jax.random.PRNGKey(0)
n_posterior_samples = 200
num_iterations = 1
n_sample_batch_size = 1
n_sample_batches = N // n_sample_batch_size


# ground truth
from functools import partial

def calculate_exact_ggn(loss_fn, model_fn, params, X, y, n_params):
    def body_fun(carry, a_tuple):
        x, y = a_tuple
        my_model_fn = partial(model_fn, x=x)  # model_fn wrt parameters
        my_loss_fn = partial(loss_fn, y=y)  # loss_fn wrt model output
        pred = my_model_fn(params)
        jacobian = jax.jacfwd(my_model_fn)(params)
        jacobian = jax.tree_map(lambda x: jnp.reshape(x, (x.shape[0], -1)), jacobian)
        jacobian = jnp.concatenate(jax.tree_util.tree_flatten(jacobian)[0], axis=-1)
        loss_hessian = jax.hessian(my_loss_fn)(pred)
        ggn = jacobian.T @ loss_hessian @ jacobian
        return jax.tree_map(lambda a, b: a + b, carry, ggn), None

    init_value = jnp.zeros((n_params, n_params))  # jacobian.T @ loss_hessian @ jacobian
    return jax.lax.scan(body_fun, init_value, (X, y))[0]


_model_fn = lambda params, x: model.apply(params, x[None, ...])[0]
ggn = calculate_exact_ggn(cross_entropy_loss, _model_fn, params, x_train, y_train, n_params)


# low rank approximation

import matfree
from matfree import decomp, lanczos, montecarlo
from matfree.backend import func, linalg, np
from typing import Callable, Literal, Optional

def lanczos_tridiag(
        Av: Callable,
        v0: jax.Array,
        order: int
):
    ncols = v0.shape[0]
    if order >= ncols or order < 1:
        raise ValueError
    algorithm = matfree.lanczos.tridiagonal_full_reortho(order)
    u0 = v0/jnp.linalg.norm(v0)
    _, tridiag = decomp.decompose_fori_loop(u0, Av, algorithm=algorithm)
    (diag, off_diag) = tridiag
    diag = linalg.diagonal_matrix(diag)
    offdiag1 = linalg.diagonal_matrix(off_diag, -1)
    offdiag2 = linalg.diagonal_matrix(off_diag, 1)
    dense_matrix = diag + offdiag1 + offdiag2
    eigvals, _ = linalg.eigh(dense_matrix)
    return eigvals
  
  
# ggn
v0 = jnp.ones(n_params)
order = 100
eigvals = lanczos_tridiag(lambda v: ggn @ v, v0, order)

eigvals_gt, _ = jnp.linalg.eigh(ggn)

plt.figure()
plt.plot(eigvals, marker="o", linestyle=None, color="k", label="Lanczos")
plt.plot(eigvals_gt[-order:], marker="x", linestyle=None, label="Ground Truth")
plt.legend()
plt.show()