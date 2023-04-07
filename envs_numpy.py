import numpy as np
from scipy.special import softmax
import torch
from sklearn import preprocessing

np.set_printoptions(precision=4)
np.set_printoptions(suppress=True)


def gen_random_directions(n, d):
    return preprocessing.normalize(np.random.randn(n, d), norm='l2')


# scoring function
def sigma(S, users, tau=0.1, type='linear_'):
    """Relevance Score Function
    Args:
        S:     (n, d) - player strategies
        users: (m, d) - user embeddings
    Return:
        matrix of size (n, m) with each entry sigma(s_i, x_j)
    """
    if len(S.shape) == 1:
        S = S.reshape(1, -1)
    if len(users.shape) == 1:
        users = users.reshape(1, -1)

    m, n = users.shape[0], S.shape[0]
    if type == 'linear':
        return S @ users.T / tau
    else:
        dist_m = (np.sum(S ** 2, axis=1).reshape(n, 1).repeat(m, 1) + np.sum(users ** 2, axis=1).reshape(m, 1).repeat(n,
                                                                                                                      1).T - 2 * S @ users.T) ** 0.5
        return (1 - dist_m ** 1.0) / tau


# utility function
def u(S, users, tau=0.1, top_k=None, user_weight=None):
    """Players' Utility Function
    Args:
        S:           (n, d) - player strategies
        users:       (m, d) - user embeddings
        user_weight: (m, )  - assigned weight for each user
    Return:
        vector of size (n, ) with i-th entry u_i(s_i, s_{-i}; users)
    """
    scores = sigma(S, users, tau)
    m = users.shape[0]
    n = S.shape[0]

    if user_weight is None:
        user_weight = np.ones(m)

    if top_k is None:
        top_k = 1
    ind = np.argsort(scores, axis=0)[0:n - top_k]
    for i in range(m):
        scores[ind[:, i], i] -= 1E3
    allocation_matrix = softmax(scores, axis=0)
    utilities = np.sum(allocation_matrix * np.log(np.exp(scores).sum(axis=0)) * user_weight, axis=1) / m
    return utilities, allocation_matrix


# welfare function - definition 1
# def W(S, users, tau=0.1, top_k=None, random=False):
#   """Social Welfare Function
#   Return:
#       summation of all players' utilities
#   """
#   utilities, allocation_matrix = u(S, users, tau, top_k)
#   return np.sum(utilities)

# welfare function - definition 2
def V(s, users):
    m = len(users)
    M = [np.log(sum([np.exp(sigma(s_i, x_j)) for s_i in s])) for x_j in users]
    return sum(M)[0][0] / m


# def R(s, a=None, beta=0.1):
#     """Reward Function (Stochastic Version)
#     Args:
#         s: (n, m) - the state, represented by a relevance matrix whose (i, j)-th entry denotes the relevance score between the i-th player and the j-th user.
#         a: (2, )  - the action, a tuple containing two values K, \tau.
#         beta:     - the temperature in user decision
#     Return:
#         The total user welfare under the allocation rule determined by (K, \tau).
#     """
#     if a is None:
#         K, tau = 1, 0.1
#     else:
#         if torch.is_tensor(a):
#             K, tau = a[0].long().item(), a[1].item()
#         else:
#             K, tau = a[0], a[1]
#     # generate the distribution where the K-list is sampled from
#     dist = tfp.distributions.PlackettLuce(softmax(s / tau, axis=0).T)
#     # sampled indices. size=(m, K)
#     K_list = dist.sample()[:, :K]
#     # sampled scores. size=(K, m)
#     scores = tf.gather_nd(
#         indices=np.stack([K_list.numpy().T, np.vstack([np.array(range(m))] * K)]).transpose(1, 2, 0),
#         params=s).numpy()
#     # welfare defined as total user utility
#     welfare = beta * np.sum(np.log(np.exp(scores / beta).sum(axis=0)))
#     return welfare

def R_diff(s, new_s, beta=0.1):
    """Reward Function (Stochastic Version)
    Args:
        s: (n, m) - the state, represented by a relevance matrix whose (i, j)-th entry denotes the relevance score between the i-th player and the j-th user.
        new_s: (n, m)  - the newer state
        beta:     - the temperature in user decision
    Return:
        The total user welfare under the allocation rule determined by (K, \tau).
    """
    K, tau = 1, 0.1
    # generate the distribution where the K-list is sampled from
    dist = tfp.distributions.PlackettLuce( softmax(s / tau, axis=0).T )
    # sampled indices. size=(m, K)
    K_list = dist.sample()[:, :K]
    # sampled scores. size=(K, m)
    scores = tf.gather_nd(
        indices=np.stack([K_list.numpy().T, np.vstack([np.array(range(m))] *K) ]).transpose(1,2,0),
        params = s).numpy()
    # welfare defined as total user utility
    old_welfare = beta * np.sum(np.log(np.exp(scores / beta).sum(axis=0)))

    # new welfare
    dist = tfp.distributions.PlackettLuce( softmax(new_s / tau, axis=0).T )
    # sampled indices. size=(m, K)
    K_list = dist.sample()[:, :K]
    # sampled scores. size=(K, m)
    scores = tf.gather_nd(
        indices=np.stack([K_list.numpy().T, np.vstack([np.array(range(m))] *K) ]).transpose(1,2,0),
        params = new_s).numpy()
    new_welfare = beta * np.sum(np.log(np.exp(scores / beta).sum(axis=0)))
    return new_welfare - old_welfare

def R(s, a=None):
    """Reward Function (Deterministic Version)
    Args:
        s: (n, m) - the state, represented by a relevance matrix whose (i, j)-th entry denotes the relevance score between the i-th player and the j-th user.
        a: (2, )  - the action, a tuple containing two values K, \tau.
        beta:     - the temperature in user decision
    Return:
        The total user welfare under the allocation rule determined by (K, \tau).
    """
    if a is None:
        K, tau = 1, 0.1
    else:
        if torch.is_tensor(a):
            K, tau = a[0].long().item(), a[1].item()
        else:
            K, tau = a[0], a[1]
    n = s.shape[0]
    ## mask non-topK elements with -infty
    ind = np.argsort(s, axis=0)[0:n - K]
    print('ind:', ind)
    mask_scores = np.copy(s)
    for i in range(m):
        mask_scores[ind[:, i], i] -= 1E3
    print('mask_scores:', mask_scores)
    # sample probability
    sample_prob = softmax(mask_scores / tau, axis=0)
    # welfare defined as expected user utility
    welfare = np.sum(sample_prob * s)
    return welfare

# # test R()
# n, m = 4, 5
# s = np.random.rand(n, m)
# print(s)
# print("-----")
# # print(R(s=s, a=(2, 0.1), beta=0.01))
# print("-----")
# print(R(s=s, a=(2, 0.1)))

# # test sigma()
# m, n, d = 3, 4, 5
# S = np.random.rand(n, d)
# users = np.random.rand(m, d)
# sigma(S, users, tau=0.1, type='linear_')

# # test u()
# m, n, d = 3, 4, 5
# S = np.random.rand(n, d)
# users = np.random.rand(m, d)
# u(S, users, tau=0.1, top_k=2, user_weight=None)

