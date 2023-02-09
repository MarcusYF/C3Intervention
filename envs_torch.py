import torch


def gen_random_directions(n, d):
    return torch.nn.functional.normalize(torch.randn(n, d), dim=1)


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
        return torch.matmul(S, users.t()) / tau
    else:
        S_2 = torch.sum(S ** 2, dim=1, keepdim=True).repeat(1, m)
        users_2 = torch.sum(users ** 2, dim=1, keepdim=True).repeat(1, n).t()
        dist_m = (S_2 + users_2 - 2 * torch.matmul(S, users.t())) ** 0.5
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
        user_weight = torch.ones(m)

    if top_k is None:
        top_k = 1
    ind = torch.argsort(scores, dim=0)[0:n - top_k]
    scores[ind, torch.arange(m)] -= 1E3
    allocation_matrix = torch.nn.functional.softmax(scores, dim=0)
    exp_scores = torch.exp(scores)
    sum_exp_scores = torch.sum(exp_scores, dim=0)
    utilities = torch.sum(allocation_matrix * torch.log(sum_exp_scores) * user_weight, dim=1) / m
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
    exp_scores = [torch.exp(sigma(s_i, users)) for s_i in s]
    sum_exp_scores = torch.sum(torch.stack(exp_scores), dim=0)
    M = torch.log(sum_exp_scores)
    return torch.mean(M)


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
    n, m = s.shape[0], s.shape[1]
    # mask non-topK elements with -infty
    ind = torch.argsort(s, dim=0, descending=False)[0:n - K]
    mask_scores = s.clone()
    for i in range(m):
        mask_scores[ind[:, i], i] = float("-inf")
    # sample probability
    sample_prob = torch.softmax(mask_scores / tau, dim=0)
    # welfare defined as expected user utility
    welfare = torch.sum(sample_prob * s)
    return welfare

# # test R()
# n, m = 4, 5
# st = torch.tensor(s)
# print(st)
# print("-----")
# # print(R(s=s, a=(2, 0.1), beta=0.01))
# print("-----")
# print(R(s=st, a=torch.tensor([2, 0.1])))

# # test sigma()
# S = torch.tensor(S)
# users = torch.tensor(users)
# sigma(S, users, tau=0.1, type='linear_')

# # test u()
# m, n, d = 3, 4, 5
# S = torch.tensor(S)
# users = torch.tensor(users)
# u(S, users, tau=0.1, top_k=2, user_weight=None)

