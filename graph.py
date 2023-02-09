from networkx.algorithms import bipartite
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd
import dgl
import sys
import copy
import networkx as nx
sys.path.append('/u/fy4bc/code/research/C3Intervention')
from envs_torch import gen_random_directions

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class LightGraph:
    batch_size: int = 1  # batch size
    s: int = 0  # number of strategies
    u: int = 0  # number of users
    ndata: dict = {}  # node attributes
    edata: dict = {}  # edge attributes

    # extend_info: tuple

    def __init__(self, users, players, sigma):

        self.n_user = len(users)
        self.n_player = len(players)
        self.G = nx.DiGraph()
        self.dim = users.shape[1]

        for index in range(self.n_user):
            self.G.add_node(index, position=users[index], bipartite=0)
        for index in range(self.n_player):
            self.G.add_node(index + self.n_user, position=players[index], bipartite=1)

        adj_mat = torch.zeros((self.n_user, self.n_player))
        for i in range(self.n_user):
            for j in range(self.n_player):
                self.G.add_edge(i, j + self.n_user, weight=sigma(players[j], users[i]))
                self.G.add_edge(j + self.n_user, i, weight=sigma(players[j], users[i]))
        self.ndata['adj'] = adj_mat
        self.G = dgl.from_networkx(self.G, node_attrs=["position"], edge_attrs=["weight"])
        self.G = self.G.to(device)

    def number_of_users(self):
        return self.n_user

    def number_of_players(self):
        return self.n_player

    def number_of_edges(self):
        return self.n * self.s * 2

    def update_state(self, utility_function, lr=0.01, tau=0.1, top_k=None, user_weight=None):
        users = self.G.ndata["position"][0:self.n_user]
        S = self.G.ndata["position"][self.n_user:]

        if user_weight is None:
            user_weight = torch.ones(self.n_user)
        # print("This is data upon retrieval")
        # print(users, S)

        u = utility_function
        # generate random direction
        g = gen_random_directions(self.n_player, self.dim).to(device)
        # find improvement direction
        utilities_new, _ = u(S + g * lr, users, tau=tau, top_k=top_k, user_weight=user_weight)
        utilities, allocation_matrix = u(S, users, tau=tau, top_k=top_k, user_weight=user_weight)
        diff = utilities_new - utilities
        g = torch.sign(diff).unsqueeze(1) * g
        old_S = S.detach()
        S += g * lr
        # Store the updated graph
        for i in range(self.n_player):
            self.G.ndata["position"][i + self.n_user] = S[i].detach()
        return S, old_S


# Need a better GNN structure
class GNN(nn.Module):
    def __init__(self, state_size=100):
        super(GNN, self).__init__()
        self.l1 = dgl.nn.pytorch.conv.GraphConv(2, 1, weight=True, bias=True)

    def forward(self, graphs):
        # graphs are 10x2
        # layer is 2x1
        # so result is 10x1, and then take a transpose, making it 1x10
        h = self.l1(graphs, graphs.ndata["position"].type(torch.FloatTensor).to(device), \
                    edge_weight=graphs.edata["weight"].reshape(-1, ).type(torch.FloatTensor).to(device))
        # print(h.shape)
        return h.transpose(0, 1)


class DQN(nn.Module):
    def __init__(self, input_size=10, hidden_size=8, action_size=6):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, action_size)

        self.fc3 = nn.Linear(input_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, action_size)

    def forward(self, x, action=None):
        return self.fc2(F.relu(self.fc1(x)))

    def forward_prop(self, x, action=None):
        return self.fc4(F.relu(self.fc3(x)))


class Critic(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Critic, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, output_size)

    def forward(self, state, action):
        """
        Params state and actions are torch tensors
        """
        x = torch.cat([state, action], 1)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)

        return x


class Actor(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Actor, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, output_size)

    def forward(self, state):
        """
        Param state is a torch tensor
        """
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        x = torch.tanh(self.linear3(x))

        return x