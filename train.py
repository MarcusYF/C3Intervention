import sys
import random
import torch
import torch.nn as nn
import torch.optim as optim
import itertools
import copy
from tqdm import tqdm
from collections import namedtuple

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward', 'step'))

sys.path.append('/u/fy4bc/code/research/C3Intervention')
from object import OUNoise, ReplayMemory
from graph import LightGraph, GNN, DQN, Actor, Critic
from envs_torch import R, u, sigma, gen_random_directions

if torch.cuda.is_available():
    N_EPOCH = 40
else:
    N_EPOCH = 40


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 16
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
TAU = 0.01
LR = 1e-3
n_episode = 500
HIDDEN_SIZE = 8
STATE_DIM = 10
NUM_CLUSTER = 5

eps = list(torch.linspace(float(EPS_START), float(EPS_END), int(N_EPOCH * 0.4)))
eps.extend(list(torch.linspace(float(EPS_END), 0.0, int(N_EPOCH * 0.6))))
eps.extend([0] * int(N_EPOCH))

ACTION_TYPE = 'cont_weight'  # ['cont_weight', 'temp', 'weight']
allowed_temp = [0.1, 0.3, 1.0, 3.3, 10.]
allowed_topk = [1, 2, 3]

users = torch.tensor([[0.,0], [0,1], [-1,0], [0,-1], [1,0]]).to(device)
d, m, n, top_k, random_topk = users.shape[1], users.shape[0], 5, None, False
method, lr, T = 'cgd', 1e-3, 5000
initial_strategy = gen_random_directions(n, d).to(device) * 0.1
initial_strategy = torch.tensor([[1.1,0.1], [0.9,0.1], [1.0,0.2], [0.8,0.1], [1.1,-0.1]]).to(device)*0.5

torch.manual_seed(1)
if ACTION_TYPE == 'weight':
    allowed_action = [torch.tensor([1., 1, 1, 1, 1]).to(device),
                      torch.tensor([2., 1, 1, 1, 1]).to(device),
                      torch.tensor([1., 2, 1, 1, 1]).to(device),
                      torch.tensor([1., 1, 2, 1, 1]).to(device),
                      torch.tensor([1., 1, 1, 2, 1]).to(device),
                      torch.tensor([1., 1, 1, 1, 2]).to(device)]
else:
    allowed_action = list(itertools.product(allowed_topk, allowed_temp))

graph_encoder = GNN().to(device)
policy_net = DQN(action_size=len(allowed_action)).to(device)
target_net = DQN(action_size=len(allowed_action)).to(device)
target_net.load_state_dict(policy_net.state_dict())
optimizer = optim.AdamW(list(graph_encoder.parameters()) + list(policy_net.parameters()), lr=LR, amsgrad=True)

actor = Actor(input_size=STATE_DIM, hidden_size=HIDDEN_SIZE, output_size=NUM_CLUSTER).to(device)
actor_target = Actor(input_size=STATE_DIM, hidden_size=HIDDEN_SIZE, output_size=NUM_CLUSTER).to(device)
critic = Critic(input_size=STATE_DIM + NUM_CLUSTER, hidden_size=HIDDEN_SIZE, output_size=1).to(device)
critic_target = Critic(input_size=STATE_DIM + NUM_CLUSTER, hidden_size=HIDDEN_SIZE, output_size=1).to(device)
actor_target.load_state_dict(actor.state_dict())
critic_target.load_state_dict(critic.state_dict())
critic_criterion = nn.MSELoss()
actor_optimizer = optim.AdamW(list(graph_encoder.parameters()) + list(actor.parameters()), lr=LR, amsgrad=True)
critic_optimizer = optim.AdamW(list(graph_encoder.parameters()) + list(critic.parameters()), lr=LR, amsgrad=True)

ou_noise = OUNoise(NUM_CLUSTER)

memory = ReplayMemory(1000)


def select_action(state, step, num_action=15):
    # global steps_done
    sample = random.random()
    eps_threshold = eps[step]
    # steps_done += 1
    if sample > eps_threshold:
        # if sample > 0:
        with torch.no_grad():
            # t.max(1) will return largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            # print(policy_net(graph_encoder(state)).shape)
            # print(policy_net(graph_encoder(state)))
            action_num = policy_net(graph_encoder(state)).mean(0).argmax().view(1, 1)

            # torch.tensor([allowed_topk[action_num // 5], allowed_temp[action_num % 5]], device=device)
            return action_num.item()
    else:
        return random.randint(0, num_action - 1)


def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([graph_encoder(s.G) for s in batch.next_state
                                       if s is not None])
    state_batch = torch.cat([graph_encoder(s.G) for s in batch.state
                             if s is not None])
    action_batch = torch.stack(batch.action)
    reward_batch = torch.stack(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # state_action_values = policy_net(state_batch).gather(1, action_batch.unsqueeze(1)) # (gnn output size * batch size, action space size + 1)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.

    # Take care of the final states
    next_state_values = torch.zeros((BATCH_SIZE, STATE_DIM), device=device)
    with torch.no_grad():
        next_state_values[non_final_mask] = non_final_next_states

    # Critic loss
    Qvals = critic.forward(state_batch, action_batch)
    next_actions = actor_target.forward(next_state_values)
    next_Q = critic_target.forward(next_state_values, next_actions)

    Qprime = reward_batch.unsqueeze(1) + GAMMA * next_Q
    critic_loss = critic_criterion(Qvals, Qprime)

    # Actor loss
    policy_loss = -critic.forward(state_batch, actor.forward(state_batch)).mean()
    # why is this loss a negative number? the loss is exploding, and gives trivial solution
    # print("Policy Loss: " + str(policy_loss))

    # update networks
    actor_optimizer.zero_grad()
    policy_loss.backward(retain_graph=True)
    torch.nn.utils.clip_grad_value_(actor.parameters(), 100)
    actor_optimizer.step()

    # print("Critic Loss: " + str(critic_loss))
    critic_optimizer.zero_grad()
    critic_loss.backward()
    torch.nn.utils.clip_grad_value_(critic.parameters(), 100)
    critic_optimizer.step()

    # Soft update of the target network's weights
    # θ′ ← τ θ + (1 −τ )θ′
    for target_param, param in zip(actor_target.parameters(), actor.parameters()):
        target_param.data.copy_(param.data * TAU + target_param.data * (1.0 - TAU))

    for target_param, param in zip(critic_target.parameters(), critic.parameters()):
        target_param.data.copy_(param.data * TAU + target_param.data * (1.0 - TAU))


for i_episode in tqdm(range(N_EPOCH)):
    # Initialize the environment and get it's state
    state = LightGraph(users, copy.deepcopy(initial_strategy), sigma)

    for t in range(n_episode):

        action = select_action(state.G, i_episode, num_action=len(allowed_action))

        if ACTION_TYPE == 'weight':
            user_weight = allowed_action[action]
            new_S, old_S = state.update_state(u, user_weight=user_weight)
        elif ACTION_TYPE == 'temp':
            action_specifics = allowed_action[action]
            new_S, old_S = state.update_state(u, tau=action_specifics[1], top_k=action_specifics[0])
        elif ACTION_TYPE == 'cont_weight':
            action = actor.forward(graph_encoder(state.G)).squeeze()
            noise = ou_noise.sample().to(device)
            # action = scipy.special.softmax(np.clip(action.detach().cpu().numpy()[0] + noise, -1, 1))
            action = torch.clamp(action + noise, -1, 1)
            action = torch.softmax(action, dim=0)
            # Added a softmax layer, clip between 0.1 and 10
            # do we want this to back propagate? Maybe not?
            # print(action)
            new_S, old_S = state.update_state(u, user_weight=action)
        else:
            print("Action Type Not Supported")
            break

        old_S_matrix = torch.zeros((state.n_player, state.n_user))
        for i_strategy in range(state.n_player):
            for i_user in range(state.n_user):
                old_S_matrix[i_strategy][i_user] = sigma(state.G.ndata["position"][i_user],
                                                         state.G.ndata["position"][state.n_user + i_strategy])

        if ACTION_TYPE in ['weight', 'cont_weight']:
            # if we define the
            # print(action)
            reward = R(old_S_matrix).to(device)  # need to change state to matrix
        else:
            reward = R(old_S_matrix, action_specifics).to(device)  # need to change state to matrix
        # print(reward)
        # print(new_S)
        # reward = torch.tensor([reward], device=device)
        old_state = LightGraph(users, old_S.detach(), sigma)
        if t == n_episode - 1:
            next_state = None
        else:
            next_state = LightGraph(users, new_S.detach(), sigma)
            # print(next_state.G.nodes)
            # print(next_state.G.edges)

        # Store the transition in memory
        # print('-------')
        # print(action)
        # print(torch.tensor(action, dtype=torch.int64))

        memory.push(old_state,
                    action.detach(),
                    next_state,
                    reward.detach(),
                    t)

        # Move to the next state
        state = next_state

        # Perform one step of the optimization (on the policy network)
        optimize_model()

    for param in graph_encoder.parameters():
        print(param)

# print('Complete')
# plot_durations(show_result=True)
# plt.ioff()
# plt.show()
