import sys
import random
import torch
import torch.nn as nn
import torch.optim as optim
import itertools
import copy
from tqdm import tqdm
from collections import namedtuple
import datetime
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward', 'step'))

sys.path.append('/u/fy4bc/code/research/C3Intervention')
from object import OUNoise, ReplayMemory
from graph import LightGraph, GNN, DQN, Actor, Critic
from envs_torch import R, u, sigma, gen_random_directions, V


# from IPython.core.autocall import ZMQExitAutocall
from numpy import exp,arange
from pylab import meshgrid,cm,imshow,contour,clabel,colorbar,axis,title,show
from matplotlib import pyplot as plt
from tqdm import tqdm

# the function that I'm going to plot
def z_func(X, Y, users):
    Z = np.zeros_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            s = np.vstack([np.array([X[i,j], Y[i,j]]),zero_pad])
            Z[i, j] = V(torch.Tensor(s), users)
    return Z

def fit_xyticks(trace):
    return (trace+2)*20

def plot_S_trace(S_history, users, file_name="", n_step=100):
    x = arange(-2.0,2.0,0.05)
    y = arange(-2.0,2.0,0.05)
    X, Y = meshgrid(x, y) # grid of point
    Z = z_func(X, Y, users) # evaluation of the function on the grid

    colors = ['aqua', 'black', 'darkblue', 'darkgreen', 'darkviolet', 'gold']
    plt.figure(figsize = (15,15))
    im = plt.imshow(Z, cmap=cm.RdBu) # drawing the function
    print(S_history.shape)
    for i in range(n):
        plt.scatter(fit_xyticks(S_history[:n_step,i,0]),fit_xyticks(S_history[:n_step,i,1]), color=colors[i], marker='.')

    # adding the Contour lines with labels
    cset = contour(Z, arange(3.1,4.3,0.1), linewidths=1, cmap=cm.Set2)
    clabel(cset, inline=True, fmt='%1.1f', fontsize=10)
    colorbar(im) # adding the colobar on the right
    # latex fashion title
    title('V(s)')
    plt.savefig("./" + result_dir + "trace_visualization" + file_name)
    show()
    plt.clf()

if torch.cuda.is_available():
    N_EPOCH = 20
else:
    N_EPOCH = 20

current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
train_log_dir = './' + current_time + '/'
result_dir = './results/' + train_log_dir + '/'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 16
GAMMA = 0.995
EPS_START = 0.9
EPS_END = 0.5
TAU = 0.001
LR = 1e-4
n_episode = 2000
env_steps = 10
HIDDEN_SIZE = 32
NUM_CLUSTER = 5

target_update_step = 10
update_rate_start = 0.1
update_rate_end = 0.01
update_rate = list(torch.linspace(float(update_rate_start), float(update_rate_end), int(n_episode * 0.4)))
update_rate.extend([float(update_rate_end)] * int(n_episode * 0.6))

eps = list(torch.linspace(float(EPS_START), float(EPS_END), int(N_EPOCH * 0.4)))
eps.extend(list(torch.linspace(float(EPS_END), 0.0, int(N_EPOCH * 0.6))))
eps.extend([0] * int(N_EPOCH))

ACTION_TYPE = 'cont_weight'  # ['cont_weight', 'temp', 'weight']
allowed_temp = [0.1, 0.3, 1.0, 3.3, 10.]
allowed_topk = [1, 2, 3]

''' 5 users '''
users = torch.tensor([[0.,0], [0,1], [-1,0], [0,-1], [1,0]]).to(device)

''' 9 users '''
# SQT2 = np.sqrt(2)/2
# users = torch.FloatTensor([[0.,0], [0,1], [-1,0], [0,-1], [1,0], [SQT2, SQT2], [-SQT2, SQT2], [-SQT2, -SQT2], [SQT2, -SQT2]]).to(device)

method = 'cgd'

initial_strategy = torch.tensor([[1.1,0.1], [0.9,0.1], [1.0,0.2], [0.8,0.1], [1.1,-0.1]]).to(device)*0.5
# initial_strategy = torch.tensor([[0.1, 1.1], [0.1, 0.9], [0.2, 1.0], [0.1, 0.8], [-0.1, 1.1]]).to(device)*0.5
# initial_strategy = torch.tensor([[0,0], [0.,1], [1.0,0], [-1,0.], [0.5,0.5]]).to(device) # local optimum 1
# initial_strategy = torch.tensor([[0,0], [0.,1], [1.0,0], [0,0.5], [-0.5,-0.5]]).to(device) # local optimum 2
d, m, n, top_k, random_topk = users.shape[1], users.shape[0], len(initial_strategy), None, False
STATE_DIM = m+n

# initial_strategy = gen_random_directions(n, d).to(device) * 0.1

torch.manual_seed(21)
if ACTION_TYPE == 'weight':
    allowed_action = [
        torch.tensor([1., 1, 1, 1, 1]).to(device),
        torch.tensor([2., 1, 1, 1, 1]).to(device),
        torch.tensor([1., 2, 1, 1, 1]).to(device),
        torch.tensor([1., 1, 2, 1, 1]).to(device),
        torch.tensor([1., 1, 1, 2, 1]).to(device),
        torch.tensor([1., 1, 1, 1, 2]).to(device),
    ]

    # allowed_action = [
    #                 #   torch.tensor([1., 1, 1, 1, 1]).to(device),
    #                 #   torch.tensor([2., 1, 1, 1, 1]).to(device),
    #                 #   torch.tensor([1., 2, 1, 1, 1]).to(device),
    #                   torch.tensor([1., 1, 100, 1, 1]).to(device),
    #                 #   torch.tensor([1., 1, 1, 2, 1]).to(device),
    #                 #   torch.tensor([1., 1, 1, 1, 100]).to(device),
    #                 ]
else:
    allowed_action = list(itertools.product(allowed_topk, allowed_temp))

graph_encoder = GNN().to(device)
policy_net = DQN(input_size=m+n, action_size=len(allowed_action)).to(device)
target_net = DQN(input_size=m+n, action_size=len(allowed_action)).to(device)
target_net.load_state_dict(policy_net.state_dict())
optimizer = optim.AdamW(list(graph_encoder.parameters()) + list(policy_net.parameters()), lr=LR, amsgrad=True)

actor = Actor(input_size=STATE_DIM, hidden_size=HIDDEN_SIZE, output_size=NUM_CLUSTER).to(device)
actor_target = Actor(input_size=STATE_DIM, hidden_size=HIDDEN_SIZE, output_size=NUM_CLUSTER).to(device)
critic = Critic(state_size=STATE_DIM, action_size=NUM_CLUSTER, hidden_size=HIDDEN_SIZE, output_size=1).to(device)
critic_target = Critic(state_size=STATE_DIM, action_size=NUM_CLUSTER, hidden_size=HIDDEN_SIZE, output_size=1).to(device)
actor_target.load_state_dict(actor.state_dict())
critic_target.load_state_dict(critic.state_dict())
critic_criterion = nn.MSELoss()
actor_optimizer = optim.AdamW(list(actor.parameters()), lr=LR, amsgrad=True)
critic_optimizer = optim.AdamW(list(critic.parameters()), lr=1e-3, amsgrad=True)

ou_noise = OUNoise(NUM_CLUSTER)

memory = ReplayMemory(5000)

def select_action_dqn(state, eps_threshold, num_action=15):
    # return random.randint(0, num_action - 1) # TODO: change this back

    sample = random.random()
    # eps_threshold = eps[step]
    # eps_threshold = 1
    # print(eps_threshold)
    if sample > eps_threshold:
        # if sample > 0:
        with torch.no_grad():
            # t.max(1) will return largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.

            action_num = policy_net(graph_encoder(state)).mean(0).argmax().view(1, 1)
            # print(policy_net(graph_encoder(state)))
            # print(action_num)
            return action_num.item()
    else:
        return random.randint(0, num_action - 1)

def select_action(state, step):
    if step == -1:
        noise = ou_noise.sample().to(device)

    # eps_threshold = eps[step]
    action = actor.forward(graph_encoder(state)).squeeze()

    ''' Debugging purpose: if we manually set the weight, what would the result be like? '''
    # action = torch.Tensor([-1, -1, -1, -1, -1]).to(device)
    # action = torch.Tensor([-1, 1, 0.5, -1, -1]).to(device)

    # add randomness to action selection for exploration
    noise = ou_noise.sample().to(device)

    # print("-------")
    # print("Initial Action: ")
    # print(action)
    action = torch.clamp(action + noise, -10, 10)
    # print("Noise: ")
    # print(noise)
    action = torch.softmax(action, dim=0)
    # print("Final Action: ")
    # print(action)
    noise = noise.detach().cpu()

    return action

def optimize_model(return_loss=True):
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
    # print("----- State Batch -----")
    # print(torch.mean(torch.abs(torch.cat([s.G.edata["weight"] for s in batch.state]))))
    # print(torch.mean(torch.abs(state_batch)))
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

    print("----- critic loss -----")
    # Critic loss
    next_actions = actor_target.forward(next_state_values)
    next_Q = critic_target.forward(next_state_values, next_actions)
    Qprime = reward_batch.unsqueeze(1) + GAMMA * next_Q
    critic.zero_grad()
    # print("----- State and Action -----")
    # print(torch.mean(torch.abs(state_batch)))
    # print(torch.mean(action_batch))

    # print("----- critic param -----")
    # for param in critic.parameters():
    #     print(torch.mean(torch.abs(param)))
    Qvals = critic.forward(state_batch, action_batch) # TODO: this part explodes
    critic_loss = critic_criterion(Qvals, Qprime)

    # print("Critic Loss: " + str(critic_loss))
    critic_optimizer.zero_grad()
    critic_loss.backward(retain_graph=True)
    torch.nn.utils.clip_grad_value_(critic.parameters(), 100)
    critic_optimizer.step()

    # Actor loss

    print("----- actor loss -----")
    actor.zero_grad()
    # Why do we need a negative sign here??
    # policy_lo、ss = -critic.forward(state_batch, actor.forward(state_batch)).mean()

    policy_loss = -critic.forward(state_batch, actor.forward(state_batch)).mean()

    # TODO: This is still the issue
    # print(critic_loss)
    # print(policy_loss)
    # why is this loss a negative number? the loss is exploding, and gives trivial solution
    # print("Policy Loss: " + str(policy_loss))

    # update networks
    actor_optimizer.zero_grad()
    policy_loss.backward()
    torch.nn.utils.clip_grad_value_(actor.parameters(), 100)
    actor_optimizer.step()


    if return_loss:
        return policy_loss.cpu().detach().numpy()


def optimize_model_dqn(return_loss=False):
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
                                            batch.next_state)), device=device, dtype=torch.bool).to(device)
    non_final_next_states = torch.cat([graph_encoder(s.G) for s in batch.next_state
                                       if s is not None]).to(device)
    state_batch = torch.cat([graph_encoder(s.G) for s in batch.state
                             if s is not None]).to(device)
    action_batch = torch.stack(batch.action).to(device)
    reward_batch = torch.Tensor(batch.reward).to(device)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken

    # state_batch = (batch_size * 1, 10)
    # where (1, 10) is the shape of output of GNN

    # shape (16, 1) because 16 is a batch,
    # policy_net(state_batch)
    state_action_values = policy_net(state_batch).gather(1, action_batch.unsqueeze(1)) # (gnn output size * batch size, action space size + 1)
    # print("Policy net values: ")
    # print(policy_net(state_batch))

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0]
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    # criterion = nn.SmoothL1Loss()
    criterion = nn.MSELoss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))
    # print("-----")
    # print(batch.step)
    # print(non_final_mask)
    # print(next_state_values)
    # print(state_action_values - expected_state_action_values.unsqueeze(1))
    # print("Q_loss = " + str(loss))
    # print(loss)

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    # In-place gradient clipping
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()

    if return_loss:
        return loss.cpu().detach().numpy()

if not os.path.exists(result_dir):
    os.makedirs(result_dir)

if N_EPOCH > 0:

    initial_strategies = [
        torch.tensor([[1.1,0.1], [0.9,0.1], [1.0,0.2], [0.8,0.1], [1.1,-0.1]]).to(device)*1.5,
        torch.tensor([[1.1,0.1], [0.9,0.1], [1.0,0.2], [0.8,0.1], [1.1,-0.1]]).to(device)*-1.5,
        torch.tensor([[0.1, 1.1], [0.1, 0.9], [0.2, 1.0], [0.1, 0.8], [-0.1, 1.1]]).to(device)*1.5,
        torch.tensor([[0.1, 1.1], [0.1, 0.9], [0.2, 1.0], [0.1, 0.8], [-0.1, 1.1]]).to(device)*-1.5,
        ]
    action_history = []
    welfare_history = []
    S_history = []
    q_loss_history = []
    reward_history = []
    for i_episode in tqdm(range(N_EPOCH)): # 20
        initial_strategy = random.choice(initial_strategies)
        print(initial_strategy)
        # Initialize the environment and get it's state
        state = LightGraph(users, copy.deepcopy(initial_strategy), sigma)

        for t in range(n_episode//env_steps): # 1000
            if ACTION_TYPE != 'cont_weight':
                action = select_action_dqn(state.G, eps[i_episode], num_action=len(allowed_action))
                # print(action)
                # action = 1
            else:
                action = select_action(state.G, i_episode)

            if ACTION_TYPE == 'weight':
                # Final state
                if t == n_episode - 1:
                    action = 0
                    old_S = state.G.ndata["position"][m:]
                    new_S = copy.deepcopy(old_S)
                    user_weight = allowed_action[0]
                    W, _ = u(old_S, users, tau=0.1, top_k=1, user_weight=user_weight)
                    # print(W)
                user_weight = allowed_action[action]
                for inner in range(env_steps): # inner loop has env_steps steps
                    new_S, old_S, W = state.update_state(u, lr=update_rate[t], tau=0.1, top_k=1, user_weight=user_weight, return_W=True)
                action = torch.tensor(action, dtype=torch.int64)
                action_history.append(action.cpu().detach())
                welfare_history.append(W.cpu().detach())
                S_history.append(new_S.cpu().detach())
            elif ACTION_TYPE == 'temp':
                action_specifics = allowed_action[action]
                new_S, old_S = state.update_state(u, lr=update_rate[t], tau=action_specifics[1], top_k=action_specifics[0])
            elif ACTION_TYPE == 'cont_weight':
                # Final state
                if t == n_episode - 1:
                    old_S = state.G.ndata["position"][m:]
                    new_S = copy.deepcopy(old_S)
                    W, _ = u(old_S, users, tau=0.1, top_k=1)

                for inner in range(env_steps):
                    new_S, old_S, W = state.update_state(u, lr=update_rate[t], user_weight=action, return_W=True)
                action_history.append(action.cpu().detach())
                welfare_history.append(W.cpu().detach())
                S_history.append(new_S.cpu().detach())
                action = action.detach()
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
            reward_history.append(reward.cpu().detach())

            old_state = LightGraph(users, old_S.detach(), sigma)
            # if t == n_episode - 1:
            #     next_state = None
            # else:
            next_state = LightGraph(users, new_S.detach(), sigma)

            # Store the transition in memory
            memory.push(old_state,
                        action,
                        next_state,
                        reward.detach(),
                        t)

            # Move to the next state
            state = next_state

            # Perform one step of the optimization (on the policy network)
            if ACTION_TYPE != 'cont_weight':
                q_loss = optimize_model_dqn(return_loss=True)
                q_loss_history.append(q_loss)
            else:
                q_loss = optimize_model(return_loss=True)
                q_loss_history.append(q_loss)

            # Soft update of the target network's weights
            # θ′ ← τ θ + (1 −τ )θ′
            if t % target_update_step == 0:
                if ACTION_TYPE == 'weight':
                    for target_param, param in zip(target_net.parameters(), policy_net.parameters()):
                        target_param.data.copy_(param.data * TAU + target_param.data * (1.0 - TAU))

                elif ACTION_TYPE == 'cont_weight':
                    for target_param, param in zip(actor_target.parameters(), actor.parameters()):
                        target_param.data.copy_(param.data * TAU + target_param.data * (1.0 - TAU))

                    for target_param, param in zip(critic_target.parameters(), critic.parameters()):
                        target_param.data.copy_(param.data * TAU + target_param.data * (1.0 - TAU))


        np_users = users.detach().cpu()
        zero_pad = np.zeros_like(np_users[:-1,:])

        test_S_history = np.zeros((n_episode//env_steps, n, 2))
        for index in range(n_episode//env_steps):
            test_S_history[index] = S_history[i_episode*n_episode//env_steps + index].detach().cpu().numpy()
        print(np.array(test_S_history))
        print(np.array(test_S_history).shape)
        print(np.array(action_history[i_episode*n_episode//env_steps:]))
        plot_S_trace(test_S_history, np_users, file_name=str(i_episode), n_step=n_episode)

    ts = np.vstack(action_history)

    # # Plot the time series data
    # plt.plot(ts[:, 0], label='center')
    # plt.plot(ts[:, 1], label='top')
    # plt.plot(ts[:, 2], label='left')
    # plt.plot(ts[:, 3], label='bottom')
    # plt.plot(ts[:, 4], label='right')
    # plt.plot([np.mean(ts.flatten()[i:i+100]) for i in range(20000-100)], label='action')
    fig = plt.figure(figsize=(10, 6))
    plt.plot(welfare_history, color='r', label='welfare')
    # plt.plot(reward_history, color='r', label='reward')
    # plt.plot(q_loss_history[:], color='r', label='q-loss')
    # plt.plot(action_history[1000:1500], color='b', label='action')
    plt.xlabel('time step')
    plt.ylabel('weight')
    plt.title('training')
    plt.legend(loc='best')
    plt.show()
    plt.savefig("./" + result_dir + "welfare")
    plt.clf()

    fig = plt.figure(figsize=(10, 6))
    # plt.plot(welfare_history, color='r', label='welfare')
    plt.plot(reward_history, color='r', label='reward')
    # plt.plot(q_loss_history[:], color='r', label='q-loss')
    # plt.plot(action_history[1000:1500], color='b', label='action')
    plt.xlabel('time step')
    plt.ylabel('weight')
    plt.title('training')
    plt.legend(loc='best')
    plt.show()
    plt.savefig("./" + result_dir + "reward")


    fig = plt.figure(figsize=(10, 6))
    # plt.plot(welfare_history, color='r', label='welfare')
    # plt.plot(reward_history, color='r', label='reward')
    plt.plot(q_loss_history, color='r', label='q-loss')
    # plt.plot(action_history[1000:1500], color='b', label='action')
    plt.xlabel('time step')
    plt.ylabel('weight')
    plt.title('training')
    plt.legend(loc='best')
    plt.show()
    plt.savefig("./" + result_dir + "q-loss")


    fig = plt.figure(figsize=(10, 6))
    plt.plot([np.mean(ts.flatten()[i:i+100]) for i in range(20000-100)], label='action')
    # plt.plot(welfare_history, color='r', label='welfare')
    # plt.plot(reward_history, color='r', label='reward')
    # plt.plot(q_loss_history[:], color='r', label='q-loss')
    # plt.plot(action_history[1000:1500], color='b', label='action')
    plt.xlabel('time step')
    plt.ylabel('weight')
    plt.title('training')
    plt.legend(loc='best')
    plt.show()
    plt.savefig("./" + result_dir + "action")
    plt.clf()

test_strategy = torch.tensor([[1.1,0.1], [0.9,0.1], [1.0,0.2], [0.8,0.1], [1.1,-0.1]]).to(device)*0.5
# test_strategy = torch.tensor([[0.1, 1.1], [0.1, 0.9], [0.2, 1.0], [0.1, 0.8], [-0.1, 1.1]]).to(device)*0.5
# test_strategy = torch.tensor([[0,0], [0.,1], [1.0,0], [-1,0.], [0.5,0.5]]).to(device) # local optimum 1
# test_strategy = torch.tensor([[0,0], [0.,1], [1.0,0], [0,0.5], [-0.5,-0.5]]).to(device) # local optimum 2
# test_strategy = initial_strategy

# test_strategies = [
#         torch.tensor([[1.1,0.1], [0.9,0.1], [1.0,0.2], [0.8,0.1], [1.1,-0.1]]).to(device)*1.5,
#         torch.tensor([[1.1,0.1], [0.9,0.1], [1.0,0.2], [0.8,0.1], [1.1,-0.1]]).to(device)*-1.5,
#         torch.tensor([[0.1, 1.1], [0.1, 0.9], [0.2, 1.0], [0.1, 0.8], [-0.1, 1.1]]).to(device)*1.5,
#         torch.tensor([[0.1, 1.1], [0.1, 0.9], [0.2, 1.0], [0.1, 0.8], [-0.1, 1.1]]).to(device)*-1.5,
# ]
final_rewards = []

for i in range(20):
    torch.manual_seed(i)
    # test_strategy = random.choice(initial_strategies)
    state = LightGraph(users, copy.deepcopy(test_strategy), sigma)
    test_S_history = np.zeros((n_episode, n, 2))
    action_history = []
    for t in range(n_episode//env_steps):

        if ACTION_TYPE != 'cont_weight':
            action = select_action_dqn(state.G, 0.0, num_action=len(allowed_action))
        else:
            action = select_action(state.G, -1)
        # print(action)
        if ACTION_TYPE == 'weight':
            user_weight = allowed_action[action]
            for inner in range(env_steps):
                new_S, old_S, welfare = state.update_state(u, lr=update_rate[t], tau=0.1, top_k=1, user_weight=user_weight, return_W=True)
            action_history.append(action)
        elif ACTION_TYPE == 'temp':
            action_specifics = allowed_action[action]
            new_S, old_S = state.update_state(u, lr=update_rate[t], tau=action_specifics[1], top_k=action_specifics[0])
        elif ACTION_TYPE == 'cont_weight':
            # Added a softmax layer, clip between 0.1 and 10
            # do we want this to back propagate? Maybe not?
            # print(action)
            for inner in range(env_steps):
                new_S, old_S, welfare = state.update_state(u, lr=update_rate[t], user_weight=action, return_W=True)
            action_history.append(action.cpu().detach())
            # welfare_history.append(W.cpu().detach())
            # S_history.append(new_S.cpu().detach())
            action = action.detach()
            print(new_S)
        else:
            print("Action Type Not Supported")
            break

        # old_S_matrix = torch.zeros((state.n_player, state.n_user))
        # for i_strategy in range(state.n_player):
        #     for i_user in range(state.n_user):
        #         old_S_matrix[i_strategy][i_user] = sigma(state.G.ndata["position"][i_user],
        #                                                     state.G.ndata["position"][state.n_user + i_strategy])
        # if ACTION_TYPE in ['weight', 'cont_weight']:
        #     # if we define the
        #     # print(action)
        #     reward = R(old_S_matrix) # need to change state to matrix
        # else:
        #     reward = R(old_S_matrix, action_specifics) # need to change state to matrix
        # print(reward)
        # print(new_S)
        # reward = torch.tensor([reward], device=device)
        # old_state = copy.deepcopy(state)
        test_S_history[t, :, :] = old_S.cpu().detach().numpy()
        # if t == n_episode - 1:
        #     next_state = None
        # else:
        #     next_state = LightGraph(users, copy.deepcopy(new_S), sigma)
        # # Move to the next state
        # state = next_state

    print(action_history)
    print(new_S)

    print(V(new_S, users))

    test_users = users.detach().cpu()
    zero_pad = np.zeros_like(test_users[:-1,:])

    plot_S_trace(test_S_history, test_users, file_name="test_" + str(i), n_step=n_episode)
    final_rewards.append(V(new_S, users))
print(final_rewards)