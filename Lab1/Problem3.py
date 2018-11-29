import os
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict
import time;

# GAME PROPRETIES
BOARD_WIDTH = 4
BOARD_HEIGHT = 4
N_TILES = BOARD_HEIGHT * BOARD_WIDTH

STAY = 0
UP = -BOARD_WIDTH
RIGHT = 1
DOWN = BOARD_WIDTH
LEFT = -1

ROBBER_ACTIONS = [STAY, UP, RIGHT, DOWN, LEFT]
POLICE_ACTIONS = [UP, RIGHT, DOWN, LEFT]

N_ROBBER_ACTIONS = len(ROBBER_ACTIONS)
N_POLICE_ACTIONS = len(POLICE_ACTIONS)

N_STATES = N_TILES**2

INITIAL_STATE=0+15*N_TILES;

# PARAMETERS
n_iterations=10**7;
discount_factor = 0.8

##PLOTTING PARAMS

ivs_step=10000;



def rc2idx(rc):	
	'''Converts row column position to index'''
	return rc[0] * BOARD_WIDTH + rc[1]


def idx2rc(idx):
	'''Converts index to row column position '''
	return np.array([idx // BOARD_WIDTH, idx % BOARD_WIDTH])


def rc_exists(rc):
	'''Check that a certain position exists.'''
	return -1 < rc[0] < BOARD_HEIGHT and -1 < rc[1] < BOARD_WIDTH


def ai2si(player_idx, police_idx):
	'''Converts agent indices to state index (int)'''
	return police_idx * N_TILES + player_idx


def si2ai(state_index):
	'''Takes state index and returns player idx and police idx'''
	return state_index % N_TILES, state_index // N_TILES


def iv2si(index_vector):
	'''Converts [player_idx,police_idx] (list-like) to state index (int)'''
	return index_vector[1] * BOARD_WIDTH * BOARD_HEIGHT + index_vector[0]


def move(current_pos, action):
	'''takes a position index and a move (int representing number of squares to move)
			and outputs the resulting position.
	'''
	try_pos = current_pos + action
	column = current_pos % BOARD_WIDTH
	if (not -1 < try_pos < BOARD_HEIGHT * BOARD_WIDTH)\
			or (column == 0 and action == -1)\
			or (column == BOARD_WIDTH - 1 and action == 1):
		return current_pos
	else:
		return try_pos

def reward(state, action_idx):
	'''computes reward for a given state and action'''
	reward = 0
	player_idx, police_idx = si2ai(state)
	if player_idx == 5:
		reward += 1
	if police_idx == player_idx:
		reward -= 10
	return reward

def next_state(state, action_idx):
	action=ROBBER_ACTIONS[action_idx];
	player_idx, police_idx = si2ai(state)
	new_player_idx = move(player_idx, action)
	police_move = POLICE_ACTIONS[int(np.random.uniform(0, N_POLICE_ACTIONS))]
	new_police_idx = move(police_idx, police_move)
	return ai2si(new_player_idx, new_police_idx)

def try_policy(policy,T,_print=False):
	state=INITIAL_STATE
	total_reward=0;
	for t in range(T):
		action=policy[state];
		total_reward+=reward(state,action);
		state=next_state(state,action);
		if _print:
			display_board(state);
			print("cumulative reward: {}".format(total_reward));
			time.sleep(0.2)
	return total_reward;

def clear():
    os.system( 'clear' )

def display_board(state):
	clear();
	board_strings=[["A"," "," "," "],
				   [" ","B"," "," "],
				   [" "," "," "," "],
				   [" "," "," "," "]];
	player_idx,police_idx=si2ai(state);
	board_strings[player_idx // BOARD_WIDTH][player_idx%BOARD_WIDTH]="R";
	board_strings[police_idx // BOARD_WIDTH][police_idx%BOARD_WIDTH]="P";
	if player_idx==police_idx:
		board_strings[player_idx // BOARD_WIDTH][player_idx%BOARD_WIDTH]="✞";

	for l in board_strings:
		print("".join(l))
	return


def Q_learning(n_iterations):	
	print("START Q-learning")
	# initialize Q
	Q = np.zeros((N_STATES,N_ROBBER_ACTIONS));
	initial_state_value=np.zeros((n_iterations,N_ROBBER_ACTIONS));
	n_updates=np.ones((N_STATES,N_ROBBER_ACTIONS));
	##Q-learning
	state=np.random.randint(0,N_STATES);
	for t in range(0,n_iterations):
		action_idx=int(np.random.uniform(0, N_ROBBER_ACTIONS));
		#print(action_idx)
		new_state=next_state(state,action_idx);
		step_size=1/(n_updates[state,action_idx])**(2/3);
		#learning_rate=0.02;
		Q[state,action_idx]=(1-step_size)*Q[state,action_idx]+step_size*(reward(state,action_idx)+discount_factor*np.max(Q[new_state,:]));
		n_updates[state,action_idx]+=1;
		state=new_state;
		if t%ivs_step==0:
			print("training..("+str(100*t/n_iterations)+"% complete)",end="\r");
			#print(Q[INITIAL_STATE,:])
		initial_state_value[t]=np.max(Q[INITIAL_STATE,:]);
	policy=np.argmax(Q,1);
	print(n_updates)
	return policy,initial_state_value

def SARSA(n_iterations,epsilon):
	print("START SARSA")
	Q = np.ones((N_STATES,N_ROBBER_ACTIONS))*0;
	initial_state_value=np.zeros((n_iterations,N_ROBBER_ACTIONS));
	n_updates=np.ones((N_STATES,N_ROBBER_ACTIONS));
	state=np.random.randint(0,N_STATES);
	for t in range(n_iterations):
		if np.random.uniform(0,1)<epsilon:	
			action_idx=int(np.random.uniform(0, N_ROBBER_ACTIONS));
		else:
			policy=np.argmax(Q,1);
			action_idx=policy[state];
		step_size=1/(n_updates[state,action_idx])**(2/3);
		new_state=next_state(state,action_idx);
		Q[state,action_idx]=Q[state,action_idx]+step_size*(reward(state,action_idx)+discount_factor*np.max(Q[new_state,:])-Q[state,action_idx]);
		n_updates[state,action_idx]+=1;
		state=new_state;
		if t%ivs_step==0:
			print("training..("+str(100*t/n_iterations)+"% complete)",end="\r");
			#print(Q[INITIAL_STATE,:])
		initial_state_value[t]=np.max(Q[INITIAL_STATE,:]);
	policy=np.argmax(Q,1);
	return policy,initial_state_value






policy,initial_state_value=Q_learning(n_iterations);
#policy,initial_state_value=SARSA(n_iterations,0.1);

'''
epsilons=[0.01,0.05,0.1,0.2,0.6];
ivs=epsilons.copy();

c = ['#ddb464','#83192d','#451930','#211b29','#6d8a90']

for e_idx,eps in enumerate(epsilons):
	policy,ivs[e_idx]=SARSA(n_iterations,eps);
	plt.plot(ivs[e_idx],color=c[e_idx],label="epsilon = {}".format(eps));


handles, labels = plt.gca().get_legend_handles_labels()
by_label = OrderedDict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys())
plt.xlabel('Iteration')
plt.ylabel('initial state value')

plt.title("SARSA with different epsilon")


plt.show();

'''


plt.plot(initial_state_value);

plt.xlabel('Iteration')
plt.ylabel('initial state value')

plt.title("Q-learning")

plt.show();

T_trial=1000;

#print("\nReturn from policy over {} timesteps: {}".format(T_trial,try_policy(policy,T_trial,True)));



