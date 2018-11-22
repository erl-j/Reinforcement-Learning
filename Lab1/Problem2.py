import os
import numpy as np
import matplotlib.pyplot as plt
import time;


discount_factor=0.99;

# GAME PROPRETIES
BOARD_WIDTH = 6
BOARD_HEIGHT = 3
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


BANK_IDXS=[0,5,12,17];

def rc2idx(rc):	
	'''Converts row column position to index'''
	return rc[0] * BOARD_WIDTH + rc[1]


def idx2rc(idx):
	'''Converts index to row column position '''
	return idx // BOARD_WIDTH, idx % BOARD_WIDTH


def rc_exists(rc):
	'''Check that a certain position exists.'''
	return -1 < rc[0] < BOARD_HEIGHT and -1 < rc[1] < BOARD_WIDTH


def ai2si(player_idx, police_idx):
	'''Converts agent indices to state index (int)'''
	return police_idx * N_TILES + player_idx


def si2ai(state_index):
	'''Takes state index and returns player idx and police idx'''
	return state_index % N_TILES, state_index // N_TILES

INITIAL_STATE=ai2si(0,8);

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

def isAbove(pos_idx, ref_idx ):
	pr,pc=idx2rc(pos_idx);
	rr,rc=idx2rc(ref_idx);
	if pr<rr:
		return True
	else:
		return False

def isBelow(pos_idx, ref_idx ):
	pr,pc=idx2rc(pos_idx);
	rr,rc=idx2rc(ref_idx);
	if pr>rr:
		return True
	else:
		return False


def isLeft(pos_idx,ref_idx):
	pr,pc=idx2rc(pos_idx);
	rr,rc=idx2rc(ref_idx);
	if pc<rc:
		return True;
	else:
		return False;

def isRight(pos_idx,ref_idx):
	pr,pc=idx2rc(pos_idx);
	rr,rc=idx2rc(ref_idx);
	if pc>rc:
		return True;
	else:
		return False;



def stpf(state,action_idx):
	tp=np.zeros(N_STATES);
	player_idx,police_idx=si2ai(state);
	if player_idx==police_idx:
		tp[INITIAL_STATE]=1;
	else:
		player_action=ROBBER_ACTIONS[action_idx];
		new_player_idx=move(player_idx,player_action);
		possible_police_actions=POLICE_ACTIONS.copy();
		if isAbove(player_idx,police_idx):
			possible_police_actions.remove(DOWN);
		elif isBelow(player_idx,police_idx):
			possible_police_actions.remove(UP);
		if isRight(player_idx,police_idx):
			possible_police_actions.remove(LEFT);
		elif isLeft(player_idx,police_idx):
			possible_police_actions.remove(RIGHT);
		n_possible_police_actions=len(possible_police_actions);
		for a in possible_police_actions:
			new_police_idx=move(police_idx,a);
			tp[ai2si(new_player_idx,new_police_idx)]=1/n_possible_police_actions;
	return tp;

def reward(state,action_idx):
	reward=0;
	player_idx,police_idx=si2ai(state)
	if player_idx==police_idx:
		reward=-50;
	elif player_idx in BANK_IDXS:
		reward=10;
	return reward;

# precompute transition probability matrix.
stps = np.zeros((N_STATES, N_STATES, N_ROBBER_ACTIONS))
for a_idx in range(N_ROBBER_ACTIONS):
	for state in range(0, N_STATES):
		stps[state, :, a_idx] = stpf(state, a_idx)

rewards = np.zeros((N_STATES, N_ROBBER_ACTIONS))

# precompute rewards
for state in range(0, N_STATES):
	for a_idx in range(N_ROBBER_ACTIONS):
		rewards[state, a_idx] = reward(state, a_idx)

def try_policy(policy, T,_print=False):
	p = 0
	m = 8
	reward=0;
	for t in range(0, T):
		if _print:
			time.sleep(0.2)
			display_board(ai2si(p,m))
			print("at t={}, total reward is {}".format(t,reward));

		if p == m:
			reward-=50;
			print("Died")
		elif p in BANK_IDXS:
			reward+=10;

		state = ai2si(p,m)
		a_idx = policy[state]
		p=move(p,ROBBER_ACTIONS[a_idx]);

		possible_police_actions=POLICE_ACTIONS.copy();
		if isAbove(p,m):
			possible_police_actions.remove(DOWN);
		elif isBelow(p,m):
			possible_police_actions.remove(UP);
		if isRight(p,m):
			possible_police_actions.remove(LEFT);
		elif isLeft(p,m):
			possible_police_actions.remove(RIGHT);
		n_possible_police_actions=len(possible_police_actions);
		police_move_idx=np.random.randint(0,n_possible_police_actions);
		m=move(m,possible_police_actions[police_move_idx]);
	return

def clear():
    os.system( 'clear' )

def display_policy(policy, m):

	policy_wrt_m = policy[m * N_TILES:m * N_TILES + N_TILES]
	action_icons = ["⧖", "⥣", "⥤", "⥥", "⥢"]
	minotaur_icon = "𓃾"
	policy_str = [action_icons[a] for a in policy_wrt_m]
	policy_str[m] = minotaur_icon
	for r in range(0, 5):
		print('  '.join(policy_str[r * BOARD_WIDTH:r * BOARD_WIDTH + BOARD_WIDTH]))
	return


def display_board(state):
	clear();
	board_strings=[["b"," "," "," "," ","b"],
				   [" "," "," "," "," "," "],
				   ["b"," "," "," "," ","b"]]
				
	player_idx,police_idx=si2ai(state);
	board_strings[player_idx // BOARD_WIDTH][player_idx%BOARD_WIDTH]="R";
	board_strings[police_idx // BOARD_WIDTH][police_idx%BOARD_WIDTH]="P";
	if player_idx==police_idx:
		board_strings[player_idx // BOARD_WIDTH][player_idx%BOARD_WIDTH]="✞";

	for l in board_strings:
		print("".join(l))
	return

#Returns a N_STATES*T policy matrix
def backward_induction(stps,rewards,T):
	# Bellman induction
	policy=np.zeros((N_STATES,T),dtype=np.int8);
	u_star = np.amax(rewards, 1)
	u_a = np.argmax(rewards, 1)
	for t in range(T-2, 0, -1):
		u = np.zeros((N_STATES, N_ROBBER_ACTIONS))
		for s_idx in range(0, N_STATES):
			for a_idx in range(0, N_ROBBER_ACTIONS):
				u[s_idx, a_idx] += discount_factor*np.sum(stps[s_idx, :, a_idx]@u_star) + rewards[s_idx, a_idx]
		u_star = np.amax(u, 1)
		u_a = np.argmax(u, 1)
	policy=u_a;
		# print(u_star)
	return policy





policy=backward_induction(stps,rewards,1000);

try_policy(policy,10,True)

for police_idx in range(0,N_TILES):
	display_policy(policy,police_idx)








