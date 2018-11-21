import os
import numpy as np
import matplotlib.pyplot as plt
import time;

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
POLICE_ACTIONS = [STAY, UP, RIGHT, DOWN, LEFT]

N_ROBBER_ACTIONS = len(ROBBER_ACTIONS)
N_POLICE_ACTIONS = len(POLICE_ACTIONS)

N_STATES = N_TILES**2

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

def stpf(state,action_idx):
	player_idx,police_idx=si2ai(state);
	player_action=ROBBER_ACTIONS[action_idx];
	new_player_idx=move(player_idx,player_action);
	



