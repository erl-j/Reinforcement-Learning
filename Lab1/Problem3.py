

##Define some constants
BOARD_WIDTH=4
BOARD_HEIGHT=4

STAY=0
UP=-BOARD_WIDTH
RIGHT=1
DOWN=BOARD_WIDTH
LEFT=-1

ACTIONS=[STAY,UP,RIGHT,DOWN,LEFT]

def rc2idx(rc):
	'''Converts row column position to index'''
	return rc[0] * BOARD_WIDTH + rc[1]


def idx2rc(idx):
	'''Converts index to row column position '''
	return np.array([idx // BOARD_WIDTH, idx % BOARD_WIDTH])


def rc_exists(rc):
	'''Check that a certain position exists.'''
	return -1 < rc[0] < BOARD_HEIGHT and -1 < rc[1] < BOARD_WIDTH


def iv2si(index_vector):
	'''Converts [player_idx,police_idx] (list-like) to state index (int)'''
	return index_vector[1]*BOARD_WIDTH*BOARD_HEIGHT+index_vector[0]


def move(current_pos,action):
	'''takes a position index and a move (int representing number of squares to move)
		and outputs the resulting position.
	'''
	try_pos=current_pos+action;
	column=current_pos % BOARD_WIDTH;
	if (not -1<try_pos<BOARD_HEIGHT*BOARD_WIDTH)\
		or (column==0 and action==-1)\
		or (column==BOARD_WIDTH-1 and action==1):
			return current_pos;
	else:
		return try_pos;



