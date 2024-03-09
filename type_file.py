from enum import Enum 

class Types(Enum): 
	CURSOR = 1
	POSITION = 2 # value is the axis
	LEAF = 3 # bare value
	BOX = 4
	ACTION = 5
	
class Axes(float, Enum): 
	N_AX = 0 #null, nop
	X_AX = 1
	Y_AX = 2
	B_AX = 3 # block
	H_AX = 4 # highlight

class Action(Enum):
	UP = 0
	RIGHT = 1
	DOWN = 2
	LEFT = 3
	SET_GUESS = 4
	UNSET_GUESS = 5
	SET_NOTE = 6
	UNSET_NOTE = 7
	NOP = 8