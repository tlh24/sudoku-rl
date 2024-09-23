import math
import numpy as np
import numpy as np
import torch
from enum import Enum
from sudoku_gen import Sudoku
import matplotlib.pyplot as plt
from constants import *
from type_file import Types, Axes, Action, getActionName
import pdb

# encode the board as a graph that can be passed to sparse attention.
# keep a list of nodes and edges

class Node:
	def __init__(self, typ):
		self.typ = typ
		self.loc = -1 # for edges
		self.refcnt = 0
		self.kids = []
		self.parents = []
		self.axval = np.zeros(7)

	def addChild(self, node):
		# kid is type `Node
		self.kids.append(node)
		node.parents.append(self)

	def setAxVal(self, ax, val):
		self.axval[ax.value - Axes.N_AX.value] = val

	def print(self, indent):
		if self.refcnt < 1:
			self.refcnt = self.refcnt + 1
			print(indent, self.typ.name, self.axval[0])
			indent2 = indent + "  "
			for k in self.kids:
				k.print(indent2)

	def printGexf(self, fil):
		if self.refcnt < 1:
			self.refcnt = self.refcnt + 1
			print(f'<node id="{self.loc}" label="{self.typ} val:{self.axval[0]}">',file=fil)
			print('</node>',file=fil)
			for k in self.kids: 
				k.printGexf(fil)
			
	def resetRefcnt(self): 
		if self.refcnt > 0: # graph can contain loops! 
			self.refcnt = 0
			for k in self.kids: 
				k.resetRefcnt()
		
	def clearLoc(self): 
		if self.loc >= 0: 
			self.loc = -1
			for k in self.kids: 
				k.clearLoc()
	
	def setLoc(self, i):
		if self.loc < 0: 
			self.loc = i
			i = i+1
			for k in self.kids: 
				i = k.setLoc(i)
		return i
		
	def flatten(self, node_list): 
		if self.refcnt < 1: 
			self.refcnt = self.refcnt + 1
			node_list.append(self)
			for k in self.kids: 
				node_list = k.flatten(node_list)
		return node_list
		
	def gatherEdges(self, edges, a2a_set): 
		if self.refcnt <= 0: 
			self.refcnt = self.refcnt+1
			for k in self.kids:
				edges.append((self.loc, k.loc))
				a2a_set.discard(k.loc)
				edges,a2a_set = k.gatherEdges(edges,a2a_set)
		return edges,a2a_set
		
def sudokuActionNodes(action_type, action_value): 
	
	def makeAction(ax,v):
		na = Node(Types.MOVE_ACTION)
		na.setAxVal( ax, v )
		return na
		
	# Uses matrix coordinates: (0,0) is at the top left.
	# x is down/up, y is right/left
	match action_type: 
		case Action.LEFT.value:
			na = makeAction(Axes.Y_AX, -1)
		case Action.RIGHT.value:
			na = makeAction(Axes.Y_AX, 1)
		case Action.UP.value:
			na = makeAction(Axes.X_AX, -1)
		case Action.DOWN.value:
			na = makeAction(Axes.X_AX, 1)
		
		case Action.SET_GUESS.value: 
			na = Node(Types.GUESS_ACTION)
			na.setAxVal( Axes.G_AX, action_value )
		case Action.UNSET_GUESS.value:
			na = Node(Types.GUESS_ACTION)
			na.setAxVal( Axes.G_AX, 0 )
			
		case _ : 
			print(action_type)
			assert(False)
	return na

def puzzleToNodes(puzzl_mat, guess_mat=None, curs_pos=None):
	'''
	Very simple version: encode the board as a set of box nodes
	and set / set2 nodes.
	'''
	if guess_mat is None:
		guess_mat = np.zeros((9,9), dtype=int)
	if curs_pos is None:
		curs_pos = np.ones((2,), dtype=int) * -1

	nodes = []
	board_nodes = [[] for _ in range(SuN)]

	posOffset = (SuN - 1) / 2.0

	for x in range(SuN): # x = row
		for y in range(SuN): # y = column
			b = (x // SuH)*SuH + (y // SuH)
			v = puzzl_mat[x,y]
			nb = Node(Types.BOX)
			nb.setAxVal( Axes.N_AX, v )
			g = guess_mat[x,y]
			nb.setAxVal( Axes.G_AX, g )

			# think of these as named attributes, var.x, var.y etc
			# the original encoding is var.pos[0], var.pos[1], var.pos[2]
			# can do a bit of both by encoding the axes with integers
			nb.setAxVal( Axes.X_AX, x - posOffset )
			nb.setAxVal( Axes.Y_AX, y - posOffset )
			nb.setAxVal( Axes.B_AX, b - posOffset )

			highlight = 0
			if x == curs_pos[0] and y == curs_pos[1]:
				highlight = 2
			nb.setAxVal( Axes.H_AX, highlight )

			board_nodes[x].append(nb)

	xsets = Node(Types.SET2)
	xsets.setAxVal( Axes.N_AX, 2)
	nodes.append(xsets)
	# nboard.addChild(xsets)
	for x in range(SuN):
		nb = Node(Types.SET)
		nb.setAxVal( Axes.N_AX, 1)
		nb.setAxVal( Axes.X_AX, x - posOffset )
		highlight = 0
		if x == curs_pos[0] :
			highlight = -1
		nb.setAxVal( Axes.H_AX, highlight )
		for y in range(SuN):
			nb.addChild( board_nodes[x][y] )
		xsets.addChild(nb)

	ysets = Node(Types.SET2)
	ysets.setAxVal( Axes.N_AX, 4)
	nodes.append(ysets)
	# nboard.addChild(ysets)
	for y in range(SuN):
		nb = Node(Types.SET)
		nb.setAxVal( Axes.N_AX, 3)
		nb.setAxVal( Axes.Y_AX, y - posOffset )
		highlight = 0
		if y == curs_pos[1] :
			highlight = -1
		nb.setAxVal( Axes.H_AX, highlight )
		for x in range(SuN):
			nb.addChild( board_nodes[x][y] )
		ysets.addChild(nb)

	bsets = Node(Types.SET2)
	bsets.setAxVal( Axes.N_AX, 6)
	nodes.append(bsets)
	# nboard.addChild(bsets)
	for b in range(SuN):
		nb = Node(Types.SET)
		nb.setAxVal( Axes.N_AX, 5)
		nb.setAxVal( Axes.B_AX, b - posOffset )
		highlight = 0
		bc = (curs_pos[0] // SuH)*SuH + (curs_pos[1] // SuH)
		if b == bc :
			highlight = -1
		nb.setAxVal( Axes.H_AX, highlight )
		for x in range(SuN): # x = row
			for y in range(SuN): # y = column
				bb = (x // SuH)*SuH + (y // SuH)
				if b == bb:
					nb.addChild( board_nodes[x][y] )
		bsets.addChild(nb)

	for n in nodes:
		n.clearLoc()
	i = 0
	for n in nodes:
		i = n.setLoc(i)

	board_loc = np.zeros((SuN,SuN),dtype=int)
	for x in range(SuN): # x = row
		for y in range(SuN): # y = column
			board_loc[x,y] = board_nodes[x][y].loc

	return nodes, board_nodes, board_loc


def sudokuToNodes(puzzl_mat, guess_mat, curs_pos, action_type:int, action_value:int, reward:float, many_reward=False):
	nodes = []
	posOffset = (SuN - 1) / 2.0

	na = sudokuActionNodes(action_type, action_value)
	nodes.append(na) # put at beginning for better visibility
	
	# make the cursor
	ncursor = Node(Types.CURSOR)
	ncursor.setAxVal( Axes.X_AX, curs_pos[0] - posOffset )
	ncursor.setAxVal( Axes.Y_AX, curs_pos[1] - posOffset )
	bc = (curs_pos[0] // SuH)*SuH + (curs_pos[1] // SuH)
	ncursor.setAxVal( Axes.B_AX, bc - posOffset )
	# if the cursor loc is empty, indicate that.
	if puzzl_mat[curs_pos[0],curs_pos[1]] == 0 and guess_mat[curs_pos[0],curs_pos[1]] == 0: 
		ncursor.setAxVal( Axes.H_AX, 1.0 )
	else:
		ncursor.setAxVal( Axes.H_AX, 0.0 )
	nodes.append(ncursor)
	
	# reward token (used for one-step reward prediction)
	nreward = Node(Types.REWARD)
	nreward.setAxVal( Axes.R_AX, reward*5 )
	nodes.append(nreward)
	
	nodes_b, board_nodes, _ = puzzleToNodes(puzzl_mat, guess_mat, curs_pos)
	nodes.extend(nodes_b)

	# add in at the end.
	if many_reward:
		axes = [Axes.X_AX, Axes.Y_AX]
		dirs = [-1, 1]
		for ax in axes:
			for d in dirs:
				nr = Node(Types.REWARD)
				nr.setAxVal( Axes.R_AX, 0.0 ) # fill this in later.
				nr.setAxVal( ax, d )
				nodes.append(nr)
	
	# set the node indexes.
	# some nodes are in the top-level list; others are just children.
	for n in nodes: 
		n.clearLoc()
	i = 0
	for n in nodes: 
		i = n.setLoc(i)
	if i > token_cnt:
		print(f'expect {token_cnt}, encoded {i} tokens')
		pdb.set_trace()
	
	board_loc = np.zeros((SuN,SuN),dtype=int)
	for x in range(SuN): # x = row
		for y in range(SuN): # y = column
			board_loc[x,y] = board_nodes[x][y].loc
	cursor_loc = ncursor.loc
	
	return nodes, nreward.loc, (board_loc,cursor_loc)
	
def nodesToCoo(nodes): 
	# coo is [dst, src] -- see l1attnSparse
	edges = [] # edges from kids to parents
	# to get from parents to kids, switch dst and src. 
	a2a_set = set() # nodes that have no set relation, for a2a attention
					# aka top-level nodes or objects.
	for n in nodes: 
		n.resetRefcnt()
		a2a_set.add(n)
	for n in nodes: 
		edges,a2a_set = n.gatherEdges(edges,a2a_set)
	a2a = []
	for n in a2a_set: 
		a2a.append(n.loc)
	a2a.sort() # in-place
	return torch.tensor(edges), torch.tensor(a2a)
	
def encodeNodes(nodes): 
	# returns a matrix encoding of the nodes + coo vector
	# redo the loc, jic
	for n in nodes: 
		n.clearLoc()
	i = 0
	for n in nodes: 
		i = n.setLoc(i)
	# flatten the tree-list
	for n in nodes: 
		n.resetRefcnt()
	nodes_flat = []
	for n in nodes: 
		nodes_flat = n.flatten(nodes_flat)
	count = len(nodes_flat)
	assert(i == count)
	benc = np.zeros((count, 32))
	for n in nodes_flat: 
		i = n.loc # must be consistent with edges for coo
		benc[i, n.typ.value] = 1.0 # categorical
		# ii = 31
		# if n.typ.value >= Axes.N_AX.value and n.typ.value <= Axes.G_AX.value:
		# 	ii = (20 - n.typ.value) + 31
		benc[i, Axes.N_AX.value:32] = n.axval
		# add in categorical encoding of value
		def encode(x):
			if v >= 0.6 and v <= 9.4:
				benc[i,11:20] = 0.0
				vi = round(v)
				benc[i,10+vi] = 1.0
					
		ntv = n.typ.value
		if ntv == Types.BOX.value:
			v = n.axval[0]
			encode( v )
			v = n.axval[Axes.G_AX.value - Axes.N_AX.value]
			encode( v )
			benc[i, Axes.N_AX] = n.axval[0]/10.0 # FIXME primarily categorical!
		if ntv == Types.GUESS_ACTION.value:
			v = n.axval[Axes.G_AX.value - Axes.N_AX.value]
			encode( v )
		
	coo,a2a = nodesToCoo(nodes)
	return benc, coo, a2a
	
def encodeActionNodes(action_type, action_value): 
	# action nodes are at the beginning of the board encoding, 
	# easy to replace. 
	na = sudokuActionNodes(action_type, action_value)
	aenc,_,_ = encodeNodes([na])
	return aenc
	
def decodeBoard(benc, board_loc, argmax=True):
	puzzle = np.zeros((SuN, SuN), dtype = int)
	for x in range(SuN): # x = row
		for y in range(SuN): # y = column
			if argmax:
				if np.sum(benc[board_loc[x,y], 11:20] > 0.5) > 0.5:
					puzzle[x,y] = np.argmax(benc[board_loc[x,y], 11:20]) + 1
			else:
				puzzle[x,y] = round(benc[board_loc[x,y], Axes.N_AX.value].item() )
	return puzzle

def decodeNodes(indent, benc, locs):
	# prints the output; returns nothing.
	posOffset = (SuN - 1) / 2.0
	board_loc,cursor_loc = locs
	puzzle = np.zeros((SuN, SuN), dtype = int)
	guess_mat = np.zeros((SuN, SuN), dtype = int)
	xo = Axes.X_AX.value
	yo = Axes.Y_AX.value
	go = Axes.G_AX.value
	for x in range(SuN): # x = row
		for y in range(SuN): # y = column
			puzzle[x,y] = round(benc[board_loc[x,y], Axes.N_AX.value].item() )
			guess_mat[x,y] = round(benc[board_loc[x,y], go].item() )
	cursor_pos = [round(benc[cursor_loc, xo].item() + posOffset), \
		round(benc[cursor_loc, yo].item() + posOffset ) ]
	
	su = Sudoku(SuN,SuN)
	su.printSudoku(indent, puzzle.numpy(), guess_mat.numpy(), cursor_pos)
	can_guess = round(benc[cursor_loc, Axes.H_AX.value].item())
	print(f" can guess:{can_guess}")
	# print("guess_mat", guess_mat)
	print(f"{indent}cursor_pos", cursor_pos)
	
def outputGexf(nodes): 
	for n in nodes: 
		n.resetRefcnt()
	header = '''<?xml version="1.0" encoding="UTF-8"?>
<gexf xmlns="http://gexf.net/1.3" version="1.3">
<graph mode="static" defaultedgetype="directed">
<attributes class="node">
<attribute id="0" title="progt" type="string"/>
</attributes>
<attributes class="edge">
<attribute id="0" title="typ" type="string"/>
<attribute id="1" title="cost" type="int"/>
</attributes>
<nodes> '''
	fil = open('sudoku.gexf', 'w')
	print(header,file=fil)
	for n in nodes: 
		n.printGexf(fil)
	print('</nodes>',file=fil)
	print('<edges>',file=fil)
	# make a flat list of all nodes to get all edges. 
	for n in nodes: 
		n.resetRefcnt()
	flat_nodes = []
	for n in nodes: 
		flat_nodes = n.flatten(flat_nodes)
	for n in flat_nodes: 
		for k in n.kids: 
			print(f'<edge source="{n.loc}" target="{k.loc}">',file=fil)
			print(f'<attvalues>',file=fil)
			print(f'<attvalue for="0" values="first"/>',file=fil)
			print(f'</attvalues>',file=fil)
			print('</edge>',file=fil)
			# # add in grandkids
			# for kk in k.kids: 
			# 	print(f'<edge source="{n.loc}" target="{kk.loc}">',file=fil)
			# 	print(f'<attvalues>',file=fil)
			# 	print(f'<attvalue for="0" value="second"/>',file=fil)
			# 	print(f'</attvalues>',file=fil)
			# 	print('</edge>',file=fil)
	footer = '''
</edges>
</graph>
</gexf>'''
	print(footer,file=fil)
	fil.close()
	
	
if __name__ == "__main__":
	puzzle = np.arange(SuN*SuN) / 10
	puzzle = np.reshape(puzzle,(SuN,SuN))
	curs_pos = [0,0]
	guess_mat = np.zeros((SuN,SuN))
	nodes, reward_loc, locs = sudokuToNodes(puzzle, guess_mat, curs_pos, Action.LEFT.value, 0, 0.0)
	
	for n in nodes: 
		n.resetRefcnt()
	for n in nodes: 
		n.print("")
	
	plot_rows = 1
	plot_cols = 2
	figsize = (16, 8)
	fig, axs = plt.subplots(plot_rows, plot_cols, figsize=figsize)
	im = [0,0]
	
	benc,coo,a2a = encodeNodes(nodes)
	print('benc shape:',benc.shape,'coo shape',coo.shape,"a2a shape",a2a.shape)
	
	im[0] = axs[0].imshow(benc.T.numpy())
	plt.colorbar(im[0], ax=axs[0])
	axs[0].set_title('board encoding')
	
	im[1] = axs[1].plot(coo[:,0].numpy(), coo[:,1].numpy(), 'o')
	axs[1].set_title('coo vector (child -> parent)')
	axs[1].set_xlabel('dst')
	axs[1].set_ylabel('src')
	
	plt.show()
		
	outputGexf(nodes)
	# seems to be working..

