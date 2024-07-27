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
	def __init__(self, typ, val):
		self.typ = typ
		# Payload. -1 for null, also holds Axes values.
		self.value = float(val)
		self.loc = -1 # for edges
		self.refcnt = 0
		self.kids = []
		self.parents = []
		self.axval = np.zeros(6)

	def addChild(self, node):
		# kid is type `Node
		self.kids.append(node)
		node.parents.append(self)

	def setAxVal(self, ax, val):
		self.axval[ax.value - Axes.N_AX.value] = val

	def print(self, indent):
		if self.refcnt < 1:
			self.refcnt = self.refcnt + 1
			print(indent, self.typ.name, self.value)
			indent2 = indent + "  "
			for k in self.kids:
				k.print(indent2)

	def printGexf(self, fil):
		if self.refcnt < 1:
			self.refcnt = self.refcnt + 1


# encode the board as a graph that can be passed to sparse attention.
# keep a list of nodes and edges

class Node:
	def __init__(self, typ, val):
		self.typ = typ
		# Payload. -1 for null, also holds Axes values.
		self.value = float(val)
		self.loc = -1 # for edges
		self.refcnt = 0
		self.kids = []
		self.parents = []
		self.axval = np.zeros(6)

	def addChild(self, node):
		# kid is type `Node
		self.kids.append(node)
		node.parents.append(self)

	def setAxVal(self, ax, val):
		self.axval[ax.value - Axes.N_AX.value] = val

	def print(self, indent):
		if self.refcnt < 1:
			self.refcnt = self.refcnt + 1
			print(indent, self.typ.name, self.value)
			indent2 = indent + "  "
			for k in self.kids:
				k.print(indent2)

	def printGexf(self, fil):
		if self.refcnt < 1:
			self.refcnt = self.refcnt + 1
			print(f'<node id="{self.loc}" label="{self.typ} val:{self.value}">',file=fil)
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
		na = Node(Types.MOVE_ACTION, 0)
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
			na = Node(Types.GUESS_ACTION, action_value)
			# na.setAxVal( Axes.N_AX, action_value ) FIXME! turn this on
			# na.addChild( Node(Types.GUESS, action_value) ) # dummy.
		case Action.UNSET_GUESS.value:
			na = Node(Types.GUESS_ACTION, 0)
			na.setAxVal( Axes.N_AX, 0 )
			# na.addChild( Node(Types.GUESS, -2) ) # dummy.
			
		case Action.SET_NOTE.value: 
			na = Node(Types.NOTE_ACTION, action_value)
		case Action.UNSET_NOTE.value:
			na = Node(Types.NOTE_ACTION, 0)
			
		case _ : 
			print(action_type)
			assert(False)
	return na
		
def sudokuToNodes(puzzl_mat, guess_mat, curs_pos, action_type:int, action_value:int, reward:float): 
	nodes = []
	posOffset = (SuN - 1) / 2.0
	
	# make the cursor
	ncursor = Node(Types.CURSOR, 0)
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
	
	# reward token (used for reward prediction)
	nreward = Node(Types.REWARD, reward*5)
	nreward.setAxVal( Axes.N_AX, reward*5 )
	nodes.append(nreward)
	
	full_board = True
	board_nodes = [[] for _ in range(SuN)]
	if full_board: 
		for x in range(SuN): # x = row
			for y in range(SuN): # y = column
				b = (x // SuH)*SuH + (y // SuH)
				v = puzzl_mat[x,y]
				nb = Node(Types.BOX, v)
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
		
		# make the sets
		# nboard = Node(Types.SET, 2) # node of the whole board
		
		xsets = Node(Types.SET2, 1.25)
		nodes.append(xsets)
		# nboard.addChild(xsets)
		for x in range(SuN): 
			nb = Node(Types.SET, 0.25)
			nb.setAxVal( Axes.X_AX, x - posOffset )
			highlight = 0
			if x == curs_pos[0] :
				highlight = -1 
			nb.setAxVal( Axes.H_AX, highlight )
			for y in range(SuN): 
				nb.addChild( board_nodes[x][y] )
			xsets.addChild(nb)
		
		ysets = Node(Types.SET2, 1.5)
		nodes.append(ysets)
		# nboard.addChild(ysets)
		for y in range(SuN): 
			nb = Node(Types.SET, 0.5)
			nb.setAxVal( Axes.Y_AX, y - posOffset )
			highlight = 0
			if y == curs_pos[1] :
				highlight = -1
			nb.setAxVal( Axes.H_AX, highlight )
			for x in range(SuN): 
				nb.addChild( board_nodes[x][y] )
			ysets.addChild(nb)
			
		bsets = Node(Types.SET2, 1.75)
		nodes.append(bsets)
		# nboard.addChild(bsets)
		for b in range(SuN): 
			nb = Node(Types.SET, 0.75)
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
		
		# nodes.append(nboard)
	
	na = sudokuActionNodes(action_type, action_value)
	
	# do we need these relations?
	# they are basically independent tokens, whose relation to the each other must  to be learned..
	# if full_board:
	# 	na.addChild(nboard) # should this be the other way around?
	# 	nreward.addChild(nboard)
	# na.addChild(ncursor)
	# na.addChild(nreward) # action obviously affects reward
	# ncursor.addChild(nreward)
	nodes.insert(0,na) # put at beginning for better visibility
	
	# set the node indexes.
	# some nodes are in the top-level list; others are just children.
	for n in nodes: 
		n.clearLoc()
	i = 0
	for n in nodes: 
		i = n.setLoc(i)
	assert(i == token_cnt)
	
	board_loc = torch.zeros((SuN,SuN),dtype=int)
	if full_board:
		for x in range(SuN): # x = row
			for y in range(SuN): # y = column
				board_loc[x,y] = board_nodes[x][y].loc
	cursor_loc = ncursor.loc
	
	return nodes, nreward.loc, (board_loc,cursor_loc)
	
def sudoku1DToNodes(puzzle, guess_mat, curs_pos, action_type:int, action_value:int, reward:float):
	# really boring 1D sudoku!  
	# to teach our angent what a set is.. 
	# ax can be Axes.X_AX, Y_AX, B_AX
	nodes = []
	posOffset = (SuN - 1) / 2.0
	# d = np.random.randint(0,3)
	# if d == 0:
	# 	ax = Axes.X_AX
	# elif d == 1:
	# 	ax = Axes.Y_AX
	# else:
	# 	ax = Axes.B_AX
	ax = Axes.X_AX
	
	ncursor = Node(Types.CURSOR, 0)
	ncursor.setAxVal( ax, curs_pos - posOffset )
	nodes.append(ncursor)
	
	# reward token (used for reward prediction)
	nreward = Node(Types.REWARD, reward*5)
	nreward.setAxVal( Axes.N_AX, reward*5 )
	nodes.append(nreward)
	
	board_nodes = []
	for x in range(SuN):
		v = puzzle[x]
		nb = Node(Types.BOX, v)
		nb.setAxVal( Axes.N_AX, v )
		g = guess_mat[x]
		nb.setAxVal( Axes.G_AX, g )
		
		nb.setAxVal( ax, x - posOffset )
		highlight = 2 
		nb.setAxVal( Axes.H_AX, highlight )
		
		board_nodes.append(nb)
		
	nset = Node(Types.SET, 0.25)
	nset.setAxVal( Axes.H_AX, -1 )
	nodes.append(nset)
	for x in range(SuN): 
		nset.addChild( board_nodes[x] )
		
	na = sudokuActionNodes(action_type, action_value)
	nodes.insert(0,na)
	
	for n in nodes: 
		n.clearLoc()
	i = 0
	for n in nodes: 
		i = n.setLoc(i)
	if i != token_cnt:
		pdb.set_trace()
		
	board_loc = torch.zeros((SuN,),dtype=int)
	for x in range(SuN): 
		board_loc[x] = board_nodes[x].loc
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
	benc = np.zeros((count, world_dim))
	for n in nodes_flat: 
		i = n.loc # must be consistent with edges for coo
		benc[i, n.typ.value] = 1.0 # categorical
		# ii = 31
		# if n.typ.value >= Axes.N_AX.value and n.typ.value <= Axes.G_AX.value:
		# 	ii = (20 - n.typ.value) + 31
		benc[i, 26:32] = n.axval
		# add in categorical encoding of value
		ntv = n.typ.value
		if ntv == Types.BOX.value or ntv == Types.GUESS_ACTION.value:
			if n.value >= 0.6 and n.value <= 9.4:
				vi = round(n.value)
				benc[i,10+vi] = 1.0
		# encode the guess in the same way - just different vector encoding.
		# should be OK from a game dynamics perspective; illegal guesses aren't added.
		guess_val = n.axval[Axes.G_AX.value - Axes.N_AX.value]
		if guess_val > 0.5: 
			vi = round(guess_val)
			benc[i,10+vi] = 1.0
		
	coo,a2a = nodesToCoo(nodes)
	return torch.tensor(benc, dtype=g_dtype), coo, a2a
	
def encodeActionNodes(action_type, action_value): 
	# action nodes are at the beginning of the board encoding, 
	# easy to replace. 
	na = sudokuActionNodes(action_type, action_value)
	aenc,_,_ = encodeNodes([na])
	return aenc
	
def decodeNodes(indent, benc, locs):
	# prints the output; returns nothing.
	posOffset = (SuN - 1) / 2.0
	board_loc,cursor_loc = locs
	puzzle = torch.zeros((SuN, SuN), dtype = int)
	guess_mat = torch.zeros((SuN, SuN), dtype = int)
	xo = 26 + Axes.X_AX.value - Axes.N_AX.value
	yo = 26 + Axes.Y_AX.value - Axes.N_AX.value
	go = 26 + Axes.G_AX.value - Axes.N_AX.value
	for x in range(SuN): # x = row
		for y in range(SuN): # y = column
			puzzle[x,y] = round(benc[board_loc[x,y], 26].item() )
			guess_mat[x,y] = round(benc[board_loc[x,y], go].item() )
	cursor_pos = [round(benc[cursor_loc, xo].item() + posOffset), \
		round(benc[cursor_loc, yo].item() + posOffset ) ]
	
	su = Sudoku(SuN,SuN)
	su.printSudoku(indent, puzzle.numpy(), guess_mat.numpy(), cursor_pos)
	can_guess = round(benc[cursor_loc, 26 + Axes.H_AX.value - Axes.N_AX.value].item())
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

