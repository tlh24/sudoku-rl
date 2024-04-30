import math
import numpy as np
import torch
from enum import Enum
from sudoku_gen import Sudoku
import matplotlib.pyplot as plt
from constants import SuN, SuH, SuK, world_dim
from type_file import Types, Axes, Action 
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
		
	def addChild(self, node): 
		# kid is type `Node
		self.kids.append(node)
		node.parents.append(self)
		
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
		
		
def sudokuToNodes(puzzle, guess_mat, curs_pos, action_type:int, action_value:int): 
	nodes = []
	posOffset = (SuN - 1) / 2.0
	board_nodes = [[] for _ in range(SuN)]
	
	full_board = False
	if full_board: 
		for x in range(SuN): # x = row
			for y in range(SuN): # y = column
				b = (y // SuH)*SuH + (x // SuH)
				v = puzzle[x,y]
				nb = Node(Types.BOX, v)
				g = guess_mat[x,y]
				nb.addChild( Node(Axes.G_AX, g - posOffset) )
				
				# think of these as named attributes, var.x, var.y etc
				# the original encoding is var.pos[0], var.pos[1], var.pos[2]
				# can do a bit of both by encoding the axes with integers
				nb.addChild( Node(Axes.X_AX, x - posOffset) )
				nb.addChild( Node(Axes.Y_AX, y - posOffset) )
				nb.addChild( Node(Axes.B_AX, b - posOffset) )
				
				highlight = 0
				if x == curs_pos[0] and y == curs_pos[1]: 
					highlight = 1
				nb.addChild( Node(Axes.H_AX, highlight))
				
				board_nodes[x].append(nb)
				nodes.append(nb)
		
		# make the sets
		nboard = Node(Types.SET, 2) # node of the whole board
		
		xsets = Node(Types.SET, 1.1)
		for x in range(SuN): 
			nb = Node(Types.SET, 0)
			nb.addChild( Node(Axes.X_AX, x - posOffset) )
			for y in range(SuN): 
				nb.addChild( board_nodes[x][y] )
			xsets.addChild(nb)
			nodes.append(nb)
		nboard.addChild(xsets)
		nodes.append(xsets)
		
		ysets = Node(Types.SET, 1.2)
		for y in range(SuN): 
			nb = Node(Types.SET, 0)
			nb.addChild( Node(Axes.Y_AX, y - posOffset) )
			for x in range(SuN): 
				nb.addChild( board_nodes[x][y] )
			ysets.addChild(nb)
			nodes.append(nb)
		nboard.addChild(ysets)
		nodes.append(ysets)
			
		bsets = Node(Types.SET, 1.3)
		for b in range(SuN): 
			nb = Node(Types.SET, 0)
			nb.addChild( Node(Axes.B_AX, b - posOffset) )
			for y in range(SuN): # y = row
				for x in range(SuN): # x = column
					bb = (y // SuH)*SuH + (x // SuH)
					if b == bb: 
						nb.addChild( board_nodes[x][y] )
			bsets.addChild(nb)
			nodes.append(nb)
		nboard.addChild(bsets)
		nodes.append(bsets)
		nodes.append(nboard)
	
	# make the cursor
	ncursor = Node(Types.CURSOR, 0)
	ncursor.addChild( Node(Axes.X_AX, curs_pos[0] - posOffset) )
	ncursor.addChild( Node(Axes.Y_AX, curs_pos[1] - posOffset) )
	nodes.append(ncursor)
	
	def makeAction(ax,v):
		na = Node(Types.MOVE_ACTION, 0)
		na.addChild( Node(ax, v) )
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
		case Action.UNSET_GUESS.value:
			na = Node(Types.GUESS_ACTION, 0)
			
		case Action.SET_NOTE.value: 
			na = Node(Types.NOTE_ACTION, action_value)
		case Action.UNSET_NOTE.value:
			na = Node(Types.NOTE_ACTION, 0)
	
	if full_board: 
		na.addChild(nboard) # should this be the other way around?
	na.addChild(ncursor)
	nodes.insert(0,na) # put at beginning for better visibility
	
	# set the node indexes.
	# some nodes are in the top-level list; others are just children.
	for n in nodes: 
		n.clearLoc()
	i = 0
	for n in nodes: 
		i = n.setLoc(i)
	# print("total number of nodes:", i)
	
	return nodes # action node & board nodes.
	
def nodesToCoo(nodes): 
	# coo is [dst, src] -- see l1attnSparse
	edges = [] # edges from kids to parents
	# to get from parents to kids, switch dst and src. 
	for n in nodes: 
		n.resetRefcnt()
	for n in nodes: 
		if n.refcnt <= 0: 
			n.refcnt = n.refcnt+1
			for k in n.kids:
				edges.append((n.loc, k.loc))
	return torch.tensor(edges)
	
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
	benc = np.zeros((count, world_dim), dtype=np.float32)
	for n in nodes_flat: 
		i = n.loc # muct be consistent with edges for coo
		benc[i, n.typ.value] = 1.0 # categorical
		#if len(n.parents) > 0: 
		#	benc[i, n.parents[0].typ.value] = 1.0 # inheritance?
		benc[i, 20] = n.value
		
	coo = nodesToCoo(nodes)
	return torch.tensor(benc), coo
		
	
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
	for n in nodes: 
		for k in n.kids: 
			print(f'<edge source="{n.loc}" target="{k.loc}">',file=fil)
			print('</edge>',file=fil)
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
	nodes = sudokuToNodes(puzzle, guess_mat, curs_pos, Action.LEFT.value, 0)
	
	for n in nodes: 
		n.resetRefcnt()
	for n in nodes: 
		n.print("")
	
	plot_rows = 1
	plot_cols = 2
	figsize = (16, 8)
	fig, axs = plt.subplots(plot_rows, plot_cols, figsize=figsize)
	im = [0,0]
	
	benc,coo = encodeNodes(nodes)
	print('benc shape:',benc.shape,'coo shape',coo.shape)
	
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

