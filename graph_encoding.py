import math
import numpy as np
import torch
from enum import Enum
from sudoku_gen import Sudoku
import matplotlib.pyplot as plt
from constants import SuN, SuH, SuK
import pdb

class Types(Enum): 
	CURSOR = 1
	POSITION = 2 # value is the axis
	LEAF = 3 # bare value
	BOX = 4
	ACTION = 5
	
	
class Node: 
	def __init__(self, typ, val):
		self.typ = typ
		self.value = float(val) # payload
		self.loc = 0
		self.kids = []
		self.parents = []
		
	def add_child(self, node): 
		# kid is type `Node
		self.kids.append(node)
		node.parents.append(self)
		
	def print(self, indent): 
		print(indent, self.typ.name, self.value)
		indent2 = indent + "  "
		for k in self.kids: 
			k.print(indent2)
			
	def count(self): 
		n = 1
		for k in self.kids: 
			n = n + k.count()
		return n

# seems like we need a bidirectional graph -- parents know about their children, and children about their parents, with some sort of asymmetric edges. 
# can allow for both undirected and directed links, guess. 
# let's start with a DAG for simplicity?
# how to encode a variable number of edges then? 

def sudoku_to_nodes(puzzle, curs_pos, action_type): 
	nodes = []
	
	nc = Node(Types.CURSOR, 0)
	posOffset = (SuN - 1) / 2.0
	ncx = Node(Types.POSITION, 0) # x = column
	ncxx = Node(Types.LEAF, curs_pos[0] - posOffset) # -4 -> 0 4 -> 8 
	ncy = Node(Types.POSITION, 1)
	ncyy = Node(Types.LEAF, curs_pos[1] - posOffset)
	
	ncx.add_child(ncxx)
	ncy.add_child(ncyy)
	nc.add_child(ncx)
	nc.add_child(ncy)
	
	nodes.append(nc)
	
	na = Node(Types.ACTION, 0) 
	# action type = left right up down, 0 -- 3
	ax = 0
	if action_type > 1: 
		ax = 1
	v = -1 # reserve zero for zero motion
	if action_type == 1 or action_type == 3: 
		v = 1
	nax = Node(Types.POSITION, ax)
	naxx = Node(Types.LEAF, v)
	
	na.add_child(nax)
	nax.add_child(naxx)
	
	nodes.append(na)
	
	if False: 
		for y in range(SuN): 
			for x in range(SuN): 
				v = puzzle[y,x]
				nb = Node(Types.BOX, v)
				nbx = Node(Types.POSITION, 0)
				nbxx = Node(Types.LEAF, x - posOffset)
				nby = Node(Types.POSITION, 1)
				nbyy = Node(Types.LEAF, y - posOffset)
				b = (y // SuH)*SuH + (x // SuH)
				nbb = Node(Types.POSITION, 2)
				nbbb = Node(Types.LEAF, b - posOffset)
				
				highlight = 0 # this is mostly icing..
				if x == curs_pos[0] and y == curs_pos[1]: 
					highlight = 1
				nbh = Node(Types.POSITION, 3)
				nbhh = Node(Types.LEAF, highlight)
				
				nbx.add_child(nbxx)
				nby.add_child(nbyy)
				nbb.add_child(nbbb)
				nbh.add_child(nbhh)
				nb.add_child(nbx)
				nb.add_child(nby)
				nb.add_child(nbb)
				nb.add_child(nbh)
				
				nodes.append(nb)
			
	if False: 
		print("total number of nodes:", sum([n.count() for n in nodes]))
		for n in nodes: 
			n.print("")
		
	return nodes
	
def encode_nodes(nodes): 
	cnt = sum([n.count() for n in nodes])
	enc = np.zeros((cnt, 20), dtype=np.float32)
	
	def encode_node(i, node): 
		enc[i, node.typ.value] = 1.0
		# enc[i, node.value + 10] = 1.0
		enc[i, 10] = node.value 
		node.loc = i # need some sort of pointer. 
		i = i + 1
		for k in node.kids: 
			i = encode_node(i, k)
		return i
			
	i = 0
	for n in nodes: 
		i = encode_node(i, n)
	
	msk = np.zeros((cnt,cnt), dtype=np.float32)
	# in the mask: 
	# 1 = attend to self (so .. just project + nonlinearity)
	# 2 = attend to children
	# 4 = attend to parents
	# 8 = attend to peers
	# -- assume softmax is over columns.
	def mask_node(node):
		msk[node.loc, node.loc] = 1.0
		for kid in node.kids: 
			msk[kid.loc, node.loc] = 2.0
		for parent in node.parents: 
			msk[parent.loc, node.loc] = 4.0
		for kid in node.kids: 
			mask_node(kid)
	
	for n in nodes: 
		mask_node(n)
	# let all top-level nodes communicate. 
	for n in nodes: 
		for m in nodes: 
			if n != m: 
				msk[n.loc, m.loc] = 8.0
	
	return enc, msk

def test_nodes(): 
	na = Node(Types.BOX, 0)
	naa = Node(Types.LEAF, 1)
	nab = Node(Types.LEAF, 2)
	nb = Node(Types.BOX, 3)
	nba = Node(Types.POSITION, 4)
	nbaa = Node(Types.LEAF, 5)
	nbb = Node(Types.POSITION, 6)
	nbba = Node(Types.LEAF, 7)
	
	na.add_child(naa)
	na.add_child(nab)
	nb.add_child(nba)
	nb.add_child(nbb)
	nba.add_child(nbaa)
	nbb.add_child(nbba)
	
	nodes = [na, nb]
	return nodes

if __name__ == "__main__":
	plot_rows = 1
	plot_cols = 2
	figsize = (16, 8)
	# plt.ion()
	fig, axs = plt.subplots(plot_rows, plot_cols, figsize=figsize)
	im = [0,0]
	
	N = SuN
	K = SuK
	sudoku = Sudoku(N, K)
	sudoku.fillValues()
	sudoku.printSudoku()
	
	nodes = test_nodes()
	enc,msk = encode_nodes(nodes)
	
	im[0] = axs[0].imshow(enc.T)
	plt.colorbar(im[0], ax=axs[0])
	im[1] = axs[1].imshow(msk)
	plt.colorbar(im[1], ax=axs[1])
	plt.show()
	
	nodes = sudoku_to_nodes(sudoku.mat, np.ones((2,))*2.0, 0)
	
	enc,msk = encode_nodes(nodes)
	print(enc.shape, msk.shape)
	plt.imshow(enc.T)
	plt.colorbar()
	plt.show()
	plt.imshow(msk)
	plt.colorbar()
	plt.show()

