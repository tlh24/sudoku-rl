import math
import numpy as np
import torch
from enum import Enum
from sudoku_gen import Sudoku
import matplotlib.pyplot as plt
from constants import SuN, SuH, SuK
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
		if self.loc > 0: # graph can contain loops! 
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
		
		
def sudokuToNodes(puzzle, curs_pos): 
	nodes = []
	posOffset = (SuN - 1) / 2.0
	board_nodes = [[] for _ in range(SuN)]
	
	for x in range(SuN): # x = row
		for y in range(SuN): # y = column
			b = (y // SuH)*SuH + (x // SuH)
			v = puzzle[x,y]
			nb = Node(Types.BOX, v)
			
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
	
	xsets = Node(Types.SET, 1)
	for x in range(SuN): 
		nb = Node(Types.SET, 0)
		nb.addChild( Node(Axes.X_AX, x - posOffset) )
		for y in range(SuN): 
			nb.addChild( board_nodes[x][y] )
		xsets.addChild(nb)
		nodes.append(nb)
	nboard.addChild(xsets)
	nodes.append(xsets)
	
	ysets = Node(Types.SET, 1)
	for y in range(SuN): 
		nb = Node(Types.SET, 0)
		nb.addChild( Node(Axes.Y_AX, y - posOffset) )
		for x in range(SuN): 
			nb.addChild( board_nodes[x][y] )
		ysets.addChild(ysets)
		nodes.append(nb)
	nboard.addChild(ysets)
	nodes.append(ysets)
		
	bsets = Node(Types.SET, 1)
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
	
	for n in nodes: 
		n.clearLoc()
	i = 0
	for n in nodes: 
		i = n.setLoc(i)
	print("total number of nodes:", i)
	
	return nodes
	
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
	puzzle = np.zeros((SuN, SuN))
	puzzle[0,:] = [1,2,3,4]
	puzzle[1,:] = [2,3,4,5]
	puzzle[2,:] = [3,4,5,6]
	puzzle[3,:] = [4,5,6,7]
	curs_pos = [0,0]
	nodes = sudokuToNodes(puzzle, curs_pos)
	
	for n in nodes: 
		n.resetRefcnt()
	for n in nodes: 
		n.print("")
		
	outputGexf(nodes)
	# seems to be working..

