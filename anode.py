import torch
import pdb

class ANode:
	def __init__(self, typ, val, reward, board_enc, index):
		self.action_type = typ
		self.action_value = val
		self.kids = []
		assert(isinstance(reward, float))
		self.reward = reward
		# board_enc and reward are the *result* of applying the action.
		self.board_enc = board_enc.squeeze().cpu().numpy().astype('float16')
		self.parent = None
		self.reward_pred = torch.zeros(13)
		self.index = index
		self.valid = True
		self.horizon_reward = torch.zeros(13) # looking forward
		self.integral_reward = 0.0 # same as above, but local / not in parent

	def setParent(self, node):
		self.parent = node

	def addKid(self, node):
		self.kids.append(node)
		node.setParent(self)

	def getParent(self):
		return self.parent

	def setRewardPred(self, reward_pred):
		self.reward_pred = reward_pred.squeeze().cpu().numpy().astype('float16')

	def updateReward(self, new_reward):
		# also propagates reward to the parent node
		self.reward = new_reward # outcome of taking our action
		if self.parent is not None:
			if self.action_type == 4:
				self.parent.reward_pred[4 + self.action_value-1] = new_reward
			if self.action_type >= 0 and self.action_type < 4:
				self.parent.reward_pred[self.action_type] = new_reward

	def getAltern(self):
		res = []
		for k in self.kids:
			if k.reward > 0.0 and k.valid:
				res.append(k)
		return res
		
	def integrateReward(self): 
		if len(self.kids) > 0: 
			rw = -5.0
			for j,k in enumerate(self.kids): 
				r = k.integrateReward()
				self.horizon_reward[j] = r
				rw = max(rw, r)
			self.integral_reward = rw + self.reward
		else: 
			self.integral_reward = self.reward
		return self.integral_reward
			
	def flattenNoLeaves(self, node_list): 
		if len(self.kids) > 0: 
			if self.action_type == 8: # no root nodes
				node_list.append(self)
			for k in self.kids: 
				node_list = k.flattenNoLeaves(node_list)
		return node_list

	def print(self, indent, all_actions=False):
		color = "black"
		if self.reward > 0:
			color = "green"
		if self.reward < -1.0:
			color = "red"
		if self.action_type == 4 or len(self.kids) > 1 or all_actions:
			print(colored(f"{indent}[{self.index}]{self.action_type},{self.action_value},{int(self.reward*100)/100} nkids:{len(self.kids)}", color))
			indent = indent + " "
		for k in self.kids:
			k.print(indent, all_actions)
			
	def resetIndex(self): 
		self.index = -1
		for n in self.kids: 
			n.resetIndex()
			
	def setIndex(self, indx, cont): 
		# breadth-first labelling
		# must call multiple times!
		if self.index < 0: 
			self.index = indx
			indx = indx + 1
			cont = True
			return indx,cont # don't recurse
		else: 
			for n in self.kids: 
				indx,cont = n.setIndex(indx,cont)
		return indx,cont
		
	def setIndexRoot(self): 
		# call this from the root node. 
		self.resetIndex()
		indx = 0
		indx,cont = self.setIndex(indx, False)
		while(cont): 
			indx,cont = self.setIndex(indx, False)
	
	def printGexfNode(self, fil): 
		# this implementation is simpler, since it's a DAG (tree)
		def quantize(f): 
			return round(f*4)/4
		print(f'<node id="{self.index}">',file=fil)
		print(f'<attvalues>',file=fil)
		print(f'<attvalue for="0" value="{self.action_type}" />',file=fil)
		print(f'<attvalue for="1" value="{self.action_value}" />',file=fil)
		print(f'<attvalue for="2" value="{quantize(self.reward)}" />',file=fil)
		print(f'<attvalue for="3" value="{quantize(self.integral_reward)}" />',file=fil)
		s = ""
		for i in range(13): 
			s = s + " " + str(quantize(self.horizon_reward[i].item()))
		print(f'<attvalue for="4" value="{s}" />',file=fil)
		print('</attvalues>\n</node>',file=fil)
		for k in self.kids: 
			k.printGexfNode(fil)
				
	def printGexfEdge(self, fil): 
		for k in self.kids: 
			print(f'<edge id="{k.index}" source="{self.index}" target="{k.index}">',file=fil)
			print('</edge>',file=fil)
			k.printGexfEdge(fil)
		

def outputGexf(node, fname): 
	header = '''<?xml version="1.0" encoding="UTF-8"?>
<gexf xmlns="http://gexf.net/1.3" version="1.3">
<graph mode="static" defaultedgetype="directed" idtype="string">
<attributes class="node">
<attribute id="0" title="action_type" type="string"/>
<attribute id="1" title="action_value" type="string"/>
<attribute id="2" title="reward" type="float"/>
<attribute id="3" title="integral_reward" type="float"/>
<attribute id="4" title="horizon_reward" type="string"/>
</attributes>
<nodes> '''
	fil = open(fname, 'w')
	print(header, file=fil)
	node.setIndexRoot()
	node.printGexfNode(fil)
	print('</nodes>',file=fil)
	print('<edges>',file=fil)
	node.printGexfEdge(fil)
	footer = '''
</edges>
</graph>
</gexf>'''
	print(footer,file=fil)
	fil.close()
