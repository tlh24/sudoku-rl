import torch

class ANode:
	def __init__(self, typ, val, reward, board_enc, index):
		self.action_type = typ
		self.action_value = val
		self.kids = []
		self.reward = reward
		# board_enc and reward are the *result* of applying the action.
		self.board_enc = board_enc
		self.parent = None
		self.reward_pred = torch.zeros(14)
		self.index = index
		self.valid = True

	def setParent(self, node):
		self.parent = node

	def addKid(self, node):
		self.kids.append(node)
		node.setParent(self)

	def getParent(self):
		return self.parent

	def setRewardPred(self, reward_pred):
		self.reward_pred = reward_pred

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
