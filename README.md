# sudoku-rl

This repository is to investigate & develop a RL agent that learns to solve sudoku problems through active learning. In particular, we aim to learn a model of the world that internalizes the constraints of the game (set property of rows, columns, and blocks), is able to use the model to calculate if an action is legal or not, and ultimately is able to chain a series of actions to solve the game.  

Sudoku is a classic example of a constraint satisfaction problem (CSP), with the constraints consisting of the supplied clues and the rules of the game.  CSP solution most always requires variable compute, typically in the form of search (with backtracking - hence variable memory too) or dynamic programming; typical ML models have fixed compute and fixed memory, so a solution to even this tiny problem would likely generalize valuably.  Furthermore, CSP problems occur in many different domains, including the process of programming itself, hence we believe that by solving this toy problem we may make progress toward general learning. 

## Vector encoding

Initial experiments used a vectoral encoding of the board state $s$ -- each of the 81 squares on the board plus the cursor position, represented as a one-hot encoded vector in a matrix.  This 'board state' matrix is conatenated with a matrix of vectors representing actions $a_t$, which is then passed to a decoder-only transformer.  The transformer is trained via supervised learning from a replay buffer (tuples of $`(s_t, a_t, s_{t+1}, r_{t+1})`$ ) that is created by playing random actions on random boards.  The transformer thus represents a world-model corresponding to the state transition with reward emission: $(s_t, a_t) \rightarrow (s_{t+1}, r_{t+1})$

Even with a small transformer, this required a very large (> 100k) number of samples to learn an approximate world model.  A key problem is that equivariances -- that the set property applies irrespective of the axis, and that movement in one axis is functionally equivalent to movement in another axis -- have to be learned independently for each independent axis $(x,y,block)$, due to their independent encoding in the vector space.  
* This is an assumption - it may be that because attention gates all subspaces in a vector, learning one set-conditional does abstract to all axes.  I don't consider this likely because the information is *within a token rather than between tokens*, but it's clear that attentional gating does stack and generalize in this way in LLMs (e.g. by setting up a 'gather' dependency graph).  (See Anthropic's papers)

## Graph encoding

Data inefficiency -- and that programs are embedded in ASTs -- led to encoding board information **compositionally** via a tree structure.  In this tree, each node has a type, encoded with a one-hot, as well as a value, encoded as an integer (or float, in practice).  Nodes are passed to a decoder-only transformer equivalently to tokens.  Relationships between nodes are {parent, child, sibling, self}.  Since the attention matrix can serve as an adjacency matrix, relationships are encoded as masks over attention.  Each head works on a different mask of the attention matrix, hence there are $4 h$ heads, and thus each head is specialized in conditionally composing particular relation information.  
* This composition must be within the vector dimension of each node, so it remains to be concretely demonstrated that such an encoding yields better generalization.  (It seems to be true, but it might be that transformer internal data representation also needs to be a compositional datatype .. ?)

> Need a diagram here? 

Just the same as the cursor and board, actions are tree encoded, with common attributes like 'axis' shared with the other data.  Nodes are concatenated, padded, and added to actions to form the input to a masked transformer, which is trained (like the previous work) to output new state and reward via supervised learning: $(s_t, a_t) \rightarrow (s_{t+1}, r_{t+1})$

The transformer is substantially different from a baseline Vaswani 2017: 
* It uses L1 rather than dot-product attention -- thus differences along an axis (e.g. the value axis) can be readily measured.  
* LayerNorm is omitted to allow individual axes to encode ordinal information (differences in magnitude vs. differences in angle).  
* Weights in the Q,V matrices are zero-initialized, and heads are default gated off -- so, rather than eliminating random structure from Xavier initialization, structure is progressively added.  This is similar to the idea of LAR (least-angle regression), or even XGBoost, and (presumably) forms a strong form of regularization.  For the MLP layers of a transformer, this allocation can be controlled by the `STNFunction` and `StraightThroughNormal` classes in graph_transformer.py.  
* K values are per-axis gated with a scalar, default `torch.ones`.  Otherwise, the Q and K projections are redundant, since all heads are full dimensioned.  *(Might want to experiment with multi-head attention, in which case K matrix needs to be full rank)*
* Normal LLMs have two MLPs, one which expands to 3x or 4x the dimensions of the tokens.  This transformer just has one square full-rank projection.  

Training the graph transformer on a *very* simple task of predicting the outcome of cursor movements results in essentially zero loss with the PSGD optimizer, and clean step-like behavior as heads are added and information 'flows' through the graph.  

### Back Action

Can forward or causal models can be re-used to estimate inputs from desired outputs via backpropagation?  It seems yes, if latent activations are filled-in by a denoising autoencoder.  Functions `backAction` perform this procedure, facilited by denoising encoders.  

## Code structure
* `gmain.py` -- toplevel, runs the outer training loop. 
* `gracoonizer.py` -- overall model, encapsulates the graph transformer w/ `encodeBoard` and `backLatent` functions.  
* `graph_encoding.py` -- converts a sudoku board, cursor pos, and action to node encoding.  Can be run as a toplevel for testing. 
* `graph_transformer.py` -- the meat of the model.  Has its own `backAction`
* `netdenoise.py` -- dumb simple 4 hidden layer MLP for denoising hidden unit activations.  Needs to operate over the flattened list of nodes. *Under active development*

Todo: 
* Add back in the full puzzle encoding
* Add in set/unset guess actions; see if the network can perfectly predict legality (the vector-encoded network cannot, whereas a parser can (obviously)).
* Maybe write a CUDA kernel for masked L1 attention, which would (finally!) convert $O(N^2)$ to $O(N log N)$ ! 


