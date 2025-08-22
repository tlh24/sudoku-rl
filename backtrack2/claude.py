import torch
import multiprocessing as mp
import threading
from queue import Empty
from typing import List, Tuple, Any, Callable
import numpy as np
from dataclasses import dataclass
from collections import defaultdict

@dataclass
class ValueRequest:
    puzzle_state: Any
    solver_id: int
    request_id: int
    puzzle_idx: int

@dataclass
class ValueResponse:
    value: float
    solver_id: int
    request_id: int
    puzzle_idx: int

@dataclass
class SolveResult:
    puzzle_idx: int
    possibility: float
    guess: Any

class BatchedValueFunction:
    def __init__(self, model, batch_size=128, device='cuda'):
        self.model = model
        self.batch_size = batch_size
        self.device = device
        
    def process_batch(self, requests: List[ValueRequest]) -> List[ValueResponse]:
        states = [req.puzzle_state for req in requests]
        batch = torch.from_numpy(np.stack(states)).to(self.device)
        
        with torch.no_grad():
            values = self.model(batch).cpu().numpy()
            
        return [
            ValueResponse(
                value=float(values[i]),
                solver_id=req.solver_id,
                request_id=req.request_id,
                puzzle_idx=req.puzzle_idx
            )
            for i, req in enumerate(requests)
        ]

def value_function_worker(model, input_queue, output_queues, active_workers, batch_size=128):
    """Worker that runs the value function in the main process"""
    value_fn = BatchedValueFunction(model, batch_size=batch_size)
    pending_requests: List[ValueRequest] = []
    
    while active_workers.value > 0:  # Run while there are active workers
        # Try to fill up a batch
        try:
            while len(pending_requests) < batch_size:
                try:
                    req = input_queue.get(timeout=0.01)
                    if req is None:  # Shutdown signal from a worker
                        with active_workers.get_lock():
                            active_workers.value -= 1
                        continue
                    pending_requests.append(req)
                except Empty:
                    break  # Break inner loop if queue is empty
                
            # Process a batch if we have requests and either batch is full or no more incoming requests
            if pending_requests and (len(pending_requests) >= batch_size or input_queue.empty()):
                responses = value_fn.process_batch(pending_requests)
                
                # Group responses by solver_id
                grouped_responses = defaultdict(list)
                for resp in responses:
                    grouped_responses[resp.solver_id].append(resp)
                    
                # Send responses to appropriate queues
                for solver_id, resps in grouped_responses.items():
                    output_queues[solver_id].put(resps)
                
                pending_requests.clear()
                
        except Exception as e:
            print(f"Error in value function worker: {e}")
            # Handle any remaining requests before breaking
            if pending_requests:
                try:
                    responses = value_fn.process_batch(pending_requests)
                    for resp in responses:
                        output_queues[resp.solver_id].put(resp)
                except Exception as e2:
                    print(f"Error handling remaining requests: {e2}")
            break
    
    # Process any final pending requests
    if pending_requests:
        try:
            responses = value_fn.process_batch(pending_requests)
            for resp in responses:
                output_queues[resp.solver_id].put(resp)
        except Exception as e:
            print(f"Error processing final batch: {e}")

def solver_worker(
    puzzles: np.ndarray,
    start_idx: int,
    end_idx: int,
    solver_id: int,
    input_queue: mp.Queue,
    output_queue: mp.Queue,
    result_queue: mp.Queue,
    n_iterations: int,
):
    """Worker that runs the stochastic solve algorithm on a chunk of puzzles"""
    next_request_id = 0
    pending_requests = {}
    
    try:
        def value_fn(puzzle_state):
            nonlocal next_request_id
            request = ValueRequest(
                puzzle_state=puzzle_state,
                solver_id=solver_id,
                request_id=next_request_id,
                puzzle_idx=-1
            )
            next_request_id += 1
            
            input_queue.put(request)
            pending_requests[request.request_id] = puzzle_state
            
            while True:
                responses = output_queue.get()
                for resp in responses:
                    if resp.request_id in pending_requests:
                        del pending_requests[resp.request_id]
                        if resp.request_id == request.request_id:
                            return resp.value
        
        # Process each puzzle in the assigned chunk
        for idx in range(start_idx, end_idx):
            puzzle = puzzles[idx]
            poss, guess = stochasticSolve(puzzle, n_iterations, value_fn, True)
            result_queue.put(SolveResult(puzzle_idx=idx, possibility=poss, guess=guess))
    
    finally:
        # Signal completion to value function worker
        input_queue.put(None)

def parallel_solve(
    puzzles: np.ndarray,
    model: torch.nn.Module,
    n_iterations: int = 64,
    n_workers: int = 4,
    batch_size: int = 128
) -> List[Tuple[float, Any]]:
    """
    Solve multiple puzzles in parallel with efficient batching of value function calls.
    
    Args:
        puzzles: Numpy array of puzzle states to solve
        model: PyTorch model to use as value function
        n_iterations: Number of iterations for each solve attempt
        n_workers: Number of parallel solver workers
        batch_size: Batch size for value function calls
    
    Returns:
        List of (possibility, guess) tuples, one for each input puzzle
    """
    n_puzzles = len(puzzles)
    chunk_size = (n_puzzles + n_workers - 1) // n_workers
    
    # Create queues and shared counter for active workers
    value_fn_input_queue = mp.Queue()
    value_fn_output_queues = {i: mp.Queue() for i in range(n_workers)}
    result_queue = mp.Queue()
    active_workers = mp.Value('i', n_workers)
    
    # Start solver workers
    solver_processes = []
    for i in range(n_workers):
        start_idx = i * chunk_size
        end_idx = min(start_idx + chunk_size, n_puzzles)
        
        if start_idx >= n_puzzles:
            with active_workers.get_lock():
                active_workers.value -= 1
            continue
            
        p = mp.Process(
            target=solver_worker,
            args=(
                puzzles, start_idx, end_idx, i,
                value_fn_input_queue, value_fn_output_queues[i],
                result_queue, n_iterations
            )
        )
        p.start()
        solver_processes.append(p)
    
    # Start result collector
    results = [None] * n_puzzles
    def result_collector():
        collected = 0
        target = n_puzzles
        while collected < target:
            try:
                result = result_queue.get(timeout=1.0)
                results[result.puzzle_idx] = (result.possibility, result.guess)
                collected += 1
            except Empty:
                # Check if all workers are done and we've collected all results
                if active_workers.value == 0 and collected >= target:
                    break
    
    collector_thread = threading.Thread(target=result_collector)
    collector_thread.start()
    
    # Run value function in main process
    value_function_worker(model, value_fn_input_queue, value_fn_output_queues, active_workers, batch_size)
    
    # Clean up
    for p in solver_processes:
        p.join()
    collector_thread.join()
    
    return results
