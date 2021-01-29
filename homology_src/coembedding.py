import json
import numpy as np
from utils import *
from linalg import *
from glob import glob
import networkx as nx
from typing import Any, Dict, List, NewType, Tuple, Set

ndarray = NewType('numpy ndarray', np.ndarray)
Graph = NewType('networkx Graph object', nx.Graph)

def unpickled_networks(job_id: str)-> Tuple[Graph, Graph]:
	pickled_files = glob(f"{job_id}/*.gpickle")
	if len(pickled_files) != 2: print(f"[COEMBEDDING ERROR] Expected exactly two pickled networks in \"{job_id}\" directory"); exit()
	
	source_file_idx = [i for i,file in enumerate(pickled_files) if ('source' in file and 'target' not in file)][0] # glob finds the pickled
	target_file_idx = [i for i,file in enumerate(pickled_files) if ('target' in file and 'source' not in file)][0] # in no particular order
	
	networks = [nx.read_gpickle(file) for file in pickled_files]
	
	return networks[source_file_idx], networks[target_file_idx]

def save_labels(source_labels: List[str], target_labels: List[str], job_id: str, run_id: str, strawman_number: str)-> None:
	with open(f"{job_id}/strawman{strawman_number}_source_labels{run_id}.json", 'w') as sptr:
		json.dump({i:node for i,node in enumerate(source_labels)}, sptr)
	
	with open(f"{job_id}/strawman{strawman_number}_target_labels{run_id}.json", 'w') as tptr:
		json.dump({i:node for i,node in enumerate(target_labels)}, tptr)

def dsd(network: Graph, nrw: int)-> Tuple[ndarray, List[str], Dict[str, int]]:
	sorted_nodelist = [node for node, d in sorted(network.degree, key=lambda x: x[1], reverse=True)]
	adj_matrix = nx.to_numpy_matrix(network, nodelist=sorted_nodelist)
	deg_matrix = compute_degree_matrix(adj_matrix)
	
	dsd_matrix = compute_dsd_normalized(adj_matrix, deg_matrix, nrw=nrw)

	indexed_nodes = {node.split('.')[0]:i for i, node in enumerate(network)}  # remove version numbers and sort by degree
	return dsd_matrix, sorted_nodelist, indexed_nodes

def core(args: Dict[str, Any])-> None:
	verbose = boolify(args, 'verbose')
	job_id, run_id, strawman_number, compute, nrw = multiget(args, 'job_id', 'run_id', 'strawman_number', 'compute', 'n_random_walks')

	if verbose: print('\tRetrieving pickled networks...', flush=True)
	source, target = unpickled_networks(job_id)

	source_d, target_d, sorted_source_nodes, sorted_target_nodes = None, None, None, None
	if compute != 't':
		if verbose: print('\tComputing DSD for source network...')
		source_dsd_matrix, sorted_source_nodes, indexed_source_nodes = dsd(source, nrw)

		if verbose: print('\tCreating source pairwise distance matrix...')
		source_d = pairwise_distance_matrix(source_dsd_matrix)

	if compute != 's':
		if verbose: print('\tComputing DSD for target network...')
		target_dsd_matrix, sorted_target_nodes, indexed_target_nodes = dsd(target, nrw)

		if verbose: print('\tCreating target pairwise distance matrix...')
		target_d = pairwise_distance_matrix(target_dsd_matrix)
	
	if verbose: print('\tSaving source and target labels...')
	save_labels(sorted_source_nodes, sorted_target_nodes, job_id, run_id, strawman_number)

	if compute != 't':
		if verbose: print('\tSaving source dsd matrix...')
		np.save(f"{job_id}/strawman{strawman_number}_source_dsd_matrix{run_id}", source_d)

	if compute != 's': # all, both, dsd only
		if verbose: print('\tSaving target dsd matrix...')
		np.save(f"{job_id}/strawman{strawman_number}_target_dsd_matrix{run_id}", target_d)
