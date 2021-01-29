import json
import numpy as np
from utils import *
from linalg import *
from glob import glob
import networkx as nx
from typing import Any, Dict, List, NewType, Tuple, Set

ndarray = NewType('numpy ndarray', np.ndarray)
Graph = NewType('networkx Graph object', nx.Graph)

def load_matrices(job_id: str, precomputed_id: str)-> Tuple[ndarray, ndarray]:
	if not file_exists(f"{job_id}/source_dsd_matrix{precomputed_id}.npy", 'PREDICTION', ext='.npy'):  exit()
	if not file_exists(f"{job_id}/target_dsd_matrix{precomputed_id}.npy", 'PREDICTION', ext='.npy'):  exit()
	source_dsd_matrix = np.load(f"{job_id}/source_dsd_matrix{precomputed_id}.npy", allow_pickle=True)
	target_dsd_matrix = np.load(f"{job_id}/target_dsd_matrix{precomputed_id}.npy", allow_pickle=True)

	return source_dsd_matrix, target_dsd_matrix

def load_reciprocal_best_hits(job_id: str)-> Set[Tuple[str, str]]:
	if not file_exists(f'{job_id}/reciprocal_best_hits.txt', 'COEMBEDDING'): exit()
	reciprocal_best_hits = set()
	with open(f'{job_id}/reciprocal_best_hits.txt', 'r') as rptr:
		for line in rptr.readlines():
			src_query, tgt_match = line.lstrip().rstrip().split('\t')
			reciprocal_best_hits.add((src_query, tgt_match))
	return reciprocal_best_hits

def unpickled_networks(job_id: str)-> Tuple[Graph, Graph]:
	pickled_files = glob(f"{job_id}/*.gpickle")
	if len(pickled_files) != 2: print(f"[COEMBEDDING ERROR] Expected exactly two pickled networks in \"{job_id}\" directory"); exit()
	
	source_file_idx = [i for i,file in enumerate(pickled_files) if ('source' in file and 'target' not in file)][0] # glob finds the pickled
	target_file_idx = [i for i,file in enumerate(pickled_files) if ('target' in file and 'source' not in file)][0] # in no particular order
	
	networks = [nx.read_gpickle(file) for file in pickled_files]
	
	return networks[source_file_idx], networks[target_file_idx]

def save_labels(source_labels: List[str], target_labels: List[str], run_id: str, job_id: str)-> None:
	with open(f"{job_id}/source_labels{run_id}.json", 'w') as sptr:
		json.dump({i:node for i,node in enumerate(source_labels)}, sptr)
	
	with open(f"{job_id}/target_labels{run_id}.json", 'w') as tptr:
		json.dump({i:node for i,node in enumerate(target_labels)}, tptr)

def embed_matrices(source_rkhs: ndarray, target_diffusion_matrix: ndarray, landmark_indices: List[Tuple[int, int]])-> ndarray:
    source_landmark_indices, target_landmark_indices = zip(*landmark_indices)
    return np.linalg.pinv(source_rkhs[source_landmark_indices,:]).dot(target_diffusion_matrix[target_landmark_indices,:]).T

def embed_network(network: Graph, nrw: int)-> Tuple[ndarray, List[str], Dict[str, int]]:
	sorted_nodelist = [node for node, d in sorted(network.degree, key=lambda x: x[1], reverse=True)]
	indexed_nodes = {node.split('.')[0]:i for i, node in enumerate(sorted_nodelist)}  # remove version numbers and sort by degree

	if nrw:
		adj_matrix = nx.to_numpy_matrix(network, nodelist=sorted_nodelist)
		dsd_matrix = turbo_dsd(adj_matrix, nrw)
		return dsd_matrix, sorted_nodelist, indexed_nodes
	
	return sorted_nodelist, indexed_nodes

def coembed_networks(source_dsd: ndarray, target_dsd: ndarray, landmark_indices: List[Tuple[int, int]], verbose: bool)-> ndarray:

	if verbose: print('\tComputing RKHS for source network... ')
	source_rkhs = rkhs(source_dsd)
	
	if verbose: print('\tEmbedding matrices... ')
	target_rkhs_hat = embed_matrices(source_rkhs, target_dsd, landmark_indices)
	
	if verbose: print('\tCreating final munk matrix... ')
	munk_matrix = np.dot(source_rkhs, target_rkhs_hat.T)

	return munk_matrix.T # m x n

def index_landmarks(source_indexed_nodes: Dict[str, int], target_indexed_nodes: Dict[str, int], \
														  reciprocal_best_hits: Set[Tuple[str, str]])-> List[Tuple[str, str]]:
	return [(source_indexed_nodes[src_query], target_indexed_nodes[tgt_match]) for src_query, tgt_match in reciprocal_best_hits]

def core(args: Dict[str, Any])-> None:
	precomputed_id, job_id, run_id, compute, nrw, gamma, thresh = \
	multiget(args, 'precomputed_id', 'job_id', 'run_id', 'compute', 'n_random_walks', 'gamma', 'thresh')

	verbose, normalized = boolify(args, 'verbose'), boolify(args, 'normalized')

	source_dsd_matrix, target_dsd_matrix= None, None

	if verbose: print('\tRetrieving pickled networks...', flush=True)
	source, target = unpickled_networks(job_id)

	if verbose: print('\tRetrieving reciprocal best hits...')
	reciprocal_best_hits = load_reciprocal_best_hits(job_id)

	if precomputed_id:
		if verbose: print('\tLoading precomputed matrix embeddings...', flush=True)
		nrw = None
		source_dsd_matrix, target_dsd_matrix = load_matrices(job_id, precomputed_id)
		sorted_source_nodes, indexed_source_nodes = embed_network(source, nrw)
		sorted_target_nodes, indexed_target_nodes = embed_network(target, nrw)
	else:
		if verbose: print('\tComputing DSD for source network...')
		source_dsd_matrix, sorted_source_nodes, indexed_source_nodes = embed_network(source, nrw)

		if verbose: print('\tComputing DSD for target network...')
		target_dsd_matrix, sorted_target_nodes, indexed_target_nodes = embed_network(target, nrw)

	munk_matrix= None
	if compute != 'd': # all, both, combined only

		if verbose: print('\tIndexing landmarks...')
		landmark_indices = index_landmarks(indexed_source_nodes, indexed_target_nodes, reciprocal_best_hits)

		if verbose: print('\tMaking source DSD matrix hermitian...')
		s_hermit = make_hermitian(source_dsd_matrix, gamma, thresh)
		if verbose: print('\tMaking target DSD matrix hermitian...')
		t_hermit = make_hermitian(target_dsd_matrix, gamma, thresh)

		if verbose: print('\tCombedding networks...')
		munk_matrix = coembed_networks(s_hermit, t_hermit, landmark_indices, verbose)


	if verbose: print('\tSaving row and column labels...')
	save_labels(sorted_source_nodes, sorted_target_nodes, run_id, job_id)

	if compute == 'a':
		if verbose: print('\tSaving source dsd matrix...')
		np.save(f"{job_id}/source_dsd_matrix{run_id}", source_dsd_matrix)

	if compute != 'm': # all, both, dsd only
		if verbose: print('\tSaving target dsd matrix...')
		np.save(f"{job_id}/target_dsd_matrix{run_id}", target_dsd_matrix)
	
	if compute != 'd': # all, both, combined
		if verbose: print('\tSaving munk matrix...')
		np.save(f"{job_id}/munk_matrix{run_id}", munk_matrix)
