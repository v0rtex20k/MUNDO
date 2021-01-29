import ast
import json
import numpy as np
from utils import *
from linalg import *
import networkx as nx
from itertools import chain
from collections import Counter
from typing import Any, Dict, Iterable, List, NewType, Tuple, TypeVar, Set

ndarray = NewType('numpy ndarray', np.ndarray)
CountDict = TypeVar('result of Counter', Dict[str, int], Dict[str, float])
LabelDict = NewType('dictionary of {id: set of all its GO labels}', Dict[str, Set[str]])

def load_matrices(job_id: str, run_id: str)-> Tuple[ndarray, ndarray]:
	if not file_exists(f"{job_id}/dsd_pdist_matrix{run_id}.npy", 'PREDICTION', ext='.npy'):  exit()
	if not file_exists(f"{job_id}/source_dsd_pdist_matrix{run_id}.npy", 'PREDICTION', ext='.npy'):  exit()
	source_dsd_matrix = np.load(f"{job_id}/source_dsd_pdist_matrix{run_id}.npy", allow_pickle=True)
	target_dsd_matrix = np.load(f"{job_id}/dsd_pdist_matrix{run_id}.npy", allow_pickle=True)

	return source_dsd_matrix, target_dsd_matrix

def load_labels(job_id: str, network_name: str, strawman_number: str, run_id: str)-> Dict[str, str]:
	if not file_exists(f"{job_id}/strawman_{network_name}_labels{run_id}.json", 'PREDICTION', ext='.json'): exit()

	labels = None
	with open(f"{job_id}/strawman_{network_name}_labels{run_id}.json", "r") as lptr:
		labels = json.load(lptr)
	
	labels = {v:int(k) for k,v in labels.items()}
	return labels

def load_and_index_hits(source_ref_labels: Dict[str, str], target_ref_labels: Dict[str, str], job_id: str, strawman_number: str)-> Dict[int, List[int]]:
	if not file_exists(f'{job_id}/strawman{strawman_number}_hits.txt', 'PREDICTION'): exit()
	hits = dict()
	with open(f'{job_id}/strawman{strawman_number}_hits.txt', 'r') as rptr:
		for line in rptr.readlines():
			a,b = line.split('\t')
			hits[a] = ast.literal_eval(b)

	hit_idxs = dict()
	for target_query, hitlist in hits.items():
		hit_idxs[target_ref_labels.get(target_query)] = [source_ref_labels.get(source_match) for source_match in hitlist]

	return {k:v for k,v in hit_idxs.items() if k and v}

def map_ref_to_GO(go_file: str, ref_file: str, go_aspect: str)-> LabelDict:
	if not file_exists(go_file, 'PREDICTION', ext='.gaf'): exit()
	if not file_exists(ref_file, 'PREDICTION', ext='.json'): exit()
	refseq_to_uniprot_mapping, uniprot_to_GO_mapping = dict(), dict()
	
	with open(ref_file, 'r') as rptr, open(go_file, 'r') as gptr:
		refseq_to_uniprot_mapping = json.load(rptr)

		for entry in gptr.readlines()[12:]: # skip the description lines
			db, uni_id, _, _, go_id, _, _, _, aspect = entry.strip().split('\t')[:9]
			if not "uniprotkb" in db.lower(): continue # GO labels are ** not ** unique ---> {uni: {go_id1, go_id2, ...}, ...}
			if 'A' not in go_aspect and aspect.strip().upper() not in go_aspect: continue # filter by aspect
			uniprot_to_GO_mapping[uni_id] = uniprot_to_GO_mapping.get(uni_id, set()) | {go_id}

	refseq_to_GO_mapping = {r:g for r,g in {ref_id:uniprot_to_GO_mapping.get(uni_id) for ref_id, uni_id in refseq_to_uniprot_mapping.items()}.items() if g}

	return refseq_to_GO_mapping 

def filter_labels(tgt_GO_labels: Dict[int, str], annotation_counts: CountDict, low: int = 50, high: int = 500)-> LabelDict:
	filtered_tgt_labels = dict()

	# n_limited = len([ann for ann, c in annotation_counts.items() if low <= c <= high])
	# print(f'{n_limited} between {low} and {high}')

	for i, go_ids in tgt_GO_labels.items():
		for go_id in go_ids:
			n_annotations = annotation_counts.get(go_id, -1) # n_annotations from TARGET networks
			if not low <= n_annotations <= high: continue
			filtered_tgt_labels[i] = filtered_tgt_labels.get(i, set()) | {go_id}
		if not filtered_tgt_labels.get(i):
			filtered_tgt_labels[i] = set() # fill in missing ones with empties - they just have no vote.

	return filtered_tgt_labels # {i: {go_id1, go_id2, ...}, ...}

def compute_accuracy(predictions: Dict[int, List[str]], target_GO_labels: Dict[int, str])-> float:
	n_correct = 0
	n_predicted = 0
	n_empty = 0
	for test_idx, predicted_label_list in predictions.items():
		real_labels = target_GO_labels.get(test_idx)
		if not real_labels: n_empty +=1; continue
		n_predicted += 1
		if any([True for p in predicted_label_list if p in real_labels]): n_correct += 1

	#print(f"{n_empty} empty out of {n_predicted} predictions")

	if not n_predicted: return 0
	return (n_correct/n_predicted)*100

def wmv(target_counts: CountDict, hit_counts: CountDict, weights: Tuple[float, float])-> CountDict:
	tw, hw = weights
	combo = {go_label:(count * tw) for go_label, count in target_counts.items()} if target_counts else dict()
	if not hit_counts: return combo
	
	for go_label, count in hit_counts.items():
		combo[go_label] = (count * hw) + combo.get(go_label, 0)

	return combo

def poll_neighborhood(neighbors: ndarray, labels: Dict[int, str], test_idxs: ndarray, indexed_vote_dict: Dict[int, CountDict])-> None:
	real_row_idx = test_idxs[len(indexed_vote_dict)]
	votes = chain(*np.vectorize(labels.get)(neighbors).tolist())
	
	try:
		iterator = iter(votes)
		indexed_vote_dict[real_row_idx] = Counter(votes)
	except TypeError:
		indexed_vote_dict[real_row_idx] = Counter()

def poll_hits(test_idxs: ndarray, hit_idxs: Dict[int,int], source_dsd_matrix: ndarray, 
			  source_GO_labels: Dict[int, str], q: int, strawman_number: str)-> CountDict:
	
	hit_votes = dict()
	include_hit_dsd_neighbors = True if '+' in strawman_number else False
	for test_idx in test_idxs:
		hitlist = hit_idxs.get(test_idx)
		if not hitlist: hit_votes[test_idx] = dict(); continue

		for source_match_idx in hitlist:
			match_votes, match_neighbor_votes = Counter(source_GO_labels.get(source_match_idx)), Counter()
			
			if include_hit_dsd_neighbors:
				match_neighbor_idxs  = np.argsort(source_dsd_matrix[source_match_idx, :])[:q]
				match_neighbor_votes = Counter(chain(*np.vectorize(source_GO_labels.__getitem__)(match_neighbor_idxs)))

			hit_votes[test_idx] =  match_votes + match_neighbor_votes

	return hit_votes

def train_test_split(dim: int, block_size: int, seed: int)-> Tuple[ndarray, ndarray]:
	np.random.seed(seed)
	if block_size == dim: block_size = 1
	test_idxs  = np.random.choice(np.arange(dim), size=block_size, replace=False)
	train_idxs = np.delete(np.arange(dim), test_idxs)
	return train_idxs, test_idxs

def k_fold_cv(source_GO_labels: Dict[int, str], target_GO_labels: Dict[int, str], hit_idxs: ndarray, source_dsd_matrix: ndarray, target_dsd_matrix: ndarray,
k: int, seed: int, p: int, q: int, n_labels: int, weights: List[float], strawman_number: str, verbose: bool)-> List[float]:

	m = target_dsd_matrix.shape[0]
	if not k: print('Fold size (k) cannot be zero, muchacho'); exit()
	if not is_square(source_dsd_matrix) or not is_square(target_dsd_matrix): print('[PREDICTION ERROR] Provided matrices have invalid shapes'); exit()

	if not seed: seed = np.random.randint(10000)

	n_rounds = abs(k)
	accuracy = list()
	for i in range(n_rounds):
		train_idxs, test_idxs = train_test_split(m, m//n_rounds, seed+i)
		if k < 0 : train_idxs, test_idxs = test_idxs, train_idxs # cascade setting, for internal BCB use
		if verbose: print(f'\tStarting fold {i+1}/{n_rounds} with {len(train_idxs)} training nodes and {len(test_idxs)} testing nodes...')
		if verbose: print(f'\t\tExtracting fold from full matrices...')
		target_grid = np.ix_(test_idxs, train_idxs)
		target_fold = target_dsd_matrix[target_grid] # (m/k) x (m(k-1)/k)

		if verbose: print(f'\t\tLocating neighbor indexes in fold...')
		target_grid_idxs = np.argsort(target_fold, axis=1)[:,:p]

		if verbose: print(f'\t\tRe-indexing neighbors to match original matrices...')
		target_neighbor_col_idxs = np.apply_along_axis(np.vectorize(train_idxs.__getitem__), 1, target_grid_idxs) # (m/k) x (p)

		if verbose: print(f'\t\tPolling neighbors...')
		target_votes = dict()

		np.apply_along_axis(poll_neighborhood, 1, target_neighbor_col_idxs, target_GO_labels, test_idxs, target_votes)
		hit_votes = poll_hits(test_idxs, hit_idxs, source_dsd_matrix, source_GO_labels, q, strawman_number)

		if verbose: print(f'\t\tMaking predictions...')
		predictions = dict()
		for test_idx in test_idxs: # source labels have already been filtered 50-500, so FL(u) guaranteed to be in FL(H) U FL(F) forall u
			voting_results = wmv(target_votes[test_idx], hit_votes[test_idx], weights) # {go_label1: count1, go_label2: count2, ...}
			predictions[test_idx] = [go_label for go_label, count in sorted(voting_results.items(), key=lambda x: (x[1], x[0]), reverse=True)][:n_labels]

		fold_acc = compute_accuracy(predictions, target_GO_labels)
		if verbose: print(f"\t\t\t{fold_acc}%")
		accuracy.append(fold_acc)

	return np.asarray(accuracy)

def save_results(results: Dict[int, List[str]], ref_file: str, target_ref_labels: Dict[int, str], job_id: str, run_id: str, strawman_number: str)-> None:
	indexed_uniprot_ids = dict()
	with open(ref_file, 'r') as rptr:
		refseq_to_uniprot_mapping = json.load(rptr)
		indexed_ref_labels = {v:k for k,v in target_ref_labels.items()}
		for ref_id, uni_id in refseq_to_uniprot_mapping.items():
			idx = indexed_ref_labels.get(ref_id)
			if idx:
				indexed_uniprot_ids[idx] = uni_id

	with open(f'{job_id}/strawman{strawman_number}functional_predictions{run_id}.json', 'w') as fptr:
		for i, fold in enumerate(results):
			results[i] = {k:v for k,v in {indexed_uniprot_ids.get(test_idx): labels for test_idx, labels in fold.items()}.items() if k}
		json.dump(results, fptr)

def core(args: Dict[str, Any])-> Tuple[float, float]:
	verbose, flipped = boolify(args, 'verbose'), boolify(args, 'flipped')
	source_go_file, target_go_file, go_aspect, k, seed, p, q, w, l, job_id, run_id, strawman_number = \
	multiget(args, 'source_go_annotations_file', 'target_go_annotations_file', 'go_aspect', 'k_fold_size', 'constant_seed',
	'p_nearest_target_neighbors', 'q_nearest_source_neighbors', 'weights', 'n_labels', 'job_id', 'run_id', 'strawman_number')

	if verbose: print('\tLoading matrix embeddings...', flush=True)
	network_name = 'target' if not flipped else 'source'
	source_dsd_matrix, target_dsd_matrix = load_matrices(job_id, run_id)

	if verbose: print('\tLoading source and target labels...')
	source_ref_labels = load_labels(job_id, 'source', strawman_number, run_id) # {src_node: i,...}
	target_ref_labels = load_labels(job_id, 'target', strawman_number, run_id) # {tgt_node: i,...}

	if verbose: print('\tLoading hits...')
	hit_idxs = dict()
	if not flipped: hit_idxs = load_and_index_hits(source_ref_labels, target_ref_labels, job_id, strawman_number) # {t_i: s_i, ...}
	elif flipped: 	hit_idxs = load_and_index_hits(target_ref_labels, source_ref_labels, job_id, strawman_number) # {t_i: s_i, ...}

	if verbose: print('\tMapping refseq ids to GO annotations...' )
	source_refseq_to_GO_mapping = map_ref_to_GO(source_go_file, f"{job_id}/source_refseq_to_uniprot_mapping.json", go_aspect)
	target_refseq_to_GO_mapping = map_ref_to_GO(target_go_file, f"{job_id}/target_refseq_to_uniprot_mapping.json", go_aspect)

	if verbose: print('\tIndexing target labels...')
	source_GO_labels = {i:source_refseq_to_GO_mapping.get(c, set()) for i, c in source_ref_labels.items()} # {i: {go_id1, ...},...}
	target_GO_labels = {i:target_refseq_to_GO_mapping.get(c, set()) for i, c in target_ref_labels.items()} # {i: {go_id1, ...},...}

	if verbose: print('\tFiltering and indexing source and target labels...')
	annotation_counts = Counter(chain(*target_GO_labels.values())) if not flipped \
						else Counter(chain(*source_GO_labels.values())) # {go_id: n, ...}
	source_GO_labels = filter_labels(source_GO_labels, annotation_counts) # {i: {go_id1, ...},...}
	target_GO_labels = filter_labels(target_GO_labels, annotation_counts) # {i: {go_id1, ...},...}
	
	if verbose: print('\tBeginning cross-species k-fold cross validation...')
	accuracy = None
	if not flipped:
		accuracy = cross_species_k_fold_cv(source_GO_labels, target_GO_labels, dsd_matrix, munk_matrix, k, seed, p, q, l, w, verbose)
	elif flipped:
		accuracy = cross_species_k_fold_cv(target_GO_labels, source_GO_labels, dsd_matrix, munk_matrix, k, seed, p, q, l, w, verbose)
	if verbose: print(f'\t ---> {accuracy.mean()} ± {accuracy.std()} %')

	# if verbose: print('\tSaving predictions...')
	# save_results(results, f"{job_id}/target_refseq_to_uniprot_mapping.json", target_ref_labels, job_id, run_id)
	return accuracy.mean(), accuracy.std()

def compute_pdist(job_id: str, run_id: str)-> None:
	print('\tComputing source pdist matrix...')
	source_dsd_matrix = np.load(f"{job_id}/source_dsd_matrix{run_id}.npy", allow_pickle=True)
	source_pdist  = pairwise_distance_matrix(source_dsd_matrix)
	np.save(f"{job_id}/source_dsd_pdist_matrix{run_id}.npy", source_pdist)

