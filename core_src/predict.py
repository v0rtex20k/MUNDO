import json
import numpy as np
from utils import *
from linalg import *
from itertools import chain
from collections import Counter
from typing import Any, Dict, Iterable, List, NewType, Tuple, TypeVar, Set

ndarray = NewType('numpy ndarray', np.ndarray)
CountDict = TypeVar('result of Counter', Dict[str, int], Dict[str, float])
LabelDict = NewType('dictionary of {id: set of all its GO labels}', Dict[str, Set[str]])

def load_matrices(job_id: str, network_name: str, run_id: str)-> Tuple[ndarray, ndarray]:
	if not file_exists(f"{job_id}/{network_name}_dsd_pdist_matrix{run_id}.npy", 'PREDICTION', ext='.npy'):  exit()
	if not file_exists(f"{job_id}/munk_matrix{run_id}.npy", 'PREDICTION', ext='.npy'):  exit()
	dsd_matrix = np.load(f"{job_id}/{network_name}_dsd_pdist_matrix{run_id}.npy", allow_pickle=True)
	munk_matrix = np.load(f"{job_id}/munk_matrix{run_id}.npy", allow_pickle=True)

	return dsd_matrix, munk_matrix

def load_labels(job_id: str, network_name: str, run_id: str)-> Dict[str, str]:
	if not file_exists(f"{job_id}/{network_name}_labels{run_id}.json", 'PREDICTION', ext='.json'):  exit()
	with open(f"{job_id}/{network_name}_labels{run_id}.json", 'r') as lptr:
		labels = json.load(lptr)
	
	return {int(k):v for k,v in labels.items()}

def merge_label_dicts(src_labels: Dict[int, str], tgt_labels: Dict[int, str])-> Dict[int, str]:
	src_end = len(src_labels)
	combo = src_labels.copy()
	for i,t in tgt_labels.items():
		combo[src_end+i] = t
	return combo

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

	refseq_to_GO_mapping = {ref_id:uniprot_to_GO_mapping.get(uni_id) for ref_id, uni_id in refseq_to_uniprot_mapping.items()}

	return {r:g for r,g in refseq_to_GO_mapping.items() if g}

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
	n_correct, n_predicted, n_empty = 0, 0, 0

	for test_idx, predicted_label_list in predictions.items():
		real_labels = target_GO_labels.get(test_idx)
		if not real_labels: n_empty +=1; continue # nice diagnostic, but not explicitly used.
		n_predicted += 1
		if any([True for p in predicted_label_list if p in real_labels]): n_correct += 1

	if not n_predicted: return 0
	return (n_correct / n_predicted) * 100


def wmv(dsd_counts: CountDict, munk_counts: CountDict, weights: Tuple[float, float])-> CountDict:
	a, b = weights
	combo = {go_label:(count * a) for go_label, count in dsd_counts.items()} if dsd_counts else dict()
			
	if not munk_counts: return combo
	for go_label, count in munk_counts.items():
		combo[go_label] = (count * b) + combo.get(go_label, 0)

	return combo

def poll_neighborhood(neighbors: ndarray, labels: Dict[int, str],
					  test_idxs: ndarray, indexed_vote_dict: Dict[int, CountDict])-> None:
	real_row_idx = test_idxs[len(indexed_vote_dict)]

	votes = chain(*np.vectorize(labels.get)(neighbors).tolist())
	
	try:
		iterator = iter(votes)
		indexed_vote_dict[real_row_idx] = Counter(votes)
	except TypeError:
		indexed_vote_dict[real_row_idx] = Counter()


def train_test_split(dim: int, block_size: int, seed: int)-> Tuple[ndarray, ndarray]:
	np.random.seed(seed)
	if block_size == dim: block_size = 1
	test_idxs  = np.random.choice(np.arange(dim), size=block_size, replace=False)
	train_idxs = np.delete(np.arange(dim), test_idxs)
	return train_idxs, test_idxs

def cross_species_k_fold_cv(source_GO_labels: Dict[int, str], target_GO_labels: Dict[int, str],
							dsd_matrix: ndarray, munk_matrix: ndarray, k: int, seed: int,
 							p: int, q: int, n_labels: int, weights: List[float], verbose: bool)-> ndarray:

	m = dsd_matrix.shape[0]
	if k == 0: print('Fold size (k) cannot be zero, muchacho'); exit()
	if not is_square(dsd_matrix): print('[PREDICTION ERROR] Provided DSD matrix has invalid shape'); exit()

	if not seed: seed = np.random.randint(1000)

	a,b = weights

	n_rounds, accuracies = abs(k), list()
	for i in range(n_rounds):
		train_idxs, test_idxs = train_test_split(m, m//n_rounds, seed+i)
		if k < 0 : train_idxs, test_idxs = test_idxs, train_idxs # inverted cascade setting, for internal BCB use
		if verbose: print(f'\t[FOLD {i+1}/{n_rounds}] {len(train_idxs)} training nodes, {len(test_idxs)} testing nodes...')
		
		if verbose: print(f'\t\tExtracting fold from full matrices...')
		dsd_fold, munk_fold = None, None
		if a > 0.0: dsd_fold  = dsd_matrix[np.ix_(test_idxs, train_idxs)]	# (m/k) x (m(k-1)/k)
		if b > 0.0: munk_fold = munk_matrix[test_idxs, :] 					# (m/k) x (m(k-1)/k) -- selecting all
																			# columns, so no need to re-index here.
		if verbose: print(f'\t\tLocating neighbor indexes in fold...')
		dsd_grid_idxs, munk_grid_idxs = None, None
		if a > 0.0: dsd_grid_idxs  = np.argsort(dsd_fold,  axis=1)[:,:p]
		if b > 0.0: munk_grid_idxs = np.argsort(munk_fold, axis=1)[:,:q]

		if verbose: print(f'\t\tRe-indexing target neighbors to match original matrices...')
		dsd_neighbor_col_idxs = None
		if a > 0.0: dsd_neighbor_col_idxs = np.apply_along_axis(np.vectorize(train_idxs.__getitem__), 1, dsd_grid_idxs) # (m/k) x (p)

		if verbose: print(f'\t\tPolling neighbors...')
		dsd_votes, munk_votes = dict(), dict()

		if a > 0.0: np.apply_along_axis(poll_neighborhood, 1, dsd_neighbor_col_idxs, target_GO_labels, test_idxs, dsd_votes)
		if b > 0.0: np.apply_along_axis(poll_neighborhood, 1, munk_grid_idxs, source_GO_labels, test_idxs, munk_votes)

		predictions = dict()
		if verbose: print(f'\t\tMaking predictions...')
		for test_idx in test_idxs: # source labels have already been filtered, so FL(u) guaranteed to be in FL(H) U FL(F) forall u
			voting_results = wmv(dsd_votes.get(test_idx), munk_votes.get(test_idx), weights) # {go_label1: count1, go_label2: count2, ...}
			if not voting_results: continue
			predictions[test_idx] = [go_id for go_id, count in sorted(voting_results.items(), key=lambda x: (x[1], x[0]), reverse=True)][:n_labels]
		
		fold_acc = compute_accuracy(predictions, target_GO_labels)
		if verbose: print(f"\t\t\t{round(fold_acc,3)}%")
		accuracies.append(fold_acc)

	return np.asarray(accuracies)

def save_results(results: Dict[int, List[str]], ref_file: str, target_ref_labels: Dict[int, str], job_id: str, run_id: str)-> None:
	indexed_uniprot_ids = dict()
	with open(ref_file, 'r') as rptr:
		refseq_to_uniprot_mapping = json.load(rptr)
		indexed_ref_labels = {v:k for k,v in target_ref_labels.items()}
		for ref_id, uni_id in refseq_to_uniprot_mapping.items():
			idx = indexed_ref_labels.get(ref_id)
			if idx: indexed_uniprot_ids[idx] = uni_id

	with open(f'{job_id}/functional_predictions{run_id}.json', 'w') as fptr:
		for i, fold in enumerate(results):
			results[i] = {k:v for k,v in {indexed_uniprot_ids.get(test_idx): labels for test_idx, labels in fold.items()}.items() if k}
		json.dump(results, fptr)

def core(args: Dict[str, Any])-> Tuple[float, float]:
	verbose, flipped = boolify(args, 'verbose'), boolify(args, 'flipped')
	source_go_file, target_go_file, go_aspect, k, seed, p, q, w, l, job_id, run_id = \
	multiget(args, 'source_go_annotations_file', 'target_go_annotations_file', 'go_aspect', 'k_fold_size',
	'constant_seed', 'p_nearest_dsd_neighbors', 'q_nearest_munk_neighbors', 'weights', 'n_labels', 'job_id', 'run_id')

	if w == [0,0]: return 0.0 # no weights

	if verbose: print('\tLoading matrix embeddings...', flush=True)
	network_name = 'target' if not flipped else 'source'
	dsd_matrix, munk_matrix = load_matrices(job_id, network_name1, run_id)

	if verbose: print('\tLoading row and column labels...')

	source_ref_labels = load_labels(job_id, 'source', run_id) # {i: src_node,...}
	target_ref_labels = load_labels(job_id, 'target', run_id) # {i: tgt_node,...}

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
	if not flipped: accuracy = cross_species_k_fold_cv(source_GO_labels, target_GO_labels, dsd_matrix, munk_matrix, k, seed, p, q, l, w, verbose)
	elif flipped: 	accuracy = cross_species_k_fold_cv(target_GO_labels, source_GO_labels, dsd_matrix, munk_matrix, k, seed, p, q, l, w, verbose)
	if verbose: print(f'\t ---> {accuracy.mean()} ± {accuracy.std()} %')

	return accuracy.mean(), accuracy.std()

#python predict_main.py -s ../experiments/GO/goa_human.gaf -t ../experiments/GO/goa_mouse.gaf -a F P -k -2 -c 291 -j ../experiments/humanBIOGRID_mouseBIOGRID/ -r _8-18-20--10-39-AM
