from utils import *
from metrics import compute_metrics

DiGraph   = NewType('nx DiGraph', nx.DiGraph)
CountDict = TypeVar('result of Counter', Dict[str, int], Dict[str, float])
HyperTups = TypeVar('List of tuples of hyperparameters', List[Tuple[float, float, float, float]], List[Tuple[int, int, int, int]])
LabelDict = TypeVar('dictionary of {id: set of all its GO labels}', Dict[str, Set[str]], Dict[int, List[str]])

def load_matrices(job_id: str, run_id: str)-> Tuple[ndarray, ndarray]:

	if not file_exists(f"{job_id}/target_dsd_matrix{run_id}.npy", 'PREDICTION', ext='.npy'):  exit()
	if not file_exists(f"{job_id}/munk_matrix{run_id}.npy", 'PREDICTION', ext='.npy'):  exit()
	dsd_matrix = np.load(f"{job_id}/target_dsd_matrix{run_id}.npy", allow_pickle=True)
	munk_matrix = np.load(f"{job_id}/munk_matrix{run_id}.npy", allow_pickle=True)

	return dsd_matrix, munk_matrix

def load_labels(job_id: str, network_name: str, run_id: str)-> Dict[str, str]:
	if not file_exists(f"{job_id}/{network_name}_labels{run_id}.json", 'PREDICTION', ext='.json'): exit()
	with open(f"{job_id}/{network_name}_labels{run_id}.json", 'r') as lptr:
		labels = json.load(lptr)
	labels = {int(k):v for k,v in labels.items()}
	return labels

def merge_label_dicts(src_labels: Dict[int, str], tgt_labels: Dict[int, str])-> Dict[int, str]:
	src_end = len(src_labels)
	combo = src_labels.copy()
	for i,t in tgt_labels.items():
		combo[src_end+i] = t
	return combo

def map_ref_to_GO(go_file: str, ref_file: str, go_aspect: str, intermediate_mapping_file: str=None)-> LabelDict:
	assert(set(go_aspect) & {'A', 'P', 'F', 'C'}) # valid go aspect
	if not file_exists(go_file, 'PREDICTION', ext='.gaf'): exit()
	if not file_exists(ref_file, 'PREDICTION', ext='.json'): exit()
	refseq_to_uniprot_mapping, uniprot_to_GO_mapping = dict(), dict()

	approved_dbs = ["uniprotkb", "pombase"] # can be updated as needed

	with open(ref_file, 'r') as rptr, open(go_file, 'r') as gptr:
		refseq_to_uniprot_mapping = json.load(rptr)
		for entry in gptr.readlines():
			if entry[0] == '!': continue # skip the description lines
			db, uni_id, _, _, go_id, _, _, _, aspect = entry.strip().split('\t')[:9]
			if not db.lower() in approved_dbs: continue # GO labels are ** not ** unique ---> {uni: {go_id1, go_id2, ...}, ...}
			if 'A' not in go_aspect and aspect.strip().upper() not in go_aspect: continue # filter by aspect
			uniprot_to_GO_mapping[uni_id] = uniprot_to_GO_mapping.get(uni_id, set()) | {go_id}

	if intermediate_mapping_file is not None:
		print('\tUsing pombase mapping file...')
		with open(intermediate_mapping_file, 'r') as mptr:
			inter_mapping = dict()
			for entry in mptr.readlines():
				og_id, new_id = entry.strip().split('\t')
				inter_mapping[og_id] = new_id
			uniprot_to_GO_mapping = {k:v for k,v in {inter_mapping.get(k,None):v \
							for k,v in uniprot_to_GO_mapping.items()}.items() if k}

	refseq_to_GO_mapping = {r:g for r,g in {ref_id:uniprot_to_GO_mapping.get(uni_id) \
								for ref_id, uni_id in refseq_to_uniprot_mapping.items()}.items() if g}

	return refseq_to_GO_mapping

def wmv(dsd_counts: CountDict, munk_counts: CountDict, weights: Tuple[float, float])-> CountDict:
	(a, b), combo = weights, dict()
	if a > 0 and dsd_counts: combo = {go_id:(count*a) for go_id, count in dsd_counts.items()}

	if b > 0 and munk_counts:
		for go_id, count in munk_counts.items():
			combo[go_id] = (count*b) + combo.get(go_id, 0)

	return combo

def poll_neighborhood(neighbors: ndarray, labels: Dict[int, str],
					  test_idxs: ndarray, indexed_vote_dict: Dict[int, CountDict])-> None:
	real_row_idx = test_idxs[len(indexed_vote_dict)]

	votes = chain(*np.vectorize(labels.get)(neighbors).tolist())

	try: iterator = iter(votes); indexed_vote_dict[real_row_idx] = Counter(votes)
	except TypeError: indexed_vote_dict[real_row_idx] = Counter()

def train_test_split(idxs: ndarray, block_size: int, seed: int=None)-> Tuple[ndarray, ndarray]:
	np.random.seed(seed)

	try: iter(idxs)
	except TypeError: print(f'Invalid indices! Cannot split data ...'); exit()

	n = len(idxs)
	if block_size == n: block_size = 1 # equivalent to LOOCV
	test_sub_idxs = np.random.choice(np.arange(n), size=block_size, replace=False) # (m//k)
	train_idxs = np.delete(idxs, test_sub_idxs) # (m(k-1)//k)
	return train_idxs, idxs[test_sub_idxs]

def grid_search(source_GO_labels: Dict[int, str], target_GO_labels: Dict[int, str], dsd_matrix: ndarray,
munk_matrix: ndarray, seed: int, ns: HyperTups, ws: HyperTups, go_dag, term_counts, n_labels: int=None)-> ResultDict:
	m, n_rounds = dsd_matrix.shape[0], 5 # can be changed as needed
	if seed is None: seed = np.random.randint(1000) # can be set to a constant if needed

	train_idxs, valid_idxs = train_test_split(np.arange(m), m//2, seed) # 1/2, 1/2
	test_metrics = dict()

	for r in range(n_rounds):
		train_sub_idxs, test_idxs = train_test_split(train_idxs, m//n_rounds, seed+r)  # (4/5 of 1/2), (1/5 of 1/2)
		munk_test = munk_matrix[test_idxs,:]
		dsd_test  = dsd_matrix[np.ix_(test_idxs, train_sub_idxs)]

		for p,q in ns:
			dsd_test_grid_idxs  = np.argsort(dsd_test, axis=1)[:,:p]
			dsd_test_neighbor_col_idxs  = np.apply_along_axis(np.vectorize(train_sub_idxs.__getitem__), 1, dsd_test_grid_idxs) # (1m/10) x (p)
			munk_test_neighbor_col_idxs = np.argsort(munk_test, axis=1)[:,:q]

			dsd_test_votes, munk_test_votes = dict(), dict()
			np.apply_along_axis(poll_neighborhood, 1, dsd_test_neighbor_col_idxs, target_GO_labels, test_idxs, dsd_test_votes)
			np.apply_along_axis(poll_neighborhood, 1, munk_test_neighbor_col_idxs, source_GO_labels, test_idxs, munk_test_votes)

			for a,b in ws:
				test_predictions = dict()
				for t_idx in test_idxs: # source labels have already been filtered 50-500, so FL(u) guaranteed to be in FL(H) U FL(F) forall u
					d_votes, m_votes = dsd_test_votes.get(t_idx), munk_test_votes.get(t_idx)
					voting_results = wmv(d_votes, m_votes, (a,b)) # {go_label1: count1, go_label2: count2, ...}
					if not bool(voting_results): continue
					tot = (a*sum(d_votes.values())) + (b*sum(m_votes.values()))
					voting_results = {go_id:(c/tot) for go_id, c in voting_results.items()}
					test_predictions[t_idx] = list(sorted(voting_results.items(), key=lambda x: (x[1],x[0]), reverse=True))

				test_metrics[(p,q,a,b,r)] = compute_metrics(test_predictions, target_GO_labels, go_dag, term_counts, n_labels)

	test_accs = {(p,q,a,b):[test_metrics[(p,q,a,b,j)][0] for j in range(n_rounds)]\
								   for p,q,a,b,i in test_metrics.keys() if i == 0}

	P,Q,A,B = sorted([(k,np.array(v).mean(0)) for k,v in test_accs.items()], \
	key=lambda x: (x[1], x[0][0]*x[0][1]), reverse=True)[0][0] # sort by accuracy

	dsd_valid  = dsd_matrix[np.ix_(valid_idxs, train_idxs)]	# (5m/10) x (4m/10)
	munk_valid = munk_matrix[valid_idxs, :] 				# (5m/10) x n

	dsd_valid_grid_idxs = np.argsort(dsd_valid, axis=1)[:,:P]

	dsd_valid_neighbor_col_idxs  = np.apply_along_axis(np.vectorize(train_idxs.__getitem__), 1, dsd_valid_grid_idxs)  # (5m/10) x (p)
	munk_valid_neighbor_col_idxs = np.argsort(munk_valid, axis=1)[:,:Q]

	dsd_valid_votes, munk_valid_votes = dict(), dict()
	np.apply_along_axis(poll_neighborhood, 1, dsd_valid_neighbor_col_idxs, target_GO_labels, valid_idxs, dsd_valid_votes)
	np.apply_along_axis(poll_neighborhood, 1, munk_valid_neighbor_col_idxs, source_GO_labels, valid_idxs, munk_valid_votes)

	valid_predictions = dict()
	for v_idx in valid_idxs: # source labels have already been filtered 50-500, so FL(u) guaranteed to be in FL(H) U FL(F) forall u
		d_votes, m_votes = dsd_valid_votes.get(v_idx), munk_valid_votes.get(v_idx)
		voting_results = wmv(d_votes, m_votes, (A,B)) # {go_label1: count1, go_label2: count2, ...}
		if not bool(voting_results): continue
		tot = (A*sum(d_votes.values())) + (B*sum(m_votes.values()))
		voting_results = {go_id:(c/tot) for go_id, c in voting_results.items()}
		valid_predictions[v_idx] = list(sorted(voting_results.items(), key=lambda x: (x[1],x[0]), reverse=True))

	validation_metrics = compute_metrics(valid_predictions, target_GO_labels, go_dag, term_counts, n_labels)
	print(f'\t\t\t Validation metrics: {validation_metrics}')
	if validation_metrics[0] == -1.0: print('++ EMPTY VALIDATION ++'); exit()

	return test_metrics, {(P,Q,A,B): validation_metrics}

def k_fold_inverted_cv(source_GO_labels: Dict[int, str], target_GO_labels: Dict[int, str],  dsd_matrix: ndarray,
munk_matrix: ndarray, seed: int, k: int, p: int, q: int, wt: Tuple[float, float], go_dag, term_counts, n_labels: int=None)-> ndarray:
	if seed is None: seed = np.random.randint(1000) # can be set to a constant if needed
	m, fold_metrics = dsd_matrix.shape[0], np.zeros(4).astype(float)

	for i in range(abs(k)):
		known_idxs, unknown_idxs = train_test_split(np.arange(m), m//abs(k), seed+i)
		if k < 0 : known_idxs, unknown_idxs = unknown_idxs, known_idxs # cascade setting, for internal BCB use

		# Extracting fold from full matrices
		dsd_fold  = dsd_matrix[np.ix_(unknown_idxs, known_idxs)] 	# (m/k) x (m(k-1)/k)
		munk_fold = munk_matrix[unknown_idxs, :] 					# (m/k) x n

		# Locating neighbor indexes in fold...
		dsd_grid_idxs  = np.argsort(dsd_fold,  axis=1)[:,:p]
		munk_grid_idxs = np.argsort(munk_fold, axis=1)[:,:q]

		# Re-indexing neighbors to match original matrices...
		dsd_neighbor_col_idxs  = np.apply_along_axis(np.vectorize(known_idxs.__getitem__), 1, dsd_grid_idxs) # (m/k) x (p)
		munk_neighbor_col_idxs = munk_grid_idxs # (m/k) x (q)

		# Polling neighbors...
		dsd_votes, munk_votes = dict(), dict()
		np.apply_along_axis(poll_neighborhood, 1, dsd_neighbor_col_idxs, target_GO_labels, unknown_idxs, dsd_votes)
		np.apply_along_axis(poll_neighborhood, 1, munk_neighbor_col_idxs, source_GO_labels, unknown_idxs, munk_votes)

		# Compiling votes and making final predictions...
		a, b = wt
		predictions = dict()
		for u_idx in unknown_idxs: # source labels have already been filtered 50-500, so FL(u) guaranteed to be in FL(H) U FL(F) forall u
			d_votes, m_votes = dsd_votes.get(u_idx), munk_votes.get(u_idx)
			voting_results = wmv(d_votes, m_votes, wt) # {go_label1: count1, go_label2: count2, ...}
			if not bool(voting_results): continue
			tot = (a*sum(d_votes.values())) + (b*sum(m_votes.values()))
			voting_results = {go_id:(c/tot) for go_id, c in voting_results.items()}
			predictions[u_idx] = list(sorted(voting_results.items(), key=lambda x: (x[1],x[0]), reverse=True))

		fmetrics = compute_metrics(predictions, target_GO_labels, go_dag, term_counts, n_labels)
		print('++ EMPTY FOLD ++') if fmetrics[0] == -1.0 else print(f'\t\t{fmetrics}')

		if i == 0: fold_metrics = fmetrics
		else: fold_metrics = np.vstack((fold_metrics, fmetrics))
	return fold_metrics

def main(args: Dict[str, Any], start_cv: bool=True)-> Tuple[ndarray, ndarray, LabelDict, LabelDict]:
	source_go_file, target_go_file, go_aspect, k, seed, p, q, weights, n_labels, job_id, run_id = \
	multiget(args, 'source_go_file', 'target_go_file', 'go_aspect', 'k_fold_size', 'constant_seed',
	'p_nearest_dsd_neighbors', 'q_nearest_munk_neighbors', 'weights', 'n_labels', 'job_id', 'run_id')
	dsd_matrix, munk_matrix = load_matrices(job_id, run_id)

	source_ref_labels = load_labels(job_id, 'source', run_id)
	target_ref_labels = load_labels(job_id, 'target', run_id)

	source_refseq_to_GO_mapping = map_ref_to_GO(source_go_file,
	f"{job_id}/source_refseq_to_uniprot_mapping.json", go_aspect)
	target_refseq_to_GO_mapping = map_ref_to_GO(target_go_file,
	f"{job_id}/target_refseq_to_uniprot_mapping.json", go_aspect)

	source_GO_labels = {i:source_refseq_to_GO_mapping.get(c, set()) for i, c in source_ref_labels.items()} # {i: {go_id1, ...},...}
	target_GO_labels = {i:target_refseq_to_GO_mapping.get(c, set()) for i, c in target_ref_labels.items()} # {i: {go_id1, ...},...}

	annotation_counts = Counter(chain(*target_GO_labels.values()))
	source_GO_labels  = filter_labels(source_GO_labels, annotation_counts) # {i: {go_id1, ...},...}
	target_GO_labels  = filter_labels(target_GO_labels, annotation_counts) # {i: {go_id1, ...},...}

	if not start_cv: return dsd_matrix, munk_matrix, source_GO_labels, target_GO_labels
	return k_fold_inverted_cv(source_GO_labels, target_GO_labels, dsd_matrix, munk_matrix, seed, k, p, q, weights, n_labels)

#python predict_main.py -s ../experiments/GO/goa_human.gaf -t ../experiments/GO/goa_mouse.gaf -a F P -k -2 -c 291 -j ../experiments/humanBIOGRID_mouseBIOGRID/ -r _8-18-20--10-39-AM