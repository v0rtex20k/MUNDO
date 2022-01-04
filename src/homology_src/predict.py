import ast
from utils import *
from metrics import compute_metrics

DiGraph   = NewType('nx DiGraph', nx.DiGraph)
CountDict = TypeVar('result of Counter', Dict[str, int], Dict[str, float])
HyperTups = TypeVar('List of tuples of hyperparameters', List[Tuple[float, float, float, float]], List[Tuple[int, int, int, int]])
LabelDict = NewType('dictionary of {id: set of all its GO labels}', Dict[str, Set[str]])

def load_matrices(job_id: str, run_id: str)-> Tuple[ndarray, ndarray]:
	if not file_exists(f"{job_id}/target_dsd_matrix{run_id}.npy", 'PREDICTION', ext='.npy'):  exit()
	if not file_exists(f"{job_id}/source_dsd_matrix{run_id}.npy", 'PREDICTION', ext='.npy'):  exit()
	source_dsd_matrix = np.load(f"{job_id}/source_dsd_matrix{run_id}.npy", allow_pickle=True)
	target_dsd_matrix = np.load(f"{job_id}/target_dsd_matrix{run_id}.npy", allow_pickle=True)

	return source_dsd_matrix, target_dsd_matrix

def load_labels(job_id: str, network_name: str, strawman_number: str, run_id: str)-> Dict[str, str]:
	fillin, post ='$', f'{network_name}_labels{run_id}.json'
	if file_exists(f"{job_id}/strawman_{post}",'PREDICTION',ext='.json'): fillin = ''
	elif file_exists(f"{job_id}/strawman{strawman_number}_{post}", 'PREDICTION', ext='.json'): fillin = strawman_number
	if fillin == '$': print('[PREDICTION ERROR] Invalid label path'); exit()

	labels = None
	with open(f"{job_id}/strawman{fillin}_{post}", "r") as lptr:
		labels = json.load(lptr)

	return {int(k):v for k,v in labels.items()}

def load_and_index_hits(source_ref_labels: Dict[str, str], target_ref_labels: Dict[str, str], job_id: str, strawman_number: str)-> Dict[int, List[int]]:
	if not file_exists(f'{job_id}/strawman{strawman_number}_hits.txt', 'PREDICTION'): exit()
	srf, trf, hits = None, None, dict()
	with open(f'{job_id}/strawman{strawman_number}_hits.txt', 'r') as rptr:
		for line in rptr.readlines():
			a,b = line.split('\t')
			hits[a] = ast.literal_eval(b)

	sample_tk, sample_tv = list(target_ref_labels.items())[0]
	sample_sk, sample_sv = list(target_ref_labels.items())[0]
	if isinstance(sample_sk, (int, float)) and isinstance(sample_sv, str) and\
	   isinstance(sample_tk, (int, float)) and isinstance(sample_tv, str):
		srf = {v:k for k,v in source_ref_labels.items()}
		trf = {v:k for k,v in target_ref_labels.items()} # ensure compatibility
	else: srf, trf = source_ref_labels, target_ref_labels

	hit_idxs = dict()
	for target_query, hitlist in hits.items():
		hit_idxs[trf.get(target_query)] = [srf.get(source_match) for source_match in hitlist]

	return {k:v for k,v in hit_idxs.items() if k and v}

def map_ref_to_GO(go_file: str, ref_file: str, go_aspect: str, intermediate_mapping_file: str=None)-> LabelDict:
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
			uniprot_to_GO_mapping = {k:v for k,v in {inter_mapping.get(k,None):v
										 for k,v in uniprot_to_GO_mapping.items()}.items() if k}

	refseq_to_GO_mapping = {r:g for r,g in {ref_id:uniprot_to_GO_mapping.get(uni_id)
								for ref_id, uni_id in refseq_to_uniprot_mapping.items()}.items() if g}

	return refseq_to_GO_mapping

def filter_labels(tgt_GO_labels: Dict[int, str], annotation_counts: CountDict, low: int = 50, high: int = 500)-> LabelDict:
	filtered_tgt_labels = dict()
	print(f'\t\tFiltering between {low} and {high}...')
	for i, go_ids in tgt_GO_labels.items():
		for go_id in go_ids:
			n_annotations = annotation_counts.get(go_id, -1) # n_annotations from TARGET networks
			if not low <= n_annotations <= high: continue
			filtered_tgt_labels[i] = filtered_tgt_labels.get(i, set()) | {go_id}
		if not filtered_tgt_labels.get(i):
			filtered_tgt_labels[i] = set() # fill in missing ones with empties - they just have no vote.

	return filtered_tgt_labels # {i: {go_id1, go_id2, ...}, ...}

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

	try: iterator = iter(votes); indexed_vote_dict[real_row_idx] = Counter(votes)
	except TypeError: indexed_vote_dict[real_row_idx] = Counter()

def poll_hits(idxs: ndarray, hit_idxs: Dict[int,int], source_dsd_matrix: ndarray,
			  source_GO_labels: Dict[int, str], q: int, strawman_number: str)-> CountDict:

	hit_votes = dict()

	for idx in idxs:
		hitlist = hit_idxs.get(idx, None)
		if hitlist is None: hit_votes[idx] = dict(); continue

		for source_match_idx in hitlist:
			match_votes = Counter(source_GO_labels.get(source_match_idx))

			if '+' in strawman_number and match_votes is not None:
				match_n_idxs = np.argsort(source_dsd_matrix[source_match_idx, :])[:q]
				match_votes |= Counter(chain(*[source_GO_labels.get(midx,[]) for midx in match_n_idxs]))

			hit_votes[idx] = match_votes

	return hit_votes

def train_test_split(idxs: ndarray, block_size: int, seed: int=None)-> Tuple[ndarray, ndarray]:
	np.random.seed(seed)

	try: iter(idxs)
	except TypeError: print(f'Invalid indices! Cannot split train and test sets...'); exit()

	n = len(idxs)

	if block_size == n: block_size = 1 # equivalent to LOOCV
	test_sub_idxs = np.random.choice(np.arange(n), size=block_size, replace=False) # (m//k)
	train_idxs = np.delete(idxs, test_sub_idxs) # (m(k-1)//k)
	return train_idxs, idxs[test_sub_idxs]

def grid_search(source_GO_labels: Dict[int, str], target_GO_labels: Dict[int, str], hit_idxs: ndarray,
				source_dsd_matrix: ndarray, target_dsd_matrix: ndarray, seed: int, hierarchies: Union[List[DiGraph], Dict[str, int]],
			 	ns: HyperTups, ws: HyperTups, strawman_number: str, cutoff: int, n_labels: int=None)-> Union[Dict[Tuple, ndarray], Dict[Tuple, ndarray]]:
	print(f'Starting grid_search w/ a cutoff of {cutoff}')
	m, n_rounds = target_dsd_matrix.shape[0], 5 # can be changed as needed
	if seed is None: seed = np.random.randint(1000) # can be set to a constant if needed

	train_idxs, valid_idxs = train_test_split(np.arange(m), m//2, seed) # 1/2, 1/2
	test_metrics = dict()

	for i in range(n_rounds):
		train_sub_idxs, test_idxs = train_test_split(train_idxs, m//n_rounds, seed+i)  # (4/5 of 1/2), (1/5 of 1/2)
		target_test = target_dsd_matrix[np.ix_(test_idxs, train_sub_idxs)]     		   # (m/k) x (m(k-1)/k)

		for p,q in ns:
			target_test_grid_idxs = np.argsort(target_test, axis=1)[:,:p]
			target_test_neighbor_col_idxs = np.apply_along_axis(np.vectorize(train_sub_idxs.__getitem__), 1, target_test_grid_idxs) # (m/k) x (p)

			target_test_votes = dict()
			np.apply_along_axis(poll_neighborhood, 1, target_test_neighbor_col_idxs, target_GO_labels, test_idxs, target_test_votes)

			hit_test_votes = poll_hits(test_idxs, hit_idxs, source_dsd_matrix, source_GO_labels, q, strawman_number)

			for a,b in ws:
				test_predictions = dict()
				for t_idx in test_idxs: # source labels have already been filtered 50-500, so FL(u) guaranteed to be in FL(H) U FL(F) forall u
					t_votes, h_votes = target_test_votes.get(t_idx), hit_test_votes.get(t_idx)
					voting_results = wmv(t_votes, h_votes, (a,b)) # {go_label1: count1, go_label2: count2, ...}
					if not bool(voting_results): continue
					tot = (a*sum(t_votes.values())) + (b*sum(h_votes.values()))
					voting_results = {go_id:(c/tot) for go_id, c in voting_results.items()}
					test_predictions[t_idx] = list(sorted(voting_results.items(), key=lambda x: (x[1],x[0]), reverse=True))

				test_metrics[(p,q,a,b,i)] = compute_metrics(test_predictions, target_GO_labels, hierarchies, n_labels, cutoff)

	test_accs = {(p,q,a,b):[test_metrics[(p,q,a,b,j)][0] for j in range(n_rounds)]\
								   for p,q,a,b,i in test_metrics.keys() if i == 0}

	P,Q,A,B = sorted([(k,np.array(v).mean(0)) for k,v in test_accs.items()], \
	key=lambda x: (x[1], x[0][0]*x[0][1]), reverse=True)[0][0] # sort by accuracy

	print(f'Done testing --> got ({P},{Q},{A},{B})')

	target_valid = target_dsd_matrix[np.ix_(valid_idxs, train_idxs)]	# (5m/10) x (4m/10)
	target_valid_grid_idxs = np.argsort(target_valid, axis=1)[:,:P]

	target_valid_neighbor_col_idxs = np.apply_along_axis(np.vectorize(train_idxs.__getitem__), 1, target_valid_grid_idxs) # (m/k) x (p)

	target_valid_votes = dict()
	np.apply_along_axis(poll_neighborhood, 1, target_valid_neighbor_col_idxs, target_GO_labels, valid_idxs, target_valid_votes)
	hit_valid_votes = poll_hits(valid_idxs, hit_idxs, source_dsd_matrix, source_GO_labels, Q, strawman_number)

	valid_predictions = dict()
	for v_idx in valid_idxs: # source labels have already been filtered 50-500, so FL(u) guaranteed to be in FL(H) U FL(F) forall u
		t_votes, h_votes = target_valid_votes.get(v_idx), hit_valid_votes.get(v_idx)
		voting_results = wmv(t_votes, h_votes, (A,B)) # {go_label1: count1, go_label2: count2, ...}
		if not bool(voting_results): continue
		tot = (A*sum(t_votes.values())) + (B*sum(h_votes.values()))
		voting_results = {go_id:(c/tot) for go_id, c in voting_results.items()}
		valid_predictions[v_idx] = list(sorted(voting_results.items(), key=lambda x: (x[1],x[0]), reverse=True))

	validation_metrics = compute_metrics(valid_predictions, target_GO_labels, hierarchies, n_labels, cutoff)
	if validation_metrics[0] == -1.0: print('++ EMPTY VALIDATION ++'); exit()

	return test_metrics, {(P,Q,A,B): validation_metrics}

def k_fold_inverted_cv(source_GO_labels: Dict[int, str], target_GO_labels: Dict[int, str], hit_idxs: ndarray,
					   source_dsd_matrix: ndarray, target_dsd_matrix: ndarray, seed: int, hierarchies: List[DiGraph],
					   k: int, p: int, q: int, wt: Tuple[float, float], strawman_number: str, cutoff: int, n_labels: int=None)-> ndarray:
	print(f'Starting kCV w/ a cutoff of {cutoff}')
	if seed is None: seed = np.random.randint(1000) # can be set to a constant if needed
	m, fold_metrics = target_dsd_matrix.shape[0], np.zeros(4).astype(float)
	for i in range(abs(k)):
		known_idxs, unknown_idxs = train_test_split(np.arange(m), m//abs(k), seed+i)
		if k < 0 : known_idxs, unknown_idxs = unknown_idxs, known_idxs # cascade setting, for internal BCB use

		target_fold = target_dsd_matrix[np.ix_(unknown_idxs, known_idxs)] # (m/k) x (m(k-1)/k)

		target_grid_idxs = np.argsort(target_fold, axis=1)[:,:p]
		target_neighbor_col_idxs = np.apply_along_axis(np.vectorize(known_idxs.__getitem__), 1, target_grid_idxs) # (m/k) x (p)

		target_votes = dict()
		np.apply_along_axis(poll_neighborhood, 1, target_neighbor_col_idxs, target_GO_labels, unknown_idxs, target_votes)

		hit_votes = poll_hits(unknown_idxs, hit_idxs, source_dsd_matrix, source_GO_labels, q, strawman_number)

		predictions = dict()
		a,b = wt
		for u_idx in unknown_idxs: # source labels have already been filtered 50-500, so FL(u) guaranteed to be in FL(H) U FL(F) forall u
			t_votes, h_votes = target_votes.get(u_idx), hit_votes.get(u_idx)
			voting_results = wmv(t_votes, h_votes, wt) # {go_label1: count1, go_label2: count2, ...}
			if not bool(voting_results): continue
			tot = (a*sum(t_votes.values())) + (b*sum(h_votes.values()))
			voting_results = {go_id:(c/tot) for go_id, c in voting_results.items()}
			predictions[u_idx] = list(sorted(voting_results.items(), key=lambda x: (x[1],x[0]), reverse=True))

		if i == 0: fold_metrics = compute_metrics(predictions, target_GO_labels, hierarchies, n_labels, cutoff)
		else: fold_metrics = np.vstack((fold_metrics, compute_metrics(predictions, target_GO_labels, hierarchies, n_labels, cutoff)))
	return fold_metrics

def core(args: Dict[str, Any], start_cv: bool=True)-> Tuple[ndarray, ndarray, LabelDict, LabelDict]:
	source_go_file, target_go_file, go_aspect, k, seed, p, q, w, l, job_id, run_id, strawman_number = \
	multiget(args, 'source_go_annotations_file', 'target_go_annotations_file', 'go_aspect', 'k_fold_size', 'constant_seed',
	'p_nearest_target_neighbors', 'q_nearest_source_neighbors', 'weights', 'n_labels', 'job_id', 'run_id', 'strawman_number')

	source_dsd_matrix, target_dsd_matrix = load_matrices(job_id, run_id)

	source_ref_labels = load_labels(job_id, 'source', strawman_number, run_id) # {src_node: i,...}
	target_ref_labels = load_labels(job_id, 'target', strawman_number, run_id) # {tgt_node: i,...}

	hit_idxs = load_and_index_hits(source_ref_labels, target_ref_labels, job_id, strawman_number) # {t_i: s_i, ...}

	source_refseq_to_GO_mapping = map_ref_to_GO(source_go_file, f"{job_id}/source_refseq_to_uniprot_mapping.json", go_aspect)
	target_refseq_to_GO_mapping = map_ref_to_GO(target_go_file, f"{job_id}/target_refseq_to_uniprot_mapping.json", go_aspect)

	source_GO_labels = {i:source_refseq_to_GO_mapping.get(c, set()) for i, c in source_ref_labels.items()} # {i: {go_id1, ...},...}
	target_GO_labels = {i:target_refseq_to_GO_mapping.get(c, set()) for i, c in target_ref_labels.items()} # {i: {go_id1, ...},...}

	annotation_counts = Counter(chain(*target_GO_labels.values()))

	source_GO_labels  = filter_labels(source_GO_labels, annotation_counts) # {i: {go_id1, ...},...}
	target_GO_labels  = filter_labels(target_GO_labels, annotation_counts) # {i: {go_id1, ...},...}

	if not start_cv: return source_dsd_matrix, target_dsd_matrix, source_GO_labels, target_GO_labels, hit_idxs
	return k_fold_inverted_cv(source_GO_labels, target_GO_labels, hit_idxs, source_dsd_matrix,
							  target_dsd_matrix, k, seed, p, q, wt, n_labels, strawman_number)

def compute_pdist(job_id: str, run_id: str)-> None:
	print('\tComputing source pdist matrix...')
	source_dsd_matrix = np.load(f"{job_id}/source_dsd_matrix{run_id}.npy", allow_pickle=True)
	source_pdist  = pairwise_distance_matrix(source_dsd_matrix)
	np.save(f"{job_id}/source_dsd_pdist_matrix{run_id}.npy", source_pdist)
