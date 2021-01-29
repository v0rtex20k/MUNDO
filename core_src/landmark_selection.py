import coembedding
from utils import *
import networkx as nx
from blast_tools import BlastParser
from typing import Any, Dict, Generic, List, NewType, Tuple, Set

Graph = NewType('nx Graph', nx.Graph)
Hit = NewType('Hit object defined in blast_tools.py', Generic)

def get_best_hits(filepath: str, query_coverage: float, percent_id: float)-> Dict[str, str]:
	best_hits = dict()
	hits = BlastParser(filepath)._hits

	for hit in hits:
		if hit.query_coverage >= query_coverage and hit.percent_id >= percent_id:
			best_hits[hit.query] = hit.match

	return best_hits

def add_reciprocal_best_hits(source_file: str, target_file: str, query_coverage: float, percent_id: float)-> Set[Tuple[str,str]]:
	if not file_exists(source_file, 'LANDMARK SELECTION'): exit()
	if not file_exists(target_file, 'LANDMARK SELECTION'): exit()

	src_best_hits = get_best_hits(source_file, query_coverage, percent_id)
	tgt_best_hits = get_best_hits(target_file, query_coverage, percent_id)

	reciprocal_best_hits = set()
	for src_query in src_best_hits.keys():
		tgt_match = tgt_best_hits.get(src_best_hits[src_query])
		if tgt_match == src_query:
			tgt_match = src_best_hits[src_query].split('.')[0]
			src_query = src_query.split('.')[0]
			reciprocal_best_hits.add((src_query, tgt_match))

	return reciprocal_best_hits

def existing_reciprocal_best_hits(job_id: str)-> int:
	if not file_exists(f'{job_id}/reciprocal_best_hits.txt', 'LANDMARK', True): return 0
	with open(f'{job_id}/reciprocal_best_hits.txt', 'r') as rptr:
		return len(rptr.readlines())

def save_reciprocal_best_hits(reciprocal_best_hits: Set[Tuple[str,str]], job_id)-> None:
	with open(f'{job_id}/reciprocal_best_hits.txt', 'a') as rptr:
		for src_query, tgt_match in reciprocal_best_hits:
			rptr.write(f'{src_query}\t{tgt_match}\n')

def core(args: Dict[str, Any])-> None:
	verbose = boolify(args, 'verbose')
	job_id, query_coverage, percent_id, source_blast_file, target_blast_file = \
	multiget(args, 'job_id', 'query_coverage', 'percent_id', 'source_blast_results_file', 'target_blast_results_file')

	if not job_exists(job_id, 'LANDMARK'): exit()
	n_rbh = existing_reciprocal_best_hits(job_id)

	if verbose: print(f"\t{n_rbh} reciprocal best hits have been identified so far.\n\t\t~ Expanding ~", flush=True)
	reciprocal_best_hits = add_reciprocal_best_hits(source_blast_file, target_blast_file, query_coverage, percent_id)
	save_reciprocal_best_hits(reciprocal_best_hits, job_id)
	
	if verbose: print(f"A total of {n_rbh + len(reciprocal_best_hits)} reciprocal best hits now exist in {job_id}/reciprocal_best_hits.txt")

	if args['auto_embed'] == 'y':
		params = {'rbh_file': rbh_file, 'n_random_walks': -1, 'job_id': job_id, 'verbose': 'y' if verbose else 'n'}
		coembedding.core(params)

