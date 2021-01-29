import json
import coembedding
from utils import *
import networkx as nx
from blast_tools import BlastParser
from typing import Any, Callable, Dict, List, NewType, Tuple, Set

Graph = NewType('nx Graph', nx.Graph)
Hit = NewType('Hit object defined in blast_tools.py', Callable)

def meets_threshold(hit: Hit, query_coverage: float, percent_id: float)-> bool:
	return True if (hit.query_coverage >= query_coverage and hit.percent_id >= percent_id) else False

def get_best_hits(target_file: str, query_coverage: float, percent_id: float, n_hits: int)-> Dict[str, Set[str]]:
	if not file_exists(target_file, 'STRAWMAN-BLASTP'): exit()
	
	best_hits = dict()
	all_hits = BlastParser(target_file, n_hits)._hits

	for individual_hits in all_hits:
		if not individual_hits: continue
		best_hits[individual_hits[0].query] = {hit.match for hit in individual_hits if meets_threshold(hit, query_coverage, percent_id)}

	best_hits = {target_query:source_match for target_query, source_match in best_hits.items() if source_match}
	return best_hits

def existing_hits(job_id: str, strawman_number: str)-> int:
	if not file_exists(f'{job_id}/strawman{strawman_number}_hits.txt', 'STRAWMAN-BLASTP', quiet=True): return 0
	with open(f'{job_id}/strawman{strawman_number}_hits.txt', 'r') as sptr:
		return len(sptr.readlines())

def save_hits(hits: Set[Tuple[str,str]], job_id: str, strawman_number: str)-> None:
	with open(f'{job_id}/strawman{strawman_number}_hits.txt', 'a') as sptr:
		for k,v in hits.items():
			sptr.write(f'{k}\t{list(v)}\n')

def core(args: Dict[str, Any])-> None:
	verbose = boolify(args, 'verbose')
	target_blast_file, strawman_number, query_coverage, percent_id, job_id = \
	multiget(args, 'target_blast_results_file', 'strawman_number', 'query_coverage', 'percent_id', 'job_id')

	n_hits = 1 if '1' in strawman_number else 500
	percent_id = percent_id if '3' in strawman_number else 0.0
	query_coverage = query_coverage if '3' in strawman_number else 0.0

	if not job_exists(job_id, 'STRAWMAN-BLASTP'): exit()
	n_existing_hits = existing_hits(job_id, strawman_number)

	if verbose: print(f"\t{n_existing_hits} hits have been identified so far.\n\t\t~ Expanding ~", flush=True)
	hits = get_best_hits(target_blast_file, query_coverage, percent_id, n_hits)
	
	if verbose: print("\tSaving hits...")
	save_hits(hits, job_id, strawman_number)
	
	if verbose: print(f"A total of {n_existing_hits + len(hits)} successful query results now exist in {job_id}/strawman{strawman_number}_hits.txt")

	if args['auto_embed'] == 'y':
		params = {'job_id': job_id, 'run_id': '_auto', 'strawman_number': strawman_number, 'verbose': 'y' if verbose else 'n'}
		coembedding.core(params)
