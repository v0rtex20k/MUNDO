import numpy as np
from scipy.stats import mode
from typing import Callable, List, NewType

Hit = NewType('Instance of Hit class', Callable)

class BlastParser():
	"""Extract relevant alignment data from BLAST text files"""
	def __init__(self, file: str, n_hits: int):
		self._blast_file = file
		self.n_hits = n_hits
		self._hits = self.__get_hits()

	def __get_hits(self: Callable)-> List[List[Hit]]:
		query_idxs, hit_idxs, hits, results = [], [], [], []
		with open(self._blast_file, 'r') as fptr:
			results = fptr.readlines()

		for i, line in enumerate(results):
			if any([s in line.lower() for s in ['query #', 'query id']]): # start of alignment
				query_idxs.append(i)
			elif "no significant similarity found" in line.lower():
				del query_idxs[-1]
			elif 'description' in line.lower():
				j = 0
				while results[i+1+j] != '\n' and j < self.n_hits:
					j += 1
				
				hit_idxs.append([i+j for j in range(1,j+1)])

		for q_idx, h_idxs in zip(query_idxs, hit_idxs):
			query_line, hit_lines = results[q_idx].lstrip().rstrip(), [results[h_idx].lstrip().rstrip() for h_idx in h_idxs]
			query_acc = ''.join(c for c in query_line[query_line.find('ref|'):].split(' ')[0] if not c.islower() and c != '|')
			
			individual_hits = []
			for hit_line in hit_lines:
				hit_info  = [val for val in hit_line.split(' ') if val]
				hit = Hit(query_acc, *hit_info[-7:]) # see message below -
							     					 # *hit_info[-6:] might be needed.
				individual_hits.append(hit)
			hits.append(individual_hits)
		return hits

class Hit():
	def __init__(self, q, ms, ts, qc, ev, pi, acl, h):
		self.query = q
		self.max_score = float(ms)
		self.total_score = float(ts)
		self.query_coverage = float(qc.rstrip('%'))
		self.e_value = float(ev)
		self.percent_id = float(pi)
		self.acc_len = int(acl) # this field is absent in some species' BLAST results
								# for some reason - if there's an error about "cannot
								# convert to <type>", it probably has something to do with this.
		self.match = h
		