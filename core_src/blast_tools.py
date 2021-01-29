import numpy as np
from scipy.stats import mode

class BlastParser():
	"""Extract relevant alignment data from BLAST text files"""
	def __init__(self, file):
		self._blast_file = file
		self._hits = self.__get_hits()

	def __get_hits(self):
		query_idxs, hit_idxs, hits, results = [], [], [], []
		with open(self._blast_file, 'r') as fptr:
			results = fptr.readlines()

		for i, line in enumerate(results):
			if any([s in line.lower() for s in ['query #', 'query id']]): # start of alignment
				query_idxs.append(i)
			elif "no significant similarity found" in line.lower():
				del query_idxs[-1]
			elif 'description' in line.lower():
				hit_idxs.append(i+1)

		gap = mode(np.array([abs(hit_idxs[i] - query_idxs[i]) for i in range(len(list(zip(query_idxs, hit_idxs))))]))[0][0]

		for q_idx, h_idx in zip(query_idxs, hit_idxs):
			if abs(h_idx - q_idx) != gap: continue # try and eliminate mismatches - all query-hit pairs should be same distance apart
			query_line, hit_line = results[q_idx].lstrip().rstrip(), results[h_idx].lstrip().rstrip()
			query_acc = ''.join(c for c in query_line[query_line.find('ref|'):].split(' ')[0] if not c.islower() and c != '|')
			hit_info  = [val for val in hit_line.split(' ') if val]
			hit = Hit(query_acc, *hit_info[-6:])
			hits.append(hit)
		
		return hits

class Hit():
	def __init__(self, q, ms, ts, qc, ev, pi, h):
		self.query = q
		self.max_score = float(ms)
		self.total_score = float(ts)
		self.query_coverage = float(qc.rstrip('%'))
		self.e_value = float(ev)
		self.percent_id = float(pi)
		# self.acc_len = int(acl) # some species have this, for some reason.
		# if yours does, add this as an arg to init() and change line 30 to
		# ... *hit_info[-7:] ...
		self.match = h
		