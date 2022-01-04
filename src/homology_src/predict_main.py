import argparse
from predict import *
from typing import Dict, List, Tuple
from itertools import combinations, permutations, product

def standardize_weights(weights: List[float])-> List[float]:
	if not all(weights): pass
	if len(weights) == 2: return weights
	elif len(weights) == 1: return weights * 2
	print('[PREDICTION ERROR] Invalid format for weights.'); exit()

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('-m', "--matrix_files", help="source and target DSD matrix file paths", type=str, nargs=2, default=[None, None])
	parser.add_argument('-s', "--source_go_annotations_file", help="source go annotations file path", type=str, required=True)
	parser.add_argument('-t', "--target_go_annotations_file", help="target go annotations file path", type=str, required=True)
	parser.add_argument('-a', "--go_aspect", help="all, biological process, molecular function, cellular component (A/P/F/C)", type=str, nargs='*', default = ['A'], choices=['A','P','F','C'])
	parser.add_argument('-k', "--k_fold_size", help="Size of fold for standard k-fold cv (1 = LOOCV)", type=int, nargs='?', default=5)
	parser.add_argument('-c', "--constant_seed", help="Set random state for k-fold cv (y/n)", type=int, nargs='?', default=None)
	parser.add_argument('-p', "--p_nearest_target_neighbors", help="target DSD voting pool size", type=int, nargs='?', default=10)
	parser.add_argument('-q', "--q_nearest_source_neighbors", help="source DSD voting pool size", type=int, nargs='?', default=10)
	parser.add_argument('-w', "--weights", help="source and target vote weights: if single value entered, used for both", type=float, nargs='*', default=[1.0,1.0])
	parser.add_argument('-l', "--n_labels", help="n_labels predicted for each target protein", type=int, nargs='?', default=1)
	parser.add_argument("-j", "--job_id", help="name of directory where network embeddings should be saved.", type=str, required=True)
	parser.add_argument("-r", "--run_id", help="select matrix and label files based on coembedding run_id", type=str, nargs= '?', default='')
	parser.add_argument("-n", "--strawman_number", help="strawman number", type=str, nargs='?', choices=['1','1+','2','2+','3','3+'], default='1')
	parser.add_argument("-v", "--verbose", help="print status updates (y/n)", type=str, nargs='?', default = 'y', choices=['y','n'])
	args = vars(parser.parse_args())

	args['weights'] = standardize_weights(args['weights'])

	core(args)