import argparse
import coembedding

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('-p', "--precomputed_id", help="Use precomputed source and target DSD matrices", type=str, nargs='?', default = None)
	parser.add_argument('-c', "--compute", help="compute all three embeddings, \
														 both the munk and target DSD embeddings,\
														 only the munk embedding,\
														 or only the target DSD embedding (a/b/m/d)", type=str, nargs='?', default='b', choices=['a','b','m','d'])
	parser.add_argument('-w', "--n_random_walks", help="number of random walks for DSD: default is -1 (run until convergence)", type=int, nargs='?', default=-1)
	parser.add_argument('-n', "--normalized", help="Use normalized DSD - default is not normalized (y/n). ", type=str, nargs='?', default = 'n', choices=['y','n'])
	parser.add_argument('-g', "--gamma", help="value of gamma to use for RBF Kernel. If compute = \'d\', gamma is ignored", type= float, nargs= '?', default= None)
	parser.add_argument('-t', "--thresh", help="value of threshold to use for RBF Kernel. If compute = \'d\', threshold is ignored", type= float, nargs= '?', default= None)
	parser.add_argument("-j", "--job_id", help="name of directory where network embeddings should be saved.", type=str, required=True)
	parser.add_argument("-r", "--run_id", help="name of current run - appended to all outputted matrix files", type=str, nargs= '?', default= '')
	parser.add_argument("-v", "--verbose", help="print status updates (y/n)", type=str, nargs='?', default = 'y', choices=['y','n'])
	args = vars(parser.parse_args())

	coembedding.core(args)