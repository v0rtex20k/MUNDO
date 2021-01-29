import argparse
import coembedding

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('-w', "--n_random_walks", help="number of random walks for DSD: default is -1 (run until convergence)", type=int, nargs='?', default=-1)
	parser.add_argument('-c', "--compute", help="compute both DSD embeddings, \
														 only the source DSD embedding,\
														 or only the target DSD embedding (b/s/t)", type=str, nargs='?', default='b', choices=['b','s','t'])
	parser.add_argument("-j", "--job_id", help="name of directory where network embeddings should be saved.", type=str, required=True)
	parser.add_argument("-r", "--run_id", help="name of current run - appended to all outputted matrix files", type=str, nargs= '?', default= '')
	parser.add_argument("-n", "--strawman_number", help="strawman number", type=str, nargs='?', choices=['1','1+','2','2+','3','3+'], default='1')
	parser.add_argument("-v", "--verbose", help="print status updates (y/n)", type=str, nargs='?', default = 'y', choices=['y','n'])
	args = vars(parser.parse_args())
	
	coembedding.core(args)
