import predict
import argparse

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('-s', "--source_go_file", help="Source GO file path", type=str, required=True)
	parser.add_argument('-t', "--target_go_file", help="Target GO file path", type=str, required=True)
	parser.add_argument('-a', "--go_aspect", help="All, Biological, Molecular, Cellular (A/P/F/C)", type=str, nargs='*', default = ['F','P'], choices=['A','P','F','C'])
	parser.add_argument('-k', "--k_fold_size", help="Size of fold for k-fold CV", type=int, nargs='?', default=-4)
	parser.add_argument('-c', "--constant_seed", help="Random seed for k-fold cv", type=int, nargs='?', default=None)
	parser.add_argument('-p', "--p_nearest_dsd_neighbors", help="DSD voting pool size", type=int, nargs='?', default=10)
	parser.add_argument('-q', "--q_nearest_munk_neighbors", help="munk voting pool size", type=int, nargs='?', default=10)
	parser.add_argument('-w', "--weights", help="DSD and munk vote weights", type=float, nargs='*', default=[1.0,1.0])
	parser.add_argument('-l', "--n_labels", help="number of predicted labels used to compute metrics", type=int, nargs='?', default=None)
	parser.add_argument("-j", "--job_id", help="Name of directory where matrices and labels were saved.", type=str, required=True)
	parser.add_argument("-r", "--run_id", help="Auto-select matrices and labels based on suffixs", type=str, required=True)
	args = vars(parser.parse_args())

	print(predict.main(args))
