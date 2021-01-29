import blastp
import typing
import argparse

def standardize_threshold(thresh: int)-> int:
	if thresh <= 0 : return 0
	if 0 < thresh <= 1: return (thresh * 100)
	return thresh

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("-t", "--target_blast_results_file", help="target blast results filepath", type=str, required=True)
	parser.add_argument("-q", "--query_coverage", help="query coverage threshold (0-100)", type=float, nargs='?', default=90.0)
	parser.add_argument("-p", "--percent_id", help="percent ID threshold (0-100)", type=float, nargs='?', default=85.0)
	parser.add_argument("-j", "--job_id", help="name of directory where hits dict should be saved.", type=str, required=True)
	parser.add_argument("-n", "--strawman_number", help="strawman number", type=str, nargs='?', choices=['1','1+','2','2+','3','3+'], default='1')
	parser.add_argument("-a", "--auto_embed", help="embed networks after parsing BLASTP files (n_random_walks assumed to be -1)", default ='n', choices=['y','n'])
	parser.add_argument("-v", "--verbose", help="print status updates (y/n)", type=str, nargs='?', default ='y', choices=['y','n'])
	
	args = vars(parser.parse_args())

	blastp.core(args)