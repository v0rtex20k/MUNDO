import typing
import argparse
import landmark_selection

def standardize_threshold(thresh: int)-> int:
	if thresh < 0 : thresh *= 0
	if not thresh >= 1: thresh *= 100
	return thresh

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("-s", "--source_blast_results_file", help="source blast results filepath", type=str, required=True)
	parser.add_argument("-t", "--target_blast_results_file", help="target blast results filepath", type=str, required=True)
	parser.add_argument("-q", "--query_coverage", help="query Coverage threshold (0-100)", type=float, nargs='?', default=90.0)
	parser.add_argument("-p", "--percent_id", help="percent ID threshold (0-100)", type=float, nargs='?', default=85.0)
	parser.add_argument("-j", "--job_id", help="name of directory where reciprocal_best_hits file should be saved.", type=str, required=True)
	parser.add_argument("-a", "--auto_embed", help="embed networks after parsing BLASTP files", default ='n', choices=['y','n'])
	parser.add_argument("-v", "--verbose", help="print status updates (y/n)", type=str, nargs='?', default ='y', choices=['y','n'])

	args = vars(parser.parse_args())
	args['query_coverage'], args['percent_id'] = standardize_threshold(args['query_coverage']), standardize_threshold(args['percent_id'])

	landmark_selection.core(args)