import argparse
import network_processing

if __name__ == '__main__':

	supported_organisms = ['HUMAN', 'DROME', 'MOUSE',
						   'RAT', 'ARATH', 'CHICK',
						   'DICTY', 'CANLF', 'PIG',
						   'CAEEL', 'YEAST', 'SCHPO']
	supported_databases = ['BIOGRID', 'BIOPLEX', 'DIP',
						   'GENEMANIA', 'GIANT', 'HUMANNET',
						   'LEGACY_BIOGRID','REACTOME', 'STRING']

	parser = argparse.ArgumentParser()
	parser.add_argument("-s", "--source_network_file", help="source network file path.", required=True, type=str)
	parser.add_argument("-t", "--target_network_file", help="target network file path.", required=True, type=str)
	parser.add_argument("-o", "--organism_names", help="uniprot ID for each organism.", required=True, type=str, nargs=2, choices=supported_organisms)
	parser.add_argument("-d", "--databases_of_origin", help="Format of each interaction file.", required=True, type=str, nargs=2, choices=supported_databases)
	parser.add_argument("-v", "--verbose", help="print status updates (y/n)", type=str, nargs='?', default = 'y', choices=['y','n'])
	parser.add_argument("-j", "--job_id", help="name of directory where BLASTP files should be saved.", type=str, required=True)
	args = vars(parser.parse_args())

	network_processing.core(args)