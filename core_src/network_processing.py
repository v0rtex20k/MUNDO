import json
from utils import *
import networkx as nx
from itertools import chain
from math import ceil, floor
from urllib import parse, request
from typing import Any, Dict, List, NewType, Tuple, Set

Graph = NewType('nx Graph', nx.Graph)

def largest_connected_component(G: Graph, organism: str)-> Graph:
	if len(G) == 0: print(f'\tEmpty {organism} graph! Make sure correct db name and species name were inputted.'); exit()
	return G.subgraph(max(nx.connected_components(G), key=len))

def extract_fields(p1: int, p2: int, interaction: str, outer_delimiter: str, inner_delimiter: str=None, p_in: int=None)-> Tuple[str, str]:
	interaction = interaction.split(outer_delimiter)
	if not inner_delimiter and not p_in:
		return interaction[p1], interaction[p2]
	return interaction[p1].split(inner_delimiter)[p_in], interaction[p2].split(inner_delimiter)[p_in]

def map_db_to_refseq(node_set: Set[str], db: str, organism: str)-> Tuple[Dict[str, str], Dict[str, str]]:
	input_formats = {'BIOGRID': 'BIOGRID_ID', 'HUMANNET': 'P_ENTREZGENEID', 'BIOPLEX': 'ID', 'LEGACY_BIOGRID': 'GENENAME',
					 'DIP': 'DIP_ID', 'REACTOME': 'ID', 'STRING': 'STRING_ID', 'GIANT': 'P_ENTREZGENEID', 'GENEMANIA': 'ENSEMBL_ID'}
	
	uniprot_acc_params = {'from': input_formats[db], 'to': 'ACC', 'format': 'tab', 'query': ' '.join(node_set)}
	uniprot_id_params = {'from': input_formats[db], 'to': 'ID', 'format': 'tab', 'query': ' '.join(node_set)}
	
	acc_request = request.Request('https://www.uniprot.org/uploadlists/', parse.urlencode(uniprot_acc_params).encode('utf-8'))
	id_request = request.Request('https://www.uniprot.org/uploadlists/', parse.urlencode(uniprot_id_params).encode('utf-8'))
	
	refseq_to_uniprot_mapping, db_to_refseq_mapping = dict(), dict()

	with request.urlopen(acc_request) as accs, request.urlopen(id_request) as ids:
		next(accs); next(ids)
		valid_uniprot_accs = [acc for acc, pid in zip(accs.readlines(), ids.readlines()) if organism.lower() in pid.decode('utf-8').lower()]
		db_to_uniprot_mapping = {db_id:uni_id for db_id, uni_id in [acc.decode('utf-8').strip().split('\t') for acc in valid_uniprot_accs]}

		refseq_params = {'from': 'ACC', 'to': 'P_REFSEQ_AC', 'format': 'tab', 'query': ' '.join(db_to_uniprot_mapping.values())}
		ref_request = request.Request('https://www.uniprot.org/uploadlists/', parse.urlencode(refseq_params).encode('utf-8'))

		with request.urlopen(ref_request) as refs:
			uniprot_to_refseq_mapping = {uni_id:ref_id for uni_id, ref_id in [ref.decode('utf-8').strip().split('\t') for ref in refs]}
			db_to_refseq_mapping = {k:v for k,v in {(db_id, uniprot_to_refseq_mapping.get(uni_id)) for db_id, uni_id in db_to_uniprot_mapping.items()} if v}

	refseq_to_uniprot_mapping = {v:k for k, v in uniprot_to_refseq_mapping.items()}
	return refseq_to_uniprot_mapping, db_to_refseq_mapping

def build_network(dbFile: str, db: str, organism: str)-> Tuple[Dict[str, str], Graph]:
	if not file_exists(dbFile, 'NETWORK PROCESSING'): exit()
	try:
		edgeset = set()
		with open(dbFile, 'r') as fptr:
			next(fptr) # get rid of format desciption line
			for line in fptr.readlines():
				interaction = line.strip()
				if db == 'STRING': src, dst = extract_fields(0, 1, interaction, ' ')
				elif db == 'DIP': src, dst = extract_fields(0, 1, interaction, '\t', '|', 0)
				elif db == 'BIOGRID': src, dst = extract_fields(3, 4, interaction, '\t')
				elif db == 'LEGACY_BIOGRID': src, dst = extract_fields(0, 1, interaction, '\t')
				elif db == 'REACTOME': src, dst = extract_fields(0, 3, interaction, '\t', ':', -1)
				elif db == 'BIOPLEX': src, dst = extract_fields(2, 3, interaction, '\t')
				elif db == 'GIANT': src, dst = extract_fields(0, 1, interaction, '\t') if interaction.split('\t')[2] < 0.90 else ('!','!')
				else: src, dst = extract_fields(0, 1, interaction, '\t') # HumanNet, geneMANIA
				if src == dst: continue # ignore self-loops
				edgeset.add((src, dst))
	
	except ValueError: print(f'[NETWORK PROCESSING ERROR] File \"{file}\" is saved in an invalid format: expected <src> <dst> ...'); exit()

	nodeset = set(chain(*edgeset)) # get unique nodes
	refseq_to_uniprot_mapping, db_to_refseq_mapping = map_db_to_refseq(nodeset, db, organism)
	refseq_edgeset = {e for e in {(db_to_refseq_mapping.get(src), db_to_refseq_mapping.get(dst)) for src, dst in edgeset} if all(e)}
	
	G = nx.Graph()
	G.add_edges_from(refseq_edgeset)

	return refseq_to_uniprot_mapping, largest_connected_component(G, organism)

def save_mapping(mapping: Dict[str, str], network_name: str, job_id: str)-> None:
	with open(f'{job_id}/{network_name}_refseq_to_uniprot_mapping.json', 'w') as mptr:
		json.dump(mapping, mptr)

def save_proteins(proteins: List[str], organism: str, network_name: str, job_id: str, verbose: bool)-> None:
	if not len(proteins): print(f'[NETWORK PROCESSING ERROR] Empty \"{organism.lower()}\" network'); exit()
	if verbose: print(f"\n\tSaving {network_name} network ...\n")

	if network_name == "source" and not os.path.isdir(job_id): os.mkdir(job_id) # create job_id dir if it doesn't exist

	for i in range(0, len(proteins), 500): # do not go over 500 or BLASTP queries will fail !!!
		n = floor((i+500)/500)
		with open(f"{job_id}/{organism}_{network_name}{str(n)}.txt", 'w+') as fptr:
			for node in proteins[i:i+500]:
				fptr.write(node + '\n')
		if verbose: print(f'\t[PROGRESS] {n if n < len(proteins) else len(proteins)}/{ceil(len(proteins)/500)} protein batches saved ...')

def core(args: Dict[str, Any])-> Tuple[Graph, Graph]:
	verbose = boolify(args, 'verbose')
	job_id, (source_org, target_org), (source_db, target_db) = multiget(args, 'job_id', 'organism_names', 'databases_of_origin')
	
	if verbose: print("\tBuilding source network...", end='\t', flush=True)
	src_refseq_to_uniprot_mapping, source = build_network(args['source_network_file'], source_db, source_org)
	if verbose: print(f"Source network built: {len(source)} nodes, {source.size()} edges\n\tBuilding target network...", end='\t')
	tgt_refseq_to_uniprot_mapping, target = build_network(args['target_network_file'], target_db, target_org)
	if verbose: print(f"Target network built: {len(target)} nodes, {target.size()} edges\n\tSorting nodes by degree...")

	src_proteins = [node for node, degree in sorted(source.degree, key=lambda x: x[1], reverse=True)]
	tgt_proteins = [node for node, degree in sorted(target.degree, key=lambda x: x[1], reverse=True)]

	if verbose: print("\tSaving proteins and refseq-uniprot mappings...")
	save_proteins(src_proteins, source_org, "source", job_id, verbose)
	save_proteins(tgt_proteins, target_org, "target", job_id, verbose)
	save_mapping(src_refseq_to_uniprot_mapping, 'source', job_id)
	save_mapping(tgt_refseq_to_uniprot_mapping, 'target', job_id)

	if verbose: print("\n\tPickling networks...")
	nx.write_gpickle(source, f"{job_id}/{source_org}_source_network.gpickle")
	nx.write_gpickle(target, f"{job_id}/{target_org}_target_network.gpickle")
	
	if verbose: print(f'\n\tPickled networks and put them in \"{job_id}\" directory')
