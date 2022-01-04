import os
from typing import Any, Dict, List

def boolify(args: Dict[str, Any], field: str)-> bool:
	return True if args.get(field) == 'y' else False

def multiget(args: Dict[str, Any], *fields: List[str])-> List[Any]:
	vals = [args.get(field, 'field_not_found') for field in fields]
	for i,v in enumerate(vals):
		if v == 'field_not_found': print('[MULTIGET ERROR]: Invalid field'); exit()
		if isinstance(v, str):
			vals[i] = v.strip(' /\\')
	return vals

def file_exists(filepath: str, origin: str, ext: str=None, quiet: bool=False)-> bool:
	if not os.path.isfile(filepath):
		if not quiet:
			print(f'[{origin.upper()} ERROR] File not found at location \"{filepath}\"')
		return False
	elif ext and not filepath.endswith(ext): 
		if not quiet:
			print(f'[{origin.upper()} ERROR] Invalid extension for \"{filepath}\" (expected {ext})')
		return False
	return True

def find_file_variant(job_id: str, origin: str, base_substring: str, ext: str=None, quiet: bool=False)-> str:
	for file in os.listdir(job_id):
		if file.endswith(ext) and base_substring.lower() in file.lower():
				return file
	if not quiet:
		print(f'[{origin.upper()} ERROR] No suitable file variant found in \"{job_id}\"')
	return None

def job_exists(job_id: str, origin: str, quiet: bool=False)-> bool:
	if not os.path.isdir(job_id):
		if not quiet:
			print(f'[{origin.upper()} ERROR] Job ID \"{job_id}\" does not exist')
		return False
	return True