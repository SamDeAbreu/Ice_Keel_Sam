'''
Usage:
	merge.py <base_path> [--cleanup=<tf>]
'''

if __name__ == "__main__":

	from docopt import docopt
	from dedalus.tools import logging
	from dedalus.tools import post

	args = docopt(__doc__)
	base_path = args['<base_path>']
	clean_up = bool(args['--cleanup'])
	post.merge_process_files(base_path, cleanup=clean_up)
