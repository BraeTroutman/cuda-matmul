#include <getopt.h>

#include "arguments.h"

options_t parse_args(int argc, char* argv[]) {
	options_t options;
	options.check = 0;
	options.verbose = 0;
	options.timed = 0;
	
	int c;

	while ((c = getopt(argc, argv, "cvt")) != -1) {
		switch (c) {
			case 'c':
				options.check = 1;
				break;
			case 'v':
				options.verbose = 1;
				break;
			case 't':
				options.timed = 1;
			default:
				break;
		}
	}

	options.remaining = argv+optind;

	return options;
}

