#include <getopt.h>
#include <iostream>
#include <string.h>

#include "arguments.h"

options_t parse_args(int argc, char* argv[]) {
	options_t options;
	memset(&options, 0, sizeof(options_t));

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

	char** remaining = argv+optind;
	int optslen;
	for (optslen = 0; remaining[optslen] != NULL; ++optslen);

	if (optslen < 3) {
		printf("Usage: %s [options] M N K\n", argv[0]);	
		exit(1);
	}
	
	options.M = atoi(remaining[0]);
	options.N = atoi(remaining[1]);
	options.K = atoi(remaining[2]);	

	return options;
}

