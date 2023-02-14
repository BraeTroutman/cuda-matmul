#ifndef ARGUMENTS_H
#define ARGUMENTS_H

typedef struct args {
	char verbose;
	char timed;
	char check;
	char** remaining;
} options_t;

options_t parse_args(int argc, char* argv[]);

#endif
