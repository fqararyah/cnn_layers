#ifndef TESTS_UTILs
#define TESTS_UTILs

#include "../basic_defs/basic_defs_glue.h"
#include "../model/model_glue.h"
#include <iostream>
#include <fstream>

using namespace std;

void fill_layer_input_from_file(string file_name, int input_size);

void dumb_layer_output(string file_name, fms_dt ofms[max_fms_size], int ofms_size);

#endif