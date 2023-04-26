#ifndef TESTS_UTILs
#define TESTS_UTILs

#include "../model_components/basic_defs/basic_defs_glue.h"
#include "../model_components/model/headers/model_glue.h"
#include <iostream>
#include <fstream>

using namespace std;

void fill_layer_input_from_file(string file_name, int input_size);

void dump_layer_output(string file_name, fms_dt ofms[max_fms_size], const layer_specs layer_specs_struct);
void dump_layer_output(string file_name, fms_dt ofms[MAX_FMS_BUFFER_DEPTH][MIN_FMS_HEIGHT][MIN_FMS_WIDTH],
					   const layer_specs layer_specs_struct);
void dump_layer_output_no_tiling(string file_name, fms_dt ofms[max_fms_size],
								 int ofms_size, const int ofms_h, const int ofms_w);

void dump_pw_pss_tile(string file_name, pss_dt tile[pw_tile_d][pw_tile_h][pw_tile_w]);
void dump_pw_channels_tile(string file_name, fms_dt tile[pw_tile_d][pw_tile_h][pw_tile_w]);
void dump_pw_weights_tile(string file_name,
						  weights_dt tile[pw_conv_parallelism_out][max_conv_d], int layer_depth);
void dump_ouput(string file_name, fms_dt out[], int size);

string get_model_prefix();


#endif
