#ifndef TESTS_UTILs
#define TESTS_UTILs

#include <cassert>
#include <iostream>
#include <fstream>

#include "../../../fiba_v2_kernels/src/model_components/basic_defs/basic_defs_glue.h"

#if MODEL_ID == MOB_V2
#include "../../../fiba_v2_kernels/src/model_components/model/headers/mob_v2_layers_specs.h"
#elif MODEL_ID == MOB_V2_0_5
#include "../../../fiba_v2_kernels/src/model_components/model/headers/mob_v2_0_5_layers_specs.h"
#elif MODEL_ID == MOB_V2_0_75
#include "../../../fiba_v2_kernels/src/model_components/model/headers/mob_v2_0_75_layers_specs.h"
#elif MODEL_ID == MOB_V2_0_25
#include "../../../fiba_v2_kernels/src/model_components/model/headers/mob_v2_0_25_layers_specs.h"
#endif

using namespace std;

void fill_layer_input_from_file(string file_name, int input_size);

void dump_layer_output(string file_name, fms_dt ofms[max_fms_size], const layer_specs layer_specs_struct);
void dump_layer_output(string file_name, fms_dt ofms[][CHANNELS_TILE_HEIGHT][CHANNELS_TILE_WIDTH],
					   const layer_specs layer_specs_struct);
void dump_layer_output_no_tiling(string file_name, fms_dt ofms[max_fms_size],
								 int ofms_size, const int ofms_h, const int ofms_w);

void dump_pw_pss_tile(string file_name, pss_dt tile[pw_tile_d][pw_tile_h][pw_tile_w]);
void dump_pw_channels_tile(string file_name, fms_dt tile[pw_tile_d][pw_tile_h][pw_tile_w]);
void dump_pw_weights_tile(string file_name,
						  weights_dt tile[pw_conv_parallelism_out][max_conv_d], int layer_depth);
void dump_ouput(string file_name, fms_dt out[], int size);
void read_model_configs(string file_name, int configs_list[]);

string get_model_prefix();


#endif
