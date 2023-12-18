#ifndef PREPARE_WEIGHTS_INPUTS
#define PREPARE_WEIGHTS_INPUTS

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

#include "../../../fiba_v2_kernels/src/model_components/model/pipelined_engines/pipelined_engines_specs.h"

#include <fstream>
#include <iostream>
#include <cassert>

using namespace std;
using namespace pipelined_engines;

int get_num_of_pw_weights(string file_name);

void load_weights(string file_name,
				  weights_dt weights[]);

void load_image(string file_name,
				fms_dt image[]);

void load_and_quantize_image(string file_name,
							 fms_dt image[], Quantization_layer_specs quantization_l_specs);

void glue_weights(string file_name,
				  weights_grp_dt glued_weights[all_pw_s_weights]);

void validate_weights(string file_name,
					  weights_grp_dt glued_weights[all_pw_s_weights]);

void glue_input_image(string file_name,
					  fms_grp_dt input_image[input_image_depth * input_image_height * input_image_width / input_image_group_items]);

void glue_and_quantize_input_image(string file_name,
								   fms_grp_dt input_image[input_image_depth * input_image_num_fms_groups_in_a_channel],
								   Quantization_layer_specs quantization_l_specs);

void verify_glued_image(string file_name,
						fms_grp_dt input_image[input_image_depth * input_image_height * input_image_width / input_image_group_items]);

void fill_input_image(string file_name,
					  fms_dt input_image[input_image_depth][input_image_height][input_image_width]);

void verify_input_image(string file_name,
						fms_dt input_image[input_image_depth][input_image_height][input_image_width]);

void fill_layer_input(string file_name, fms_dt layer_input[max_fms_size],
					  const layer_specs layer_specs_struct);

void verify_fill_layer_input(string file_name, fms_dt ofms[max_fms_size],
							 const layer_specs layer_specs_struct);

// V2
void fill_layer_input(string file_name, fms_dt layer_input[MAX_FMS_BUFFER_DEPTH][MIN_FMS_HEIGHT][MIN_FMS_WIDTH],
					  const layer_specs layer_specs_struct);

void verify_fill_layer_input(string file_name, fms_dt layer_input[MAX_FMS_BUFFER_DEPTH][MIN_FMS_HEIGHT][MIN_FMS_WIDTH],
							 const layer_specs layer_specs_struct);

#if HW == CPU
void fill_pipe_layer_input_buffer(string file_name, fms_dt channels_buffer[MAX_PW_BUFFER_DEPTH][PW_BUFFER_HEIGHT][MAX_PW_BUFFER_WIDTH],
								  const int starting_h, const int start_filling_offset_in_buffer,
								  const layer_specs layer_specs_struct);
#endif

void glue_on_chip_weights_cpu(string file_name,
							  weights_grp_dt glued_on_chip_weights[all_on_chip_pw_s_weights_groups]);

#endif
