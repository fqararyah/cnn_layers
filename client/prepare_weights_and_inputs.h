#ifndef PREPARE_WEIGHTS_INPUTS
#define PREPARE_WEIGHTS_INPUTS
#include "../model_components/basic_defs/basic_defs_glue.h"
#include "../model_components/model/headers/model_glue.h"
#include <fstream>
#if FPGA
#include "ap_int.h"
#endif
#include <iostream>
#include <cassert>

using namespace std;

int get_num_of_pw_weights(string file_name);

void load_weights(string file_name,
				  weights_dt weights[]);

void load_image(string file_name,
				fms_dt image[]);

void glue_weights(string file_name,
				  weights_grp_dt glued_weights[all_pw_weights]);

void validate_weights(string file_name,
					  weights_grp_dt glued_weights[all_pw_weights]);

void glue_input_image(string file_name,
					  fms_grp_dt input_image[input_image_depth * input_image_height * input_image_width / input_image_group_items]);

void verify_glued_image(string file_name,
						fms_grp_dt input_image[input_image_depth * input_image_height * input_image_width / input_image_group_items]);

void fill_input_image(string file_name,
					  fms_dt input_image[input_image_depth][input_image_height][input_image_width]);

void verify_input_image(string file_name,
						fms_dt input_image[input_image_depth][input_image_height][input_image_width]);

void fill_layer_input(string file_name, fms_dt layer_input[max_fms_size],
					  const int ofms_h, const int ofms_w);

void verify_fill_layer_input(string file_name, fms_dt ofms[max_fms_size], const int ifms_size,
							 const int ifms_h, const int ifms_w);

// V2
void fill_layer_input(string file_name, fms_dt layer_input[MAX_FMS_BUFFER_DEPTH][MIN_FMS_HEIGHT][MIN_FMS_WIDTH],
					  const int ifms_h, const int ifms_w);

void verify_fill_layer_input(string file_name, fms_dt layer_input[MAX_FMS_BUFFER_DEPTH][MIN_FMS_HEIGHT][MIN_FMS_WIDTH],
							 const int ifms_size, const int ifms_h, const int ifms_w);
#endif
