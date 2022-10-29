#ifndef PREPARE_WEIGHTS_INPUTS
#define PREPARE_WEIGHTS_INPUTS
#include "../basic_defs/basic_defs_glue.h"
#include "../model/model_glue.h"
#include <fstream>
#include "ap_int.h"
#include <iostream>

using namespace std;

void glue_weights(string file_name,
		weights_grp_dt glued_weights[all_pw_weights]);

void validate_weights(string file_name,
		weights_grp_dt glued_weights[all_pw_weights]);

void fill_input_image(string file_name,
		fms_dt input_image[input_image_depth][input_image_height][input_image_width]);

void verify_input_image(string file_name,
		fms_dt input_image[input_image_depth][input_image_height][input_image_width]);

void fill_layer_input(string file_name, fms_dt layer_input[max_fms_size],
		const int ofms_h, const int ofms_w);

void verify_fill_layer_input(string file_name, fms_dt ofms[max_fms_size], const int ofms_size,
		const int ofms_h, const int ofms_w);

#endif
