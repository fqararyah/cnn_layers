#include "prepare_weights_and_input.hpp"
#include "../client/main_file.h"
#include <iostream>
//#include "ap_int.h"
//#include "ap_fixed.h"
using namespace std;

//typedef ap_uint<512> weights_grp_dt;

int main() {
	string weights_file =
			"/media/SSD2TB/wd/cnn_layers/off_chip_weights/off_chip_weights.txt";
	string input_image_file =
			"/media/SSD2TB/wd/my_repos/DL_Benchmarking/tflite_scripts_imgnt_accuracy_and_weight_extraction/fms/fms_1_3_224_224.txt";
	weights_grp_dt glued_weights[all_pw_weights];
	glue_weights(weights_file, glued_weights);
	validate_weights(weights_file, glued_weights);
	fms_dt input_image[input_image_depth][input_image_height][input_image_width];

	int result_o;
	fill_input_image(input_image_file, input_image);
	top_func(input_image, glued_weights, result_o);
	return 0;
}
