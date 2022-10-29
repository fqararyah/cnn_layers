#include "prepare_weights_and_inputs.h"
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
	string input_image_v_file =
				"/media/SSD2TB/wd/my_repos/DL_Benchmarking/tflite_scripts_imgnt_accuracy_and_weight_extraction/scratch_out/inp_img.txt";
	string output_file = "/media/SSD2TB/wd/my_repos/DL_Benchmarking/tflite_scripts_imgnt_accuracy_and_weight_extraction/comp/hls_out.txt";

	weights_grp_dt glued_weights[all_pw_weights];
	fms_dt fc_input[fc_layer_input_size];
	glue_weights(weights_file, glued_weights);
//	cout<<(weights_dt)glued_weights[0](7, 0)<<" @zero\n";
//	cout<<(weights_dt)glued_weights[512](7, 0)<<" @512\n";
	validate_weights(weights_file, glued_weights);
	fms_dt input_image[input_image_depth][input_image_height][input_image_width];

	int result_o;
	fill_input_image(input_image_file, input_image);
	verify_input_image(input_image_v_file,input_image);
//	cout<<(weights_dt)glued_weights[0](7, 0)<<" @zero\n";
//	cout<<(weights_dt)glued_weights[512](7, 0)<<" @512\n";
	top_func(input_image, glued_weights, fc_input);
	dump_ouput(output_file, fc_input, fc_layer_input_size);
//	cout<<(weights_dt)glued_weights[0](7, 0)<<" @zero\n";
//	cout<<(weights_dt)glued_weights[512](7, 0)<<" @512\n";
//	return 0;
}
