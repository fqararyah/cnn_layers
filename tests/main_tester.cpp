#include "../client/prepare_weights_and_inputs.h"
#include "../client/hls_only_main_file.h"
#include <iostream>
#include <dirent.h>
#include "test_utils.h"

//#include "ap_int.h"
//#include "ap_fixed.h"
using namespace std;

//typedef ap_uint<512> weights_grp_dt;

int main() {
	string weights_file =
			"/media/SSD2TB/wd/cnn_layers/off_chip_weights/off_chip_weights.txt";
	string input_images_folder =
			"/media/SSD2TB/wd/my_repos/DL_Benchmarking/tflite_scripts_imgnt_accuracy_and_weight_extraction/preprocessed_tst_images/";
	string input_image_v_file =
			"/media/SSD2TB/wd/my_repos/DL_Benchmarking/tflite_scripts_imgnt_accuracy_and_weight_extraction/scratch_out/inp_img.txt";
	string output_folder =
			"/media/SSD2TB/wd/my_repos/DL_Benchmarking/tflite_scripts_imgnt_accuracy_and_weight_extraction/fpga_out/";

	weights_grp_dt glued_weights[all_pw_weights];
	fms_dt fc_input[fc_layer_input_size];
	#if FPGA
	glue_weights(weights_file, glued_weights);
	validate_weights(weights_file, glued_weights);
	#elif CPU
	#endif
	fms_grp_dt input_image[input_image_depth * input_image_num_fms_groups_in_a_channel];

	DIR *dir;
	int img_count = 0;
	int images_to_test = 1;
	struct dirent *ent;
	if ((dir = opendir(input_images_folder.c_str())) != NULL) {
		/* print all the files and directories within directory */
		while ((ent = readdir(dir)) != NULL) {
			string file_name = input_images_folder + ent->d_name;
			if (file_name.find(".txt") == std::string::npos) {
				continue;
			}
			cout << file_name << "\n";
			#if FPGA
			glue_input_image(file_name, input_image);
			#elif CPU
			#endif
			//verify_glued_image(file_name, input_image);
			//validate_weights(weights_file, glued_weights);
			int ready_to_receive_new_input = 0;
			int *ready_to_receive_new_input_ptr = &ready_to_receive_new_input;
			top_func(input_image, glued_weights, fc_input,
					ready_to_receive_new_input_ptr);
			dump_ouput(output_folder + ent->d_name, fc_input,
					fc_layer_input_size);
			img_count++;
			if (img_count == images_to_test) {
				break;
			}
		}
		closedir(dir);
	} else {
		return EXIT_FAILURE;
	}
	cout << img_count;
}
