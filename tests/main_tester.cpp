#include "../client/prepare_weights_and_inputs.h"
#include "../client/hls_only_main_file.h"
#include <iostream>
#include <dirent.h>
#include "test_utils.h"
#include "../client/cpp_fc.cpp"

//#include "ap_int.h"
//#include "ap_fixed.h"
using namespace std;

// typedef ap_uint<512> weights_grp_dt;
int main(int argc, char **argv) {
	string num_of_pw_weights_file =
			"/media/SSD2TB/wd/cpu_cnn_layers/off_chip_weights/num_of_pw_weights_file.txt";
#if HW == CPU
	string weights_file =
			"/media/SSD2TB/wd/cnn_layers/off_chip_weights/off_chip_weights.txt";
#elif HW== _FPGA
	string weights_file =
			"/media/SSD2TB/wd/cnn_layers/off_chip_weights/off_chip_weights__FPGA.txt";
#endif
	string input_images_folder =
			"/media/SSD2TB/wd/my_repos/DL_Benchmarking/tflite_scripts_imgnt_accuracy_and_weight_extraction/preprocessed_tst_images/";
	string input_image_v_file =
			"/media/SSD2TB/wd/my_repos/DL_Benchmarking/tflite_scripts_imgnt_accuracy_and_weight_extraction/scratch_out/inp_img.txt";
	string output_folder =
			"/media/SSD2TB/wd/my_repos/DL_Benchmarking/tflite_scripts_imgnt_accuracy_and_weight_extraction/cpu_out/";

	string fc_weights_file =
			"/media/SSD2TB/wd/cnn_layers/off_chip_weights/fc_weights.txt";
	string weight_sums_file =
			"/media/SSD2TB/wd/cnn_layers/off_chip_weights/fc_weight_sums.txt";
	string biases_file =
			"/media/SSD2TB/wd/cnn_layers/off_chip_weights/fc_biases.txt";
#if HW == _FPGA
	string predictions_file =
			"/media/SSD2TB/wd/my_repos/DL_Benchmarking/tflite_scripts_imgnt_accuracy_and_weight_extraction/predictions_hls.json";
#elif HW == CPU
	string predictions_file =
			"/media/SSD2TB/wd/my_repos/DL_Benchmarking/tflite_scripts_imgnt_accuracy_and_weight_extraction/predictions_cpu.json";
#endif

	const int num_of_pw_weights = get_num_of_pw_weights(num_of_pw_weights_file);
#if HW == _FPGA
	weights_grp_dt weights[num_of_pw_weights / weights_group_items];
	fms_grp_dt input_image[input_image_depth
			* input_image_num_fms_groups_in_a_channel];
#elif HW == CPU
	weights_dt weights[num_of_pw_weights];
	fms_dt input_image[input_image_depth * input_image_hw];
#endif

	fms_dt fc_input[fc_layer_input_size];

	int8_t fc_weights[fc_cols * fc_layer_input_size];
	int64_t weight_sums[num_classes];
	int biases[num_classes];
	string predictions_file_content = "[";
	int top5[5];

#if HW == _FPGA
	glue_weights(weights_file, weights);
	validate_weights(weights_file, weights);
#elif HW == CPU
	load_weights(weights_file, weights);
#endif

	read_fc_weights(fc_weights_file, fc_weights);
	read_weight_sums(weight_sums_file, weight_sums);
	read_biases(biases_file, biases);

	DIR *dir;
	int img_count = 0;
	int images_to_test = 1;

	if (argc > 1) {
		images_to_test = stoi(argv[1]);
	}

	struct dirent *ent;
	if ((dir = opendir(input_images_folder.c_str())) != NULL) {
		/* print all the files and directories within directory */
		while ((ent = readdir(dir)) != NULL) {
			string file_name = input_images_folder + ent->d_name;
			if (file_name.find(".txt") == std::string::npos) {
				continue;
			}
			string formatted_file_name = ((string) ent->d_name).substr(0,
					((string) ent->d_name).find(".", 0) + 1) + "JPEG";
			cout << file_name << "\n";
// glue_input_image(file_name, input_image);
#if HW == _FPGA
			glue_input_image(file_name, input_image);
#elif HW == CPU
			load_image(file_name, input_image);
#endif
			// verify_glued_image(file_name, input_image);
			// validate_weights(weights_file, glued_weights);
			int ready_to_receive_new_input = 0;
			int *ready_to_receive_new_input_ptr = &ready_to_receive_new_input;
#if HW == _FPGA
			krnl_fibha_v2(input_image, weights, fc_input,
					ready_to_receive_new_input_ptr);
#elif HW == CPU
			top_func(input_image, weights, fc_input,
					 ready_to_receive_new_input_ptr);
#endif
			// std::cout << (int)fc_input[999] << " " << (int)fc_input[710] << " "
			// 		<< (int)fc_input[844] << " " << (int)fc_input[339] << " "
			// 		<< (int)fc_input[338] << " " << (int)fc_input[328] << " "
			// 		<< (int)fc_input[327] << " " << (int)fc_input[335] << " "
			// 		<< (int)fc_input[81] << " " << (int)fc_input[340] << " ";
			fc_layer(fc_input, fc_weights, weight_sums, top5, biases);
//			dump_ouput(output_folder + ent->d_name, fc_input,
//					   fc_layer_input_size);
			predictions_file_content += top_5_to_predictions_dict(top5,
					formatted_file_name);
			img_count++;
			if (img_count == images_to_test) {
				break;
			}
		}
		closedir(dir);
	} else {
		cout << "\nfailed\n";
		return EXIT_FAILURE;
	}
	cout << img_count;
	predictions_file_content = predictions_file_content.substr(0,
			predictions_file_content.length() - 1) + ']';
	save_predictions(predictions_file, predictions_file_content);
}
