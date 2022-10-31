#include "prepare_weights_and_inputs.h"
#include "../client/main_file.h"
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
	glue_weights(weights_file, glued_weights);
//	cout<<(weights_dt)glued_weights[0](7, 0)<<" @zero\n";
//	cout<<(weights_dt)glued_weights[512](7, 0)<<" @512\n";
	//validate_weights(weights_file, glued_weights);
	fms_dt input_image[input_image_depth][input_image_height][input_image_width];

	DIR *dir;
	int img_count = 0;
	int images_to_test = 1000;
	struct dirent *ent;
	if ((dir = opendir(input_images_folder.c_str())) != NULL) {
		/* print all the files and directories within directory */
		while ((ent = readdir(dir)) != NULL) {
			cout<<input_images_folder + ent->d_name<<"\n";
			fill_input_image(input_images_folder + ent->d_name, input_image);
			verify_input_image(input_image_v_file,input_image);
			top_func(input_image, glued_weights, fc_input);
			dump_ouput(output_folder + ent->d_name, fc_input, fc_layer_input_size);
			img_count++;
			if(img_count == images_to_test){
				break;
			}
		}
		closedir(dir);
	} else {
		return EXIT_FAILURE;
	}
	cout << img_count;
}
