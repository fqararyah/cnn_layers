#include <stdlib.h>
#include <fstream>
#include <iostream>
#include "../../../fiba_v2_kernels/src/model_components/basic_defs/basic_defs_glue.h"

#if TESTING

#include "fpga_test_utils.cpp"
#include "../../../fiba_v2_kernels/src/krnl_fibha_v2.h"
// #include "/media/SSD2TB/fareed/wd/vitis_ide_projects/fiba_v2_kernels/src/krnl_fibha_v2.cpp"

#include <fstream>
#include "ap_int.h"
#include <iostream>
#include "cstdlib"
#include <chrono>

#include "../../../fiba_v2_kernels/src/model_components/layers/headers/norm_act.h"
#include "../client/fpga_prepare_weights_and_input.cpp"
#include "../client/fpga_prepare_weights_and_input_v2.cpp"

#include <iostream>
#include <dirent.h>
#include "../client/fpga_cpp_fc.cpp"

// #include "ap_int.h"
// #include "ap_fixed.h"
using namespace std;

// typedef ap_uint<512> weights_grp_dt;
int main(int argc, char **argv)
{

	int images_to_test = 10;
	int soft_pipeline_len = 0;

	int model_configs_list[2 * max_conv_layers] = {0}; // up to 100-conv layers

	if (argc > 1)
	{
		images_to_test = stoi(argv[1]);
	}
	if (argc > 2)
	{
		soft_pipeline_len = stoi(argv[2]);
	}
	if (argc > 3)
	{
		read_model_configs(argv[2], model_configs_list);
		cout << "model_configs list is read.\n*************\n";
	}

	string dw_weights_file =
		"/media/SSD2TB/fareed/wd/cnn_layers/off_chip_weights/" + get_model_prefix() +
		"_off_chip_dw_weights_pipe_" + to_string(PIPELINE_LENGTH) + ".txt";
	string fused_scales_file =
		"/media/SSD2TB/fareed/wd/cnn_layers/off_chip_weights/" + get_model_prefix() +
		"_fused_scales_pipe_" + to_string(PIPELINE_LENGTH) + ".txt";
	string fused_zps_file =
		"/media/SSD2TB/fareed/wd/cnn_layers/off_chip_weights/" + get_model_prefix() +
		"_fused_zps_pipe_" + to_string(PIPELINE_LENGTH) + ".txt";
	string on_chip_weights_file =
		"/media/SSD2TB/fareed/wd/cnn_layers/on_chip_weights/" + get_model_prefix() + "_on_chip_weights_pipe_" + to_string(PIPELINE_LENGTH) + ".txt";
#if HW == CPU
	string weights_file =
		"/media/SSD2TB/fareed/wd/cnn_layers/off_chip_weights/" + get_model_prefix() + "_off_chip_weights_pipe_" + to_string(PIPELINE_LENGTH) + ".txt";

#elif HW == _FPGA
	string weights_file =
		"/media/SSD2TB/fareed/wd/cnn_layers/off_chip_weights/" + get_model_prefix() + "_off_chip_weights_fpga_pipe_" + to_string(PIPELINE_LENGTH) + ".txt";
#endif
	string input_images_folder =
		"/media/SSD2TB/shared/vedliot_evaluation/D3.3_Accuracy_Evaluation/imagenet/imagenet_val2012_resized_1000/";
	//"/media/SSD2TB/fareed/wd/my_repos/DL_Benchmarking/tflite_scripts_imgnt_accuracy_and_weight_extraction/preprocessed_tst_images/";
	string input_image_v_file =
		"/media/SSD2TB/fareed/wd/my_repos/DL_Benchmarking/tflite_scripts_imgnt_accuracy_and_weight_extraction/scratch_out/inp_img.txt";
	string output_folder =
		"/media/SSD2TB/fareed/wd/my_repos/DL_Benchmarking/tflite_scripts_imgnt_accuracy_and_weight_extraction/cpu_out/";

	string fc_weights_file =
		"/media/SSD2TB/fareed/wd/cnn_layers/off_chip_weights/" + get_model_prefix() + "_fc_weights.txt";
	string weight_sums_file =
		"/media/SSD2TB/fareed/wd/cnn_layers/off_chip_weights/" + get_model_prefix() + "_fc_weight_sums.txt";
	string biases_file =
		"/media/SSD2TB/fareed/wd/cnn_layers/off_chip_weights/" + get_model_prefix() + "_fc_biases.txt";

	string soft_pipe_specs_file = "/media/SSD2TB/fareed/wd/cnn_layers/model_config/fusion_granualities/mob_v2_pipe_0.txt";

#if HW == _FPGA
	string predictions_file =
		"/media/SSD2TB/fareed/wd/my_repos/DL_Benchmarking/tflite_scripts_imgnt_accuracy_and_weight_extraction/predictions/predictions_hls_" + to_string(images_to_test) + ".json";
#elif HW == CPU
	string predictions_file =
		"/media/SSD2TB/fareed/wd/my_repos/DL_Benchmarking/tflite_scripts_imgnt_accuracy_and_weight_extraction/predictions/predictions_cpu_" + to_string(images_to_test) + ".json";
#endif

#if HW == _FPGA
	static weights_grp_dt weights[all_off_chip_pw_s_weights];
	static weights_dt dw_weights[all_dw_off_chip_weights];
	static weights_grp_dt glued_on_chip_weights[all_on_chip_pw_s_weights_groups];
	static fms_grp_dt input_image[input_image_depth * input_image_num_fms_groups_in_a_channel];
	fused_scales_dt off_chip_fused_scales[all_off_chip_fused_scales_zps];
	biases_dt off_chip_fused_zeropoints[all_off_chip_fused_scales_zps];
#elif HW == CPU
	weights_dt *weights = (weights_dt *)malloc(all_off_chip_pw_s_weights * weights_dt_width / 8);
	weights_dt *dw_weights = (weights_dt *)malloc(all_dw_off_chip_weights);
	weights_grp_dt *glued_on_chip_weights = (weights_grp_dt *)malloc(all_on_chip_pw_s_weights_groups *
																	 weights_group_items * weights_dt_width / 8);
	fms_dt input_image[input_image_depth * input_image_hw];
	fused_scales_dt off_chip_fused_scales[all_off_chip_fused_scales_zps];
	biases_dt off_chip_fused_zeropoints[all_off_chip_fused_scales_zps];
#endif

	static fms_dt fc_input[fc_layer_input_size];

	static int8_t fc_weights[fc_cols * fc_layer_input_size];
	static int64_t weight_sums[num_classes];
	int biases[num_classes];
	string predictions_file_content = "[";
	int top5[5];

	soft_pipe_specs_struct soft_pipe_specs[max_conv_layers]; //{112, 0, 112, 112, 112, 0, 56, 56, 56};
	fill_soft_pipeline_configs(soft_pipe_specs_file, soft_pipe_specs, soft_pipeline_len);
	// soft_pipe_specs[0] = {19, 3, 1};
	// soft_pipe_specs[1] = {0, 3, 1};
	// soft_pipe_specs[2] = {17, 1, 0};
	// soft_pipe_specs[3] = {17, 1, 0};
	// soft_pipe_specs[4] = {17, 1, 0};
	// soft_pipe_specs[5] = {0, 1, 0};
	// soft_pipe_specs[6] = {8, 0, 0};
	// soft_pipe_specs[7] = {8, 0, 0};
	//{19, 0, 17, 17, 17, 0, 8, 8, 8};

#if HW == _FPGA
#if FIRST_PART_IMPLEMENTATION == PIPELINED_ENGINES_MODE
	glue_on_chip_weights_fpga(on_chip_weights_file,
							  glued_on_chip_weights);
#endif
	glue_weights(weights_file, weights);
	validate_weights(weights_file, weights);
#elif HW == CPU
#if FIRST_PART_IMPLEMENTATION == PIPELINED_ENGINES_MODE
	glue_on_chip_weights_cpu(on_chip_weights_file,
							 glued_on_chip_weights);
#endif
	load_weights(weights_file, weights);
#endif
	load_fused_scales(fused_scales_file, off_chip_fused_scales);
	load_fused_zps(fused_zps_file, off_chip_fused_zeropoints);
	load_weights(dw_weights_file, dw_weights);
	read_fc_weights(fc_weights_file, fc_weights);
	read_weight_sums(weight_sums_file, weight_sums);
	read_biases(biases_file, biases);

	DIR *dir;
	int img_count = 0;

	struct dirent *ent;
	if ((dir = opendir(input_images_folder.c_str())) != NULL)
	{
		/* print all the files and directories within directory */
		while ((ent = readdir(dir)) != NULL)
		{
			string file_name = input_images_folder + ent->d_name;
			if (file_name.find(".JPEG") == std::string::npos && file_name.find(".jpeg") == std::string::npos)
			{
				continue;
			}
			string formatted_file_name = ((string)ent->d_name).substr(0, ((string)ent->d_name).find(".", 0) + 1) + "JPEG";
			cout << file_name << "\n";
			// glue_input_image(file_name, input_image);
			auto start = chrono::steady_clock::now();
#if HW == _FPGA
			glue_and_quantize_input_image(file_name, input_image, quantize_layer_specs);
#elif HW == CPU
			// load_image(file_name, input_image);

			load_and_quantize_image(file_name, input_image, quantize_layer_specs);
#endif
			auto end = chrono::steady_clock::now();

			cout << "Elapsed time in milliseconds reading: "
				 << chrono::duration_cast<chrono::microseconds>(end - start).count() / 1000
				 << " ms" << endl;
			// verify_glued_image(file_name, input_image);
			// validate_weights(weights_file, glued_weights);
			int ready_to_receive_new_input = 0;
			int *ready_to_receive_new_input_ptr = &ready_to_receive_new_input;
#if HW == _FPGA
			krnl_fibha_v2(input_image, weights, dw_weights, off_chip_fused_scales,
						  off_chip_fused_zeropoints, glued_on_chip_weights, fc_input,
						  model_configs_list, soft_pipe_specs,
						  soft_pipeline_len,
						  &img_count);
#elif HW == CPU
			top_func(input_image, weights, dw_weights, off_chip_fused_scales,
					 off_chip_fused_zeropoints, glued_on_chip_weights, fc_input,
					 model_configs_list, soft_pipe_specs, soft_pipeline_len);
#endif
			// std::cout << (int)fc_input[999] << " " << (int)fc_input[710] << " "
			// 		<< (int)fc_input[844] << " " << (int)fc_input[339] << " "
			// 		<< (int)fc_input[338] << " " << (int)fc_input[328] << " "
			// 		<< (int)fc_input[327] << " " << (int)fc_input[335] << " "
			// 		<< (int)fc_input[81] << " " << (int)fc_input[340] << " ";
			start = chrono::steady_clock::now();

#if MODEL_ID == RESNET50
			fc_layer(fc_input, fc_weights, weight_sums, top5, biases, layer_74_fc_specs);
#elif MODEL_ID == MOB_V2 || MODEL_ID == MOB_V2_0_5 || MODEL_ID == MOB_V2_0_75 || MODEL_ID == MOB_V2_0_25
			// for(int i=0;i<1000;i++){printf("%d ", (int)fc_input[i]);}
			// 			printf("\n");
			fc_layer(fc_input, fc_weights, weight_sums, top5, biases, layer_68_fc_specs);
#endif
			end = chrono::steady_clock::now();

			cout << "Elapsed time in milliseconds: "
				 << chrono::duration_cast<chrono::microseconds>(end - start).count() / 1000
				 << " ms" << endl;
			//			dump_ouput(output_folder + ent->d_name, fc_input,
			//					   fc_layer_input_size);
			predictions_file_content += top_5_to_predictions_dict(top5,
																  formatted_file_name);
			img_count++;
			if (img_count == images_to_test)
			{
				break;
			}
		}
		closedir(dir);
	}
	else
	{
		cout << "\nfailed\n";
		return EXIT_FAILURE;
	}
	cout << img_count;
	predictions_file_content = predictions_file_content.substr(0,
															   predictions_file_content.length() - 1) +
							   ']';
	save_predictions(predictions_file, predictions_file_content);
#if HW == CPU
	free(weights);
#endif
}

#endif
