#include "hls_only_main_file.h"
#include "../tests/test_utils.h"
#include "../model_components/model/bottleneck/bottlenecks_chain.h"

void top_func(
		fms_grp_dt input_image[input_image_depth
				* input_image_num_fms_groups_in_a_channel],
		weights_grp_dt off_chip_weights[all_pw_weights],
		fms_dt fc_input[fc_layer_input_size],
		int *ready_to_receive_a_new_input_ptr) {

	fms_dt channels[max_fms_size];
	fms_dt result[max_fms_size];
	fms_dt result2[max_fms_size];
	fms_dt tmp_channels[max_tmp_fms_size];
	fms_dt tmp_channels2[max_tmp_fms_size];

	dw_weights_dt dw_weights_buffer[max_conv_d][max_conv_h * max_conv_w];

	fill_layer_input(
			"/media/SSD2TB/wd/my_repos/DL_Benchmarking/tflite_scripts_imgnt_accuracy_and_weight_extraction/mob_v2/fms/fms_4_16_112_112.txt",
			result2, 112, 112);
	verify_fill_layer_input(
			"/media/SSD2TB/wd/my_repos/DL_Benchmarking/tflite_scripts_imgnt_accuracy_and_weight_extraction/scratch_out/verify_4.txt",
			result2, 200704, 112, 112);

	fms_dt chain_input[_1_chain_specs.chain_input_size];
	fms_dt chain_output[_1_chain_specs.bottlenck_1_output_buffer_size];
	_1_bottlenecks_chain(chain_input, // chain_input_height*chain_input_width*chain_input_depth
			result, _1_chain_specs, 0);

	dump_layer_output(
			"/media/SSD2TB/wd/my_repos/DL_Benchmarking/tflite_scripts_imgnt_accuracy_and_weight_extraction/scratch_out/ofms_6.txt",
			result, 75264, 56, 56);

	avgpool(result, fc_input);
}
