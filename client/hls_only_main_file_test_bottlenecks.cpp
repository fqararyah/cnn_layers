#include "hls_only_main_file_test_bottlenecks.h"
#include "../tests/test_utils.h"

void top_func(
		fms_grp_dt input_image[input_image_depth
				* input_image_num_fms_groups_in_a_channel],
		weights_grp_dt off_chip_weights[all_pw_s_weights],
		fms_dt fc_input[fc_layer_input_size],
		int *ready_to_receive_a_new_input_ptr) {

	fms_dt channels[max_fms_size];
	fms_dt result[max_fms_size];
	//fms_dt result2[max_fms_size];
	fms_dt tmp_channels[max_tmp_fms_size];
	//fms_dt tmp_channels2[max_tmp_fms_size];

#pragma HLS ARRAY_PARTITION variable = channels type = cyclic factor = main_buffers_partitining_factor
#pragma HLS ARRAY_PARTITION variable = tmp_channels type = cyclic factor = main_buffers_partitining_factor
//#pragma HLS ARRAY_PARTITION variable = tmp_channels2 type = cyclic factor = main_buffers_partitining_factor
#pragma HLS ARRAY_PARTITION variable = result type = cyclic factor = main_buffers_partitining_factor
//#pragma HLS ARRAY_PARTITION variable = result2 type = cyclic factor = main_buffers_partitining_factor

	dw_weights_dt dw_weights_buffer[max_conv_d][max_conv_h * max_conv_w];

//	fill_layer_input(
//			"/media/SSD2TB/wd/my_repos/DL_Benchmarking/tflite_scripts_imgnt_accuracy_and_weight_extraction/mob_v2/fms/fms_4_16_112_112.txt",
//			result, 112, 112);
//	verify_fill_layer_input(
//			"/media/SSD2TB/wd/my_repos/DL_Benchmarking/tflite_scripts_imgnt_accuracy_and_weight_extraction/scratch_out/verify_4.txt",
//			result, 200704, 112, 112);

//first time, only fill second row with valid data
	fms_dt chain_input[chain_0_1_input_size];
	fms_dt chain_output[chain_0_1_output_size];

	_0_1_bottlenecks_chain(input_image, result);
	avgpool(result, fc_input);
	//seml(off_chip_weights, channels, result, tmp_channels, weights_1, fc_input);
}
