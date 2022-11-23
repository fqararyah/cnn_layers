#include "hls_only_main_file.h"
#include "../tests/test_utils.h"

void top_func(
		fms_grp_dt input_image[input_image_depth*input_image_height*input_image_width/input_image_group_items],
		weights_grp_dt off_chip_weights[all_pw_weights],
		fms_dt fc_input[fc_layer_input_size], int *ready_to_receive_a_new_input_ptr) {

	fms_dt channels[max_fms_size];
	fms_dt result[max_fms_size];
	fms_dt result2[max_fms_size];
	fms_dt tmp_channels[max_tmp_fms_size];
	fms_dt tmp_channels_2[max_tmp_fms_size];

#pragma HLS ARRAY_PARTITION variable = channels type = cyclic factor = main_buffers_partitining_factor
#pragma HLS ARRAY_PARTITION variable = tmp_channels type = cyclic factor = main_buffers_partitining_factor
#pragma HLS ARRAY_PARTITION variable = tmp_channels_2 type = cyclic factor = main_buffers_partitining_factor
#pragma HLS ARRAY_PARTITION variable = result2 type = cyclic factor = main_buffers_partitining_factor

	dw_weights_dt dw_weights_buffer[max_conv_d][max_conv_h][max_conv_w];
	cnn_pipeline_7_mob_v2(input_image, result2, tmp_channels);
	dump_layer_output(
			"/media/SSD2TB/wd/my_repos/DL_Benchmarking/tflite_scripts_imgnt_accuracy_and_weight_extraction/scratch_out/tmp_ofms_6.txt",
			tmp_channels, 56*56*24, 56, 56);
	dump_layer_output(
				"/media/SSD2TB/wd/my_repos/DL_Benchmarking/tflite_scripts_imgnt_accuracy_and_weight_extraction/scratch_out/tmp_ofms_7.txt",
				result2, 56*56*144, 56, 56);
	seml(input_image, off_chip_weights, channels, result, result2, tmp_channels,
			tmp_channels_2, weights_0, dw_weights_buffer, fc_input);
}
