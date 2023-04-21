#include "hls_only_main_file.h"
#include "../tests/test_utils.h"

void top_func(
	fms_grp_dt input_image[input_image_depth * input_image_num_fms_groups_in_a_channel],
	weights_grp_dt off_chip_weights[all_pw_weights],
	fms_dt fc_input[fc_layer_input_size],
	int *ready_to_receive_a_new_input_ptr)
{

#if FIBHA_VERSION == 1
	fms_dt channels[max_fms_size];
	fms_dt result[max_fms_size];
	// fms_dt result2[max_fms_size];
	fms_dt tmp_channels[max_tmp_fms_size];
	// fms_dt tmp_channels2[max_tmp_fms_size];

#pragma HLS ARRAY_PARTITION variable = channels type = cyclic factor = main_buffers_partitining_factor
#pragma HLS ARRAY_PARTITION variable = tmp_channels type = cyclic factor = main_buffers_partitining_factor
//#pragma HLS ARRAY_PARTITION variable = tmp_channels2 type = cyclic factor = main_buffers_partitining_factor
#pragma HLS ARRAY_PARTITION variable = result type = cyclic factor = main_buffers_partitining_factor
	//#pragma HLS ARRAY_PARTITION variable = result2 type = cyclic factor = main_buffers_partitining_factor

#if CHAIN_LENGTH == 9 && MODEL_ID == 2
	_0_1_2_bottlenecks_chain(input_image,
							 tmp_channels);
	dump_layer_output("/media/SSD2TB/wd/my_repos/DL_Benchmarking/tflite_scripts_imgnt_accuracy_and_weight_extraction/scratch_out/ofms_8.txt",
					  tmp_channels, 56 * 56 * 24, 56, 56);
#elif CHAIN_LENGTH == 6 && MODEL_ID == 2
	_0_1_bottlenecks_chain(input_image,
						   channels);
#if DEBUGGING
	dump_layer_output("/media/SSD2TB/wd/my_repos/DL_Benchmarking/tflite_scripts_imgnt_accuracy_and_weight_extraction/scratch_out/ofms_7.txt",
					  channels, layer_7_pw_specs);
#endif
#endif
	copy_channels_to_tmp_channels(channels, tmp_channels);
	seml(off_chip_weights, channels, result, tmp_channels, weights_1, fc_input);

#elif FIBHA_VERSION == 2
	fms_dt channels[MAX_FMS_BUFFER_DEPTH][MIN_FMS_HEIGHT][MIN_FMS_WIDTH];
	fms_dt result[MAX_FMS_BUFFER_DEPTH][MIN_FMS_HEIGHT][MIN_FMS_WIDTH];
	// fms_dt result2[max_fms_size];
	fms_dt tmp_channels[MAX_FMS_BUFFER_DEPTH][MIN_FMS_HEIGHT][MIN_FMS_WIDTH];
	// fms_dt tmp_channels2[max_tmp_fms_size];

#pragma HLS ARRAY_PARTITION variable = channels type = complete dim = 2
#pragma HLS ARRAY_PARTITION variable = channels type = complete dim = 3
#pragma HLS ARRAY_PARTITION variable = tmp_channels type = complete dim = 2
#pragma HLS ARRAY_PARTITION variable = tmp_channels type = complete dim = 3
#pragma HLS ARRAY_PARTITION variable = result type = complete dim = 2
#pragma HLS ARRAY_PARTITION variable = result type = complete dim = 3
	pipelined_engines_caller(channels);
	dump_layer_output("/media/SSD2TB/wd/my_repos/DL_Benchmarking/tflite_scripts_imgnt_accuracy_and_weight_extraction/scratch_out/ofms_15.txt",
 channels, layer_15_pw_specs);
	copy_channels_to_tmp_channels(channels, tmp_channels);
	seml(off_chip_weights, channels, result, tmp_channels, fc_input);
#endif
}
