#include "hls_only_main_file.h"

#if HW == CPU

#include "../tests/test_utils.h"

static 	weights_dt on_chip_weights[all_on_chip_pw_s_weights / ON_CHIP_WEIGHTS_PORTS][ON_CHIP_WEIGHTS_PORTS];

void top_func(
	fms_grp_dt input_image[input_image_depth * input_image_num_fms_groups_in_a_channel],
	weights_grp_dt off_chip_weights[all_pw_s_weights],
	weights_dt off_chip_dw_weights[all_dw_off_chip_weights],
	fused_scales_dt off_chip_fused_scales[all_off_chip_fused_scales_zps],
	biases_dt off_chip_fused_zeropoints[all_off_chip_fused_scales_zps],
	weights_grp_dt on_chip_weights_src[all_on_chip_pw_s_weights_groups],
	fms_dt fc_input[fc_layer_input_size],
	const int model_configs_list_src[2 * max_conv_layers])
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

#if CHAIN_LENGTH == 9 && (MODEL_ID == MOB_V2 || MODEL_ID == MOB_V2_0_5 || MODEL_ID == MOB_V2_0_75 || MODEL_ID == MOB_V2_0_25)
	_0_1_2_bottlenecks_chain(input_image,
							 tmp_channels);
	dump_layer_output("/media/SSD2TB/fareed/wd/my_repos/DL_Benchmarking/tflite_scripts_imgnt_accuracy_and_weight_extraction/scratch_out/ofms_8.txt",
					  tmp_channels, 56 * 56 * 24, 56, 56);
#elif CHAIN_LENGTH == 6 && (MODEL_ID == MOB_V2 || MODEL_ID == MOB_V2_0_75) && !ONLY_SEML
	_0_1_bottlenecks_chain(input_image,
						   channels);
#if DEBUGGING
	dump_layer_output("/media/SSD2TB/fareed/wd/my_repos/DL_Benchmarking/tflite_scripts_imgnt_accuracy_and_weight_extraction/scratch_out/ofms_7.txt",
					  channels, layer_7_pw_specs);
#endif
#endif
	copy_channels_to_tmp_channels(channels, tmp_channels);
#if ONLY_SESL == 0
	seml(off_chip_weights, channels, result, tmp_channels, fc_input);
#endif
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

	int model_configs_list[2 * max_conv_layers] = {0};
#if FIRST_PART_IMPLEMENTATION == PIPELINED_ENGINES_MODE

#pragma HLS ARRAY_PARTITION variable = on_chip_weights type = complete dim = 2

	if (!on_chip_weights_filled)
	{
		on_chip_weights_filled = true;

		fill_model_configs_list(model_configs_list_src, model_configs_list);
#if HW == CPU
		fill_on_chip_weights_cpu(on_chip_weights_src, on_chip_weights);
#elif HW == _FPGA
		fill_on_chip_weights_fpga(on_chip_weights_src,
								  on_chip_weights, 0);
#endif
	}
#endif // #if FIRST_PART_IMPLEMENTATION == PIPELINED_ENGINES_MODE

#if ONLY_SEML == 0

#if FIRST_PART_IMPLEMENTATION == BOTTLENECK_CHAIN_MODE
#if CHAIN_LENGTH == 9 && (MODEL_ID == MOB_V2 || MODEL_ID == MOB_V2_0_5 || MODEL_ID == MOB_V2_0_75 || MODEL_ID == MOB_V2_0_25)
	_0_1_2_bottlenecks_chain(input_image,
							 tmp_channels);
	dump_layer_output("/media/SSD2TB/fareed/wd/my_repos/DL_Benchmarking/tflite_scripts_imgnt_accuracy_and_weight_extraction/scratch_out/ofms_8.txt",
					  tmp_channels, 56 * 56 * 24, 56, 56);
#elif CHAIN_LENGTH == 6 && (MODEL_ID == MOB_V2 || MODEL_ID == MOB_V2_0_5 || MODEL_ID == MOB_V2_0_75 || MODEL_ID == MOB_V2_0_25) && !ONLY_SEML
	_0_1_bottlenecks_chain(input_image,
						   channels);
#if DEBUGGING
	dump_layer_output("/media/SSD2TB/fareed/wd/my_repos/DL_Benchmarking/tflite_scripts_imgnt_accuracy_and_weight_extraction/scratch_out/ofms_7.txt",
					  channels, layer_7_pw_specs);
#endif
#endif // #if CHAIN_LENGTH == 9 && MODEL_ID == 2
	copy_channels_to_tmp_channels(channels, tmp_channels);
#else
	//pipelined_engines_caller(input_image, on_chip_weights, channels, model_configs_list);
#if DEBUGGING
	dump_layer_output("/media/SSD2TB/fareed/wd/my_repos/DL_Benchmarking/tflite_scripts_imgnt_accuracy_and_weight_extraction/scratch_out/ofms_14.txt",
					  channels, layer_14_dw_specs);
#endif

#endif // PIPELINED_ENGINES_MODE == BOTTLENECK_CHAIN_MODE
	seml(off_chip_weights, off_chip_dw_weights, off_chip_fused_scales, off_chip_fused_zeropoints, channels, result, tmp_channels, fc_input, model_configs_list);
#endif // ONLY_SEML == 0

#endif
}

#endif