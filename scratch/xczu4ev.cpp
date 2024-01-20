// #include "all_includes.h"

// #include "model_components/model/headers/model_glue.h"
// //###################################################

// #include "model_components/utils/utils.h"
// #include "model_components/utils/utils.cpp"

// #include "model_components/layers/headers/layers_glue.h"

// //#if ONLY_SESL == 0
// #include "model_components/layers/impl/norm_act.cpp"
// #include "model_components/layers/impl/conv_utils.cpp"
// #include "model_components/layers/impl/dw_conv_v1_6.cpp"
// #include "model_components/layers/impl/dw_conv_v2.cpp"
// #include "model_components/layers/impl/pw_conv_v1_2.cpp"
// #include "model_components/layers/impl/pw_conv_v2.cpp"
// //#endif
// //
// #include "model_components/layers/impl/pooling.cpp"

// #include "model_components/model/fused_bottlenecks/bottleneck_kernels.cpp"
// #include "model_components/model/fused_bottlenecks/bottleneck_0.cpp"
// #include "model_components/model/fused_bottlenecks/bottleneck_1.cpp"
// #include "model_components/model/fused_bottlenecks/bottlenecks_chain.cpp"
// #include "model_components/model/fused_bottlenecks/bottleneck_2.cpp"
// #include "model_components/model/fused_bottlenecks/bottlenecks_chain_0_1_2.cpp"

// #include "model_components/model/pipelined_engines/pre_first_pipeline_layers.cpp"
// #include "model_components/model/pipelined_engines/pipelined_engines.cpp"
// #include "model_components/model/pipelined_engines/pipeline_main.cpp"

// #if ONLY_SEML == 0 && USE_FIRB == 0 && FIRST_PART_IMPLEMENTATION == BOTTLENECK_CHAIN_MODE
// #include "dep_sesl_cpp/cnn_pipeline_6_mob_v2.cpp"
// #include "dep_sesl_cpp/cnn_pipeline_7_mob_v2.cpp"
// #endif

// #include "model_components/model/SEML/imp/mob_v2_seml.cpp"
// #include "model_components/model/SEML/imp/mob_v2_seml_v2.cpp"
// #include "model_components/model/SEML/imp/mob_v2_0_5_seml.cpp"
// #include "model_components/model/SEML/imp/mob_v2_0_5_seml_v2.cpp"
// #include "model_components/model/SEML/imp/mob_v2_0_75_seml.cpp"
// #include "model_components/model/SEML/imp/mob_v2_0_75_seml_v2.cpp"
// #include "model_components/model/SEML/imp/mob_v2_0_25_seml.cpp"
// #include "model_components/model/SEML/imp/mob_v2_0_25_seml_v2.cpp"

// using namespace std;

// extern "C" {

// static weights_dt on_chip_weights[all_on_chip_pw_s_weights
// 		/ ON_CHIP_WEIGHTS_PORTS][ON_CHIP_WEIGHTS_PORTS];
// static int model_configs_list[2 * max_conv_layers];

// void krnl_fibha_v2(
// 		fms_grp_dt input_image[input_image_depth
// 				* input_image_num_fms_groups_in_a_channel],
// 		weights_grp_dt off_chip_weights[all_off_chip_pw_s_weights],
// 		weights_grp_dt on_chip_weights_src[all_on_chip_pw_s_weights_groups],
// 		fms_dt fc_input[fc_layer_input_size],
// 		const int model_config_list_src[2 * max_conv_layers],
// 		int *first_lunch) {

// //#pragma HLS INTERFACE m_axi port = input_image bundle = gmem0
// //#pragma HLS INTERFACE m_axi port = off_chip_weights bundle = gmem1
// //#pragma HLS INTERFACE m_axi port = fc_input bundle = gmem2
// //#pragma HLS INTERFACE ap_hs port = ready_to_receive_a_new_input_ptr


// #if FIBHA_VERSION == 1
// 	fms_dt channels[max_fms_size];
// 	fms_dt result[max_fms_size];
// 	// fms_dt result2[max_fms_size];
// 	fms_dt tmp_channels[max_tmp_fms_size];
// 	// fms_dt tmp_channels2[max_tmp_fms_size];

// #pragma HLS ARRAY_PARTITION variable = channels type = cyclic factor = main_buffers_partitining_factor
// #pragma HLS ARRAY_PARTITION variable = tmp_channels type = cyclic factor = main_buffers_partitining_factor
// //#pragma HLS ARRAY_PARTITION variable = tmp_channels2 type = cyclic factor = main_buffers_partitining_factor
// #pragma HLS ARRAY_PARTITION variable = result type = cyclic factor = main_buffers_partitining_factor
// 	//#pragma HLS ARRAY_PARTITION variable = result2 type = cyclic factor = main_buffers_partitining_factor

// #if CHAIN_LENGTH == 9 && MODEL_ID == 2
// 	_0_1_2_bottlenecks_chain(input_image,
// 			tmp_channels);
// 	dump_layer_output("/media/SSD2TB/wd/my_repos/DL_Benchmarking/tflite_scripts_imgnt_accuracy_and_weight_extraction/scratch_out/ofms_8.txt",
// 			tmp_channels, 56 * 56 * 24, 56, 56);
// #elif CHAIN_LENGTH == 6 && MODEL_ID == 2 && !ONLY_SEML
// 	_0_1_bottlenecks_chain(input_image,
// 			channels);
// #if DEBUGGING
// 	dump_layer_output("/media/SSD2TB/wd/my_repos/DL_Benchmarking/tflite_scripts_imgnt_accuracy_and_weight_extraction/scratch_out/ofms_7.txt",
// 			channels, layer_7_pw_specs);
// #endif
// #endif
// 	copy_channels_to_tmp_channels(channels, tmp_channels);
// #if ONLY_SESL == 0
// 	seml(off_chip_weights, channels, result, tmp_channels, fc_input);
// #endif
// #elif FIBHA_VERSION == 2
// 	fms_dt channels[][CHANNELS_TILE_HEIGHT][CHANNELS_TILE_WIDTH];
// 	fms_dt result[][CHANNELS_TILE_HEIGHT][CHANNELS_TILE_WIDTH];
// 	// fms_dt result2[max_fms_size];
// 	fms_dt tmp_channels[][CHANNELS_TILE_HEIGHT][CHANNELS_TILE_WIDTH];

// #pragma HLS bind_storage variable=channels type=ram_2p impl=URAM
// #pragma HLS bind_storage variable=result type=ram_2p impl=URAM
// #pragma HLS bind_storage variable=tmp_channels type=ram_2p impl=URAM


// 	// fms_dt tmp_channels2[max_tmp_fms_size];

// #pragma HLS array_reshape variable = channels type = complete dim = 2
// #pragma HLS array_reshape variable = channels type = complete dim = 3
// #pragma HLS array_reshape variable = tmp_channels type = complete dim = 2
// #pragma HLS array_reshape variable = tmp_channels type = complete dim = 3
// #pragma HLS array_reshape variable = result type = complete dim = 2
// #pragma HLS array_reshape variable = result type = complete dim = 3

// #if FIRST_PART_IMPLEMENTATION == PIPELINED_ENGINES_MODE

// #pragma HLS ARRAY_PARTITION variable=on_chip_weights type=complete dim = 2

// 	if (*first_lunch != 0)
// 	{
// #if HW == CPU
// 		fill_on_chip_weights_cpu(on_chip_weights_src, on_chip_weights);
// #elif HW == _FPGA
// 		fill_model_configs_list(model_config_list_src, model_configs_list);
// 		fill_on_chip_weights_fpga(on_chip_weights_src,
// 				on_chip_weights, 0);
// #endif // #if HW == CPU
// 	}
// 	for(int i=0;i<2 * max_conv_layers;i++) {
// 		fc_input[i] = model_configs_list[i];
// 		fc_input[i + 500] = model_config_list_src[i];
// 	}
// #endif // #if FIRST_PART_IMPLEMENTATION == PIPELINED_ENGINES_MODE
// #if ONLY_SEML == 0

// #if FIRST_PART_IMPLEMENTATION == BOTTLENECK_CHAIN_MODE

// #if CHAIN_LENGTH == 9 && MODEL_ID == 2

// 	_0_1_2_bottlenecks_chain(input_image,
// 			tmp_channels);
// 	dump_layer_output("/media/SSD2TB/wd/my_repos/DL_Benchmarking/tflite_scripts_imgnt_accuracy_and_weight_extraction/scratch_out/ofms_8.txt",
// 			tmp_channels, 56 * 56 * 24, 56, 56);
// #elif CHAIN_LENGTH == 6 && MODEL_ID == 2 && !ONLY_SEML
// 	_0_1_bottlenecks_chain(input_image, channels);
// #if DEBUGGING
// 	dump_layer_output("/media/SSD2TB/wd/my_repos/DL_Benchmarking/tflite_scripts_imgnt_accuracy_and_weight_extraction/scratch_out/ofms_7.txt",
// 			channels, layer_7_pw_specs);
// #endif //#if DEBUGGING

// #endif // #if CHAIN_LENGTH == 9 && MODEL_ID == 2

// 	copy_channels_to_tmp_channels(channels, tmp_channels);

// #else // FIRST_PART_IMPLEMENTATION == BOTTLENECK_CHAIN_MODE

// 	pipelined_engines_caller(input_image,on_chip_weights, channels, model_configs_list);
// #if DEBUGGING
// 	dump_layer_output("/media/SSD2TB/wd/my_repos/DL_Benchmarking/tflite_scripts_imgnt_accuracy_and_weight_extraction/scratch_out/ofms_14.txt",
// 			channels, layer_14_dw_specs);
// #endif // ONLY_SEML == 0

// #endif // PIPELINED_ENGINES_MODE == BOTTLENECK_CHAIN_MODE
// 	for(int i=0;i<1024;i++){
// 			channels[i][0][0] = off_chip_weights[i];
// 		}
// 	seml(off_chip_weights, channels, result, tmp_channels, fc_input,
// 			model_configs_list);
// //	for(int i=0;i<1024;i++){
// //		channels[i][0][0] = off_chip_weights[i];
// //	}
// //	avgpool(channels, fc_input, layer_67_avgpool_specs);
// #endif // ONLY_SEML == 0

// #endif

// }
// }
