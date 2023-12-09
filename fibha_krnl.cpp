#include "all_includes.h"

#include "weights/dw_weights.h"
#include "weights/on_chip_conv_pw_weights.h"
#include "weights/quantization_and_biases.h"

#include "utils/utils.cpp"

#include "layers_cpp/norm_act.cpp"
#include "layers_cpp/conv.cpp"
#include "layers_cpp/pw_conv.cpp"
#include "layers_cpp/dw_conv_v3.cpp"
#include "layers_cpp/pooling.cpp"

#include "sesl_cpp/cnn_pipeline_6_mob_v2.cpp"
#include "sesl_cpp/cnn_pipeline_7_mob_v2.cpp"
#include "seml_cpp/seml.cpp"


using namespace std;


extern "C" {

void krnl_fibha_v1(
		fms_grp_dt input_image[input_image_depth
				* input_image_num_fms_groups_in_a_channel],
		weights_grp_dt off_chip_weights[all_pw_s_weights],
		fms_dt fc_input[fc_layer_input_size],
		int *ready_to_receive_a_new_input_ptr) {

#pragma HLS INTERFACE m_axi port = input_image bundle = gmem0
#pragma HLS INTERFACE m_axi port = off_chip_weights bundle = gmem1
#pragma HLS INTERFACE m_axi port = fc_input bundle = gmem2
#pragma HLS INTERFACE ap_hs port = ready_to_receive_a_new_input_ptr

	fms_dt channels[max_fms_size];
	fms_dt result[max_fms_size];
	//fms_dt result2[max_fms_size];
	fms_dt tmp_channels[max_tmp_fms_size];
	//fms_dt tmp_channels_2[max_tmp_fms_size];

#pragma HLS ARRAY_PARTITION variable = channels type = cyclic factor = main_buffers_partitining_factor
#pragma HLS ARRAY_PARTITION variable = result type = cyclic factor = main_buffers_partitining_factor
#pragma HLS ARRAY_PARTITION variable = tmp_channels type = cyclic factor = main_buffers_partitining_factor
//#pragma HLS ARRAY_PARTITION variable = tmp_channels_2 type = cyclic factor = main_buffers_partitining_factor
//#pragma HLS ARRAY_PARTITION variable = result2 type = cyclic factor = main_buffers_partitining_factor

	dw_weights_dt dw_weights_buffer[max_conv_d][max_conv_h*max_conv_w];
	cnn_pipeline_7_mob_v2(input_image, result, tmp_channels);
	seml(off_chip_weights, channels, result, tmp_channels, weights_1, fc_input);
//	for (int i = 0;
//			i < input_image_depth * input_image_num_fms_groups_in_a_channel;
//			i++) {
//		if(i >= fc_layer_input_size){break;}
//		result2[i] = (fms_dt)input_image[i](7, 0);
//	}
//	seml(input_image, off_chip_weights, channels, result, result2, tmp_channels,
//				tmp_channels_2, weights_1, dw_weights_buffer, fc_input);
}
}
