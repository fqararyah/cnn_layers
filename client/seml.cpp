#include "seml.h"

void seml(
		fms_dt input_image[input_image_depth][input_image_height][input_image_width],
		weights_grp_dt off_chip_weights[all_pw_weights],
		fms_dt channels[max_fms_size], fms_dt result[max_fms_size],
		fms_dt result2[max_fms_size], fms_dt tmp_channels[max_tmp_fms_size],
		fms_dt tmp_channels_2[max_tmp_fms_size],
		const layer_0_weights_dt weights_0[layer_0_num_fils][layer_0_depth][layer_0_filter_size][layer_0_filter_size],
		dw_weights_dt dw_weights_buffer[max_conv_d][max_conv_h][max_conv_w],
		fms_dt fc_input[fc_layer_input_size]) {

	layer_0_using_pw(weights_0, input_image, result2, 0, layer_0_depth,
			layer_0_num_fils, 1, layer_0_num_of_tiles_out_d,
			layer_0_num_of_tiles_h, layer_0_num_of_tiles_w);
	//to generate layers within a range, rather than all layers, add, .e.g. [1:20]
	//where 1 is the first and 20 is the last layer to be generated
	//to the end of the previous layer
	//begin_code_generation[32:36]
fill_layer_input("/media/SSD2TB/wd/my_repos/DL_Benchmarking/tflite_scripts_imgnt_accuracy_and_weight_extraction/fms/fms_38_384_14_14.txt",
 result2, 14, 14);
verify_fill_layer_input("/media/SSD2TB/wd/my_repos/DL_Benchmarking/tflite_scripts_imgnt_accuracy_and_weight_extraction/scratch_out/verify_32.txt",
 result2, 75264, 14, 14);
fill_dw_layer_weights(dw_weights_32, dw_weights_buffer, layer_32_dw_depth, layer_32_dw_filter_size, layer_32_dw_filter_size);
    dw_conv_3x3(dw_weights_buffer, channels, result2, 32, layer_32_dw_depth,
    layer_32_dw_ifm_width, layer_32_dw_ifm_height, layer_32_dw_num_of_tiles_in_d,
    layer_32_dw_num_of_tiles_h, layer_32_dw_num_of_tiles_w,
    layer_32_dw_strides, layer_32_dw_padding_left,layer_32_dw_padding_top,
    1);
dumb_layer_output("/media/SSD2TB/wd/my_repos/DL_Benchmarking/tflite_scripts_imgnt_accuracy_and_weight_extraction/scratch_out/ofms_32.txt",
 channels, 75264, 14, 14);
pw_conv(off_chip_weights, channels, result2, 33, layer_33_pw_depth,
    layer_33_pw_num_fils, layer_33_pw_num_of_tiles_in_d,
    layer_33_pw_num_of_tiles_out_d, layer_33_pw_num_of_tiles_h,
    layer_33_pw_num_of_tiles_w, tmp_channels, 2,
    layer_33_pw_num_of_weight_groups_for_one_pass,
    0, layer_33_pw_weights_offset, layer_33_relu);
dumb_layer_output("/media/SSD2TB/wd/my_repos/DL_Benchmarking/tflite_scripts_imgnt_accuracy_and_weight_extraction/scratch_out/ofms_33.txt",
 result2, 18816, 14, 14);
pw_conv(off_chip_weights, channels, result2, 34, layer_34_pw_depth,
    layer_34_pw_num_fils, layer_34_pw_num_of_tiles_in_d,
    layer_34_pw_num_of_tiles_out_d, layer_34_pw_num_of_tiles_h,
    layer_34_pw_num_of_tiles_w, tmp_channels, 0,
    layer_34_pw_num_of_weight_groups_for_one_pass,
    1, layer_34_pw_weights_offset, layer_34_relu);
dumb_layer_output("/media/SSD2TB/wd/my_repos/DL_Benchmarking/tflite_scripts_imgnt_accuracy_and_weight_extraction/scratch_out/ofms_34.txt",
 channels, 112896, 14, 14);
fill_dw_layer_weights(dw_weights_35, dw_weights_buffer, layer_35_dw_depth, layer_35_dw_filter_size, layer_35_dw_filter_size);
    dw_conv_3x3(dw_weights_buffer, channels, result2, 35, layer_35_dw_depth,
    layer_35_dw_ifm_width, layer_35_dw_ifm_height, layer_35_dw_num_of_tiles_in_d,
    layer_35_dw_num_of_tiles_h, layer_35_dw_num_of_tiles_w,
    layer_35_dw_strides, layer_35_dw_padding_left,layer_35_dw_padding_top,
    0);
// 	//end_code_generation
// 	avgpool(result2, fc_input);
	//fc_layer(fc_weights, fc_input, fc_output);

}
