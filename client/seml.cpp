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
	//begin_code_generation[4:9]
pw_conv(off_chip_weights, channels, result2, 4, layer_4_pw_depth,
    layer_4_pw_num_fils, layer_4_pw_num_of_tiles_in_d,
    layer_4_pw_num_of_tiles_out_d, layer_4_pw_num_of_tiles_h,
    layer_4_pw_num_of_tiles_w, tmp_channels, 0,
    layer_4_pw_num_of_weight_groups_for_one_pass,
    1, layer_4_pw_weights_offset, layer_4_relu);
fill_layer_input("/media/SSD2TB/wd/my_repos/DL_Benchmarking/tflite_scripts_imgnt_accuracy_and_weight_extraction/fms/fms_5_96_112_112.txt",
 channels, 112, 112);
verify_fill_layer_input("/media/SSD2TB/wd/my_repos/DL_Benchmarking/tflite_scripts_imgnt_accuracy_and_weight_extraction/scratch_out/verify_5.txt",
 channels, 1204224, 112, 112);
fill_dw_layer_weights(dw_weights_5, dw_weights_buffer, layer_5_dw_depth, layer_5_dw_filter_size, layer_5_dw_filter_size);
    dw_conv_3x3(dw_weights_buffer, channels, result2, 5, layer_5_dw_depth,
    layer_5_dw_ifm_width, layer_5_dw_ifm_height, layer_5_dw_num_of_tiles_in_d,
    layer_5_dw_num_of_tiles_h, layer_5_dw_num_of_tiles_w,
    layer_5_dw_strides, layer_5_dw_padding_left,layer_5_dw_padding_top,
    0);
dumb_layer_output("/media/SSD2TB/wd/my_repos/DL_Benchmarking/tflite_scripts_imgnt_accuracy_and_weight_extraction/scratch_out/ofms_5.txt",
 result2, 301056, 56, 56);
pw_conv(off_chip_weights, channels, result2, 6, layer_6_pw_depth,
    layer_6_pw_num_fils, layer_6_pw_num_of_tiles_in_d,
    layer_6_pw_num_of_tiles_out_d, layer_6_pw_num_of_tiles_h,
    layer_6_pw_num_of_tiles_w, tmp_channels, 0,
    layer_6_pw_num_of_weight_groups_for_one_pass,
    1, layer_6_pw_weights_offset, layer_6_relu);
dumb_layer_output("/media/SSD2TB/wd/my_repos/DL_Benchmarking/tflite_scripts_imgnt_accuracy_and_weight_extraction/scratch_out/ofms_6.txt",
 channels, 75264, 56, 56);
pw_conv(off_chip_weights, channels, result2, 7, layer_7_pw_depth,
    layer_7_pw_num_fils, layer_7_pw_num_of_tiles_in_d,
    layer_7_pw_num_of_tiles_out_d, layer_7_pw_num_of_tiles_h,
    layer_7_pw_num_of_tiles_w, tmp_channels, 2,
    layer_7_pw_num_of_weight_groups_for_one_pass,
    0, layer_7_pw_weights_offset, layer_7_relu);
dumb_layer_output("/media/SSD2TB/wd/my_repos/DL_Benchmarking/tflite_scripts_imgnt_accuracy_and_weight_extraction/scratch_out/ofms_7.txt",
 result2, 451584, 56, 56);
fill_dw_layer_weights(dw_weights_8, dw_weights_buffer, layer_8_dw_depth, layer_8_dw_filter_size, layer_8_dw_filter_size);
    dw_conv_3x3(dw_weights_buffer, channels, result2, 8, layer_8_dw_depth,
    layer_8_dw_ifm_width, layer_8_dw_ifm_height, layer_8_dw_num_of_tiles_in_d,
    layer_8_dw_num_of_tiles_h, layer_8_dw_num_of_tiles_w,
    layer_8_dw_strides, layer_8_dw_padding_left,layer_8_dw_padding_top,
    1);
// 	//end_code_generation
// 	avgpool(result2, fc_input);
	//fc_layer(fc_weights, fc_input, fc_output);

}
