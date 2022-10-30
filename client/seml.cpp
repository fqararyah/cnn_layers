#include "seml.h"

void seml(
		fms_dt input_image[input_image_depth][input_image_height][input_image_width],
		weights_grp_dt off_chip_weights[all_pw_weights],
		fms_dt channels[max_fms_size], fms_dt result[max_fms_size],
		fms_dt result2[max_fms_size], fms_dt tmp_channels[max_tmp_fms_size],
		fms_dt tmp_channels_2[max_tmp_fms_size],
		const layer_0_weights_dt weights_0[layer_0_num_fils][layer_0_depth][layer_0_filter_dim][layer_0_filter_dim],
		dw_weights_dt dw_weights_buffer[max_conv_d][max_conv_h][max_conv_w],
		fms_dt fc_input[fc_layer_input_size]) {

	//begin_code_generation
fill_layer_input("/media/SSD2TB/wd/my_repos/DL_Benchmarking/tflite_scripts_imgnt_accuracy_and_weight_extraction/fms/fms_59_160_7_7.txt",
 channels, 7, 7);
verify_fill_layer_input("/media/SSD2TB/wd/my_repos/DL_Benchmarking/tflite_scripts_imgnt_accuracy_and_weight_extraction/scratch_out/verify_49.txt",
 channels, 7840, 7, 7);
pw_conv(off_chip_weights, channels, result2, 49, layer_49_pw_depth,
    layer_49_pw_num_fils, layer_49_pw_num_of_tiles_in_d,
    layer_49_pw_num_of_tiles_out_d, layer_49_pw_num_of_tiles_h,
    layer_49_pw_num_of_tiles_w, tmp_channels, 0,
    layer_49_pw_num_of_weight_groups_for_one_pass,
    0, layer_49_pw_weights_offset, layer_49_relu);
dump_layer_output("/media/SSD2TB/wd/my_repos/DL_Benchmarking/tflite_scripts_imgnt_accuracy_and_weight_extraction/scratch_out/ofms_49.txt",
 result2, 47040, 7, 7);
fill_dw_layer_weights(dw_weights_50, dw_weights_buffer, layer_50_dw_depth, layer_50_dw_filter_size, layer_50_dw_filter_size);
    dw_conv_3x3(dw_weights_buffer, channels, result2, 50, layer_50_dw_depth,
    layer_50_dw_ifm_width, layer_50_dw_ifm_height, layer_50_dw_num_of_tiles_in_d,
    layer_50_dw_num_of_tiles_h, layer_50_dw_num_of_tiles_w,
    layer_50_dw_strides, layer_50_dw_padding_left,layer_50_dw_padding_top,
    1);
pw_conv(off_chip_weights, channels, result2, 51, layer_51_pw_depth,
    layer_51_pw_num_fils, layer_51_pw_num_of_tiles_in_d,
    layer_51_pw_num_of_tiles_out_d, layer_51_pw_num_of_tiles_h,
    layer_51_pw_num_of_tiles_w, tmp_channels, 0,
    layer_51_pw_num_of_weight_groups_for_one_pass,
    0, layer_51_pw_weights_offset, layer_51_relu);
pw_conv(off_chip_weights, channels, result2, 52, layer_52_pw_depth,
    layer_52_pw_num_fils, layer_52_pw_num_of_tiles_in_d,
    layer_52_pw_num_of_tiles_out_d, layer_52_pw_num_of_tiles_h,
    layer_52_pw_num_of_tiles_w, tmp_channels, 0,
    layer_52_pw_num_of_weight_groups_for_one_pass,
    1, layer_52_pw_weights_offset, layer_52_relu);
dump_layer_output("/media/SSD2TB/wd/my_repos/DL_Benchmarking/tflite_scripts_imgnt_accuracy_and_weight_extraction/scratch_out/ofms_52.txt",
 channels, 62720, 7, 7);
// 	//end_code_generation
 	avgpool(channels, fc_input);
	//fc_layer(fc_weights, fc_input, fc_output);
}
