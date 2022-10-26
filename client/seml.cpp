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
	//begin_code_generation[42:]
pw_conv(off_chip_weights, channels, result2, 42, layer_42_pw_depth,
    layer_42_pw_num_fils, layer_42_pw_num_of_tiles_in_d,
    layer_42_pw_num_of_tiles_out_d, layer_42_pw_num_of_tiles_h,
    layer_42_pw_num_of_tiles_w, tmp_channels, 2,
    layer_42_pw_num_of_weight_groups_for_one_pass,
    1, layer_42_pw_weights_offset, layer_42_relu);
pw_conv(off_chip_weights, channels, result2, 43, layer_43_pw_depth,
    layer_43_pw_num_fils, layer_43_pw_num_of_tiles_in_d,
    layer_43_pw_num_of_tiles_out_d, layer_43_pw_num_of_tiles_h,
    layer_43_pw_num_of_tiles_w, tmp_channels, 0,
    layer_43_pw_num_of_weight_groups_for_one_pass,
    0, layer_43_pw_weights_offset, layer_43_relu);
fill_dw_layer_weights(dw_weights_44, dw_weights_buffer, layer_44_dw_depth, layer_44_dw_filter_size, layer_44_dw_filter_size);
    dw_conv_3x3(dw_weights_buffer, channels, result2, 44, layer_44_dw_depth,
    layer_44_dw_ifm_width, layer_44_dw_ifm_height, layer_44_dw_num_of_tiles_in_d,
    layer_44_dw_num_of_tiles_h, layer_44_dw_num_of_tiles_w,
    layer_44_dw_strides, layer_44_dw_padding_left,layer_44_dw_padding_top,
    1);
pw_conv(off_chip_weights, channels, result2, 45, layer_45_pw_depth,
    layer_45_pw_num_fils, layer_45_pw_num_of_tiles_in_d,
    layer_45_pw_num_of_tiles_out_d, layer_45_pw_num_of_tiles_h,
    layer_45_pw_num_of_tiles_w, tmp_channels, 3,
    layer_45_pw_num_of_weight_groups_for_one_pass,
    0, layer_45_pw_weights_offset, layer_45_relu);
pw_conv(off_chip_weights, channels, result2, 46, layer_46_pw_depth,
    layer_46_pw_num_fils, layer_46_pw_num_of_tiles_in_d,
    layer_46_pw_num_of_tiles_out_d, layer_46_pw_num_of_tiles_h,
    layer_46_pw_num_of_tiles_w, tmp_channels, 0,
    layer_46_pw_num_of_weight_groups_for_one_pass,
    1, layer_46_pw_weights_offset, layer_46_relu);
fill_dw_layer_weights(dw_weights_47, dw_weights_buffer, layer_47_dw_depth, layer_47_dw_filter_size, layer_47_dw_filter_size);
    dw_conv_3x3(dw_weights_buffer, channels, result2, 47, layer_47_dw_depth,
    layer_47_dw_ifm_width, layer_47_dw_ifm_height, layer_47_dw_num_of_tiles_in_d,
    layer_47_dw_num_of_tiles_h, layer_47_dw_num_of_tiles_w,
    layer_47_dw_strides, layer_47_dw_padding_left,layer_47_dw_padding_top,
    0);
pw_conv(off_chip_weights, channels, result2, 48, layer_48_pw_depth,
    layer_48_pw_num_fils, layer_48_pw_num_of_tiles_in_d,
    layer_48_pw_num_of_tiles_out_d, layer_48_pw_num_of_tiles_h,
    layer_48_pw_num_of_tiles_w, tmp_channels, 1,
    layer_48_pw_num_of_weight_groups_for_one_pass,
    1, layer_48_pw_weights_offset, layer_48_relu);
pw_conv(off_chip_weights, channels, result2, 49, layer_49_pw_depth,
    layer_49_pw_num_fils, layer_49_pw_num_of_tiles_in_d,
    layer_49_pw_num_of_tiles_out_d, layer_49_pw_num_of_tiles_h,
    layer_49_pw_num_of_tiles_w, tmp_channels, 0,
    layer_49_pw_num_of_weight_groups_for_one_pass,
    0, layer_49_pw_weights_offset, layer_49_relu);
fill_dw_layer_weights(dw_weights_50, dw_weights_buffer, layer_50_dw_depth, layer_50_dw_filter_size, layer_50_dw_filter_size);
    dw_conv_3x3(dw_weights_buffer, channels, result2, 50, layer_50_dw_depth,
    layer_50_dw_ifm_width, layer_50_dw_ifm_height, layer_50_dw_num_of_tiles_in_d,
    layer_50_dw_num_of_tiles_h, layer_50_dw_num_of_tiles_w,
    layer_50_dw_strides, layer_50_dw_padding_left,layer_50_dw_padding_top,
    1);
fill_layer_input("/media/SSD2TB/wd/my_repos/DL_Benchmarking/tflite_scripts_imgnt_accuracy_and_weight_extraction/fms/fms_61_960_7_7.txt",
 channels, 7, 7);
verify_fill_layer_input("/media/SSD2TB/wd/my_repos/DL_Benchmarking/tflite_scripts_imgnt_accuracy_and_weight_extraction/scratch_out/verify_51.txt",
 channels, 47040, 7, 7);
pw_conv(off_chip_weights, channels, result2, 51, layer_51_pw_depth,
    layer_51_pw_num_fils, layer_51_pw_num_of_tiles_in_d,
    layer_51_pw_num_of_tiles_out_d, layer_51_pw_num_of_tiles_h,
    layer_51_pw_num_of_tiles_w, tmp_channels, 0,
    layer_51_pw_num_of_weight_groups_for_one_pass,
    0, layer_51_pw_weights_offset, layer_51_relu);
dumb_layer_output("/media/SSD2TB/wd/my_repos/DL_Benchmarking/tflite_scripts_imgnt_accuracy_and_weight_extraction/scratch_out/ofms_51.txt",
 result2, 15680, 7, 7);
pw_conv(off_chip_weights, channels, result2, 52, layer_52_pw_depth,
    layer_52_pw_num_fils, layer_52_pw_num_of_tiles_in_d,
    layer_52_pw_num_of_tiles_out_d, layer_52_pw_num_of_tiles_h,
    layer_52_pw_num_of_tiles_w, tmp_channels, 0,
    layer_52_pw_num_of_weight_groups_for_one_pass,
    1, layer_52_pw_weights_offset, layer_52_relu);
dumb_layer_output("/media/SSD2TB/wd/my_repos/DL_Benchmarking/tflite_scripts_imgnt_accuracy_and_weight_extraction/scratch_out/ofms_52.txt",
 channels, 62720, 7, 7);
// 	//end_code_generation
// 	avgpool(result2, fc_input);
	//fc_layer(fc_weights, fc_input, fc_output);

}
