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
fill_layer_input("/media/SSD2TB/wd/my_repos/DL_Benchmarking/tflite_scripts_imgnt_accuracy_and_weight_extraction/fms/fms_1_3_224_224.txt",
 channels, 224, 224);
verify_fill_layer_input("/media/SSD2TB/wd/my_repos/DL_Benchmarking/tflite_scripts_imgnt_accuracy_and_weight_extraction/scratch_out/verify_0.txt",
 channels, 150528, 224, 224);
layer_0_3x3(weights_0, input_image, result2);
dump_layer_output("/media/SSD2TB/wd/my_repos/DL_Benchmarking/tflite_scripts_imgnt_accuracy_and_weight_extraction/scratch_out/ofms_0.txt",
 result2, 401408, 112, 112);
fill_dw_layer_weights(dw_weights_2, dw_weights_buffer, layer_2_dw_depth, layer_2_dw_filter_size, layer_2_dw_filter_size);
    dw_conv_3x3(dw_weights_buffer, channels, result2, 2, layer_2_dw_depth,
    layer_2_dw_ifm_width, layer_2_dw_ifm_height, layer_2_dw_num_of_tiles_in_d,
    layer_2_dw_num_of_tiles_h, layer_2_dw_num_of_tiles_w,
    layer_2_dw_strides, layer_2_dw_padding_left,layer_2_dw_padding_top,
    1);
dump_layer_output("/media/SSD2TB/wd/my_repos/DL_Benchmarking/tflite_scripts_imgnt_accuracy_and_weight_extraction/scratch_out/ofms_2.txt",
 channels, 401408, 112, 112);
pw_conv(off_chip_weights, channels, result2, 3, layer_3_pw_depth,
    layer_3_pw_num_fils, layer_3_pw_num_of_tiles_in_d,
    layer_3_pw_num_of_tiles_out_d, layer_3_pw_num_of_tiles_h,
    layer_3_pw_num_of_tiles_w, tmp_channels, 0,
    layer_3_pw_num_of_weight_groups_for_one_pass,
    0, layer_3_pw_weights_offset, layer_3_relu);
dump_layer_output("/media/SSD2TB/wd/my_repos/DL_Benchmarking/tflite_scripts_imgnt_accuracy_and_weight_extraction/scratch_out/ofms_3.txt",
 result2, 200704, 112, 112);
pw_conv(off_chip_weights, channels, result2, 4, layer_4_pw_depth,
    layer_4_pw_num_fils, layer_4_pw_num_of_tiles_in_d,
    layer_4_pw_num_of_tiles_out_d, layer_4_pw_num_of_tiles_h,
    layer_4_pw_num_of_tiles_w, tmp_channels, 0,
    layer_4_pw_num_of_weight_groups_for_one_pass,
    1, layer_4_pw_weights_offset, layer_4_relu);
// 	//end_code_generation
 	avgpool(channels, fc_input);
	//fc_layer(fc_weights, fc_input, fc_output);
}
