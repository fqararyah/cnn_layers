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
	//begin_code_generation[19:44]
pw_conv(off_chip_weights, channels, result2, 19, layer_19_pw_depth,
    layer_19_pw_num_fils, layer_19_pw_num_of_tiles_in_d,
    layer_19_pw_num_of_tiles_out_d, layer_19_pw_num_of_tiles_h,
    layer_19_pw_num_of_tiles_w, tmp_channels, 0,
    layer_19_pw_num_of_weight_groups_for_one_pass,
    1, layer_19_pw_weights_offset, layer_19_relu);
fill_layer_input("/media/SSD2TB/wd/my_repos/DL_Benchmarking/tflite_scripts_imgnt_accuracy_and_weight_extraction/fms/fms_23_192_28_28.txt",
 channels, 28, 28);
verify_fill_layer_input("/media/SSD2TB/wd/my_repos/DL_Benchmarking/tflite_scripts_imgnt_accuracy_and_weight_extraction/scratch_out/verify_20.txt",
 channels, 150528, 28, 28);
fill_dw_layer_weights(dw_weights_20, dw_weights_buffer, layer_20_dw_depth, layer_20_dw_filter_size, layer_20_dw_filter_size);
    dw_conv_3x3(dw_weights_buffer, channels, result2, 20, layer_20_dw_depth,
    layer_20_dw_ifm_width, layer_20_dw_ifm_height, layer_20_dw_num_of_tiles_in_d,
    layer_20_dw_num_of_tiles_h, layer_20_dw_num_of_tiles_w,
    layer_20_dw_strides, layer_20_dw_padding_left,layer_20_dw_padding_top,
    0);
dumb_layer_output("/media/SSD2TB/wd/my_repos/DL_Benchmarking/tflite_scripts_imgnt_accuracy_and_weight_extraction/scratch_out/ofms_20.txt",
 result2, 37632, 14, 14);
pw_conv(off_chip_weights, channels, result2, 21, layer_21_pw_depth,
    layer_21_pw_num_fils, layer_21_pw_num_of_tiles_in_d,
    layer_21_pw_num_of_tiles_out_d, layer_21_pw_num_of_tiles_h,
    layer_21_pw_num_of_tiles_w, tmp_channels, 2,
    layer_21_pw_num_of_weight_groups_for_one_pass,
    1, layer_21_pw_weights_offset, layer_21_relu);
dumb_layer_output("/media/SSD2TB/wd/my_repos/DL_Benchmarking/tflite_scripts_imgnt_accuracy_and_weight_extraction/scratch_out/ofms_21.txt",
 channels, 12544, 14, 14);
pw_conv(off_chip_weights, channels, result2, 22, layer_22_pw_depth,
    layer_22_pw_num_fils, layer_22_pw_num_of_tiles_in_d,
    layer_22_pw_num_of_tiles_out_d, layer_22_pw_num_of_tiles_h,
    layer_22_pw_num_of_tiles_w, tmp_channels, 0,
    layer_22_pw_num_of_weight_groups_for_one_pass,
    0, layer_22_pw_weights_offset, layer_22_relu);
dumb_layer_output("/media/SSD2TB/wd/my_repos/DL_Benchmarking/tflite_scripts_imgnt_accuracy_and_weight_extraction/scratch_out/ofms_22.txt",
 result2, 75264, 14, 14);
fill_dw_layer_weights(dw_weights_23, dw_weights_buffer, layer_23_dw_depth, layer_23_dw_filter_size, layer_23_dw_filter_size);
    dw_conv_3x3(dw_weights_buffer, channels, result2, 23, layer_23_dw_depth,
    layer_23_dw_ifm_width, layer_23_dw_ifm_height, layer_23_dw_num_of_tiles_in_d,
    layer_23_dw_num_of_tiles_h, layer_23_dw_num_of_tiles_w,
    layer_23_dw_strides, layer_23_dw_padding_left,layer_23_dw_padding_top,
    1);
dumb_layer_output("/media/SSD2TB/wd/my_repos/DL_Benchmarking/tflite_scripts_imgnt_accuracy_and_weight_extraction/scratch_out/ofms_23.txt",
 channels, 75264, 14, 14);
pw_conv(off_chip_weights, channels, result2, 24, layer_24_pw_depth,
    layer_24_pw_num_fils, layer_24_pw_num_of_tiles_in_d,
    layer_24_pw_num_of_tiles_out_d, layer_24_pw_num_of_tiles_h,
    layer_24_pw_num_of_tiles_w, tmp_channels, 3,
    layer_24_pw_num_of_weight_groups_for_one_pass,
    0, layer_24_pw_weights_offset, layer_24_relu);
dumb_layer_output("/media/SSD2TB/wd/my_repos/DL_Benchmarking/tflite_scripts_imgnt_accuracy_and_weight_extraction/scratch_out/ofms_24.txt",
 result2, 12544, 14, 14);
pw_conv(off_chip_weights, channels, result2, 25, layer_25_pw_depth,
    layer_25_pw_num_fils, layer_25_pw_num_of_tiles_in_d,
    layer_25_pw_num_of_tiles_out_d, layer_25_pw_num_of_tiles_h,
    layer_25_pw_num_of_tiles_w, tmp_channels, 0,
    layer_25_pw_num_of_weight_groups_for_one_pass,
    1, layer_25_pw_weights_offset, layer_25_relu);
dumb_layer_output("/media/SSD2TB/wd/my_repos/DL_Benchmarking/tflite_scripts_imgnt_accuracy_and_weight_extraction/scratch_out/ofms_25.txt",
 channels, 75264, 14, 14);
fill_dw_layer_weights(dw_weights_26, dw_weights_buffer, layer_26_dw_depth, layer_26_dw_filter_size, layer_26_dw_filter_size);
    dw_conv_3x3(dw_weights_buffer, channels, result2, 26, layer_26_dw_depth,
    layer_26_dw_ifm_width, layer_26_dw_ifm_height, layer_26_dw_num_of_tiles_in_d,
    layer_26_dw_num_of_tiles_h, layer_26_dw_num_of_tiles_w,
    layer_26_dw_strides, layer_26_dw_padding_left,layer_26_dw_padding_top,
    0);
dumb_layer_output("/media/SSD2TB/wd/my_repos/DL_Benchmarking/tflite_scripts_imgnt_accuracy_and_weight_extraction/scratch_out/ofms_26.txt",
 result2, 75264, 14, 14);
pw_conv(off_chip_weights, channels, result2, 27, layer_27_pw_depth,
    layer_27_pw_num_fils, layer_27_pw_num_of_tiles_in_d,
    layer_27_pw_num_of_tiles_out_d, layer_27_pw_num_of_tiles_h,
    layer_27_pw_num_of_tiles_w, tmp_channels, 3,
    layer_27_pw_num_of_weight_groups_for_one_pass,
    1, layer_27_pw_weights_offset, layer_27_relu);
dumb_layer_output("/media/SSD2TB/wd/my_repos/DL_Benchmarking/tflite_scripts_imgnt_accuracy_and_weight_extraction/scratch_out/ofms_27.txt",
 channels, 12544, 14, 14);
pw_conv(off_chip_weights, channels, result2, 28, layer_28_pw_depth,
    layer_28_pw_num_fils, layer_28_pw_num_of_tiles_in_d,
    layer_28_pw_num_of_tiles_out_d, layer_28_pw_num_of_tiles_h,
    layer_28_pw_num_of_tiles_w, tmp_channels, 0,
    layer_28_pw_num_of_weight_groups_for_one_pass,
    0, layer_28_pw_weights_offset, layer_28_relu);
dumb_layer_output("/media/SSD2TB/wd/my_repos/DL_Benchmarking/tflite_scripts_imgnt_accuracy_and_weight_extraction/scratch_out/ofms_28.txt",
 result2, 75264, 14, 14);
fill_dw_layer_weights(dw_weights_29, dw_weights_buffer, layer_29_dw_depth, layer_29_dw_filter_size, layer_29_dw_filter_size);
    dw_conv_3x3(dw_weights_buffer, channels, result2, 29, layer_29_dw_depth,
    layer_29_dw_ifm_width, layer_29_dw_ifm_height, layer_29_dw_num_of_tiles_in_d,
    layer_29_dw_num_of_tiles_h, layer_29_dw_num_of_tiles_w,
    layer_29_dw_strides, layer_29_dw_padding_left,layer_29_dw_padding_top,
    1);
dumb_layer_output("/media/SSD2TB/wd/my_repos/DL_Benchmarking/tflite_scripts_imgnt_accuracy_and_weight_extraction/scratch_out/ofms_29.txt",
 channels, 75264, 14, 14);
pw_conv(off_chip_weights, channels, result2, 30, layer_30_pw_depth,
    layer_30_pw_num_fils, layer_30_pw_num_of_tiles_in_d,
    layer_30_pw_num_of_tiles_out_d, layer_30_pw_num_of_tiles_h,
    layer_30_pw_num_of_tiles_w, tmp_channels, 1,
    layer_30_pw_num_of_weight_groups_for_one_pass,
    0, layer_30_pw_weights_offset, layer_30_relu);
dumb_layer_output("/media/SSD2TB/wd/my_repos/DL_Benchmarking/tflite_scripts_imgnt_accuracy_and_weight_extraction/scratch_out/ofms_30.txt",
 result2, 12544, 14, 14);
pw_conv(off_chip_weights, channels, result2, 31, layer_31_pw_depth,
    layer_31_pw_num_fils, layer_31_pw_num_of_tiles_in_d,
    layer_31_pw_num_of_tiles_out_d, layer_31_pw_num_of_tiles_h,
    layer_31_pw_num_of_tiles_w, tmp_channels, 0,
    layer_31_pw_num_of_weight_groups_for_one_pass,
    1, layer_31_pw_weights_offset, layer_31_relu);
dumb_layer_output("/media/SSD2TB/wd/my_repos/DL_Benchmarking/tflite_scripts_imgnt_accuracy_and_weight_extraction/scratch_out/ofms_31.txt",
 channels, 75264, 14, 14);
fill_dw_layer_weights(dw_weights_32, dw_weights_buffer, layer_32_dw_depth, layer_32_dw_filter_size, layer_32_dw_filter_size);
    dw_conv_3x3(dw_weights_buffer, channels, result2, 32, layer_32_dw_depth,
    layer_32_dw_ifm_width, layer_32_dw_ifm_height, layer_32_dw_num_of_tiles_in_d,
    layer_32_dw_num_of_tiles_h, layer_32_dw_num_of_tiles_w,
    layer_32_dw_strides, layer_32_dw_padding_left,layer_32_dw_padding_top,
    0);
dumb_layer_output("/media/SSD2TB/wd/my_repos/DL_Benchmarking/tflite_scripts_imgnt_accuracy_and_weight_extraction/scratch_out/ofms_32.txt",
 result2, 75264, 14, 14);
pw_conv(off_chip_weights, channels, result2, 33, layer_33_pw_depth,
    layer_33_pw_num_fils, layer_33_pw_num_of_tiles_in_d,
    layer_33_pw_num_of_tiles_out_d, layer_33_pw_num_of_tiles_h,
    layer_33_pw_num_of_tiles_w, tmp_channels, 2,
    layer_33_pw_num_of_weight_groups_for_one_pass,
    1, layer_33_pw_weights_offset, layer_33_relu);
dumb_layer_output("/media/SSD2TB/wd/my_repos/DL_Benchmarking/tflite_scripts_imgnt_accuracy_and_weight_extraction/scratch_out/ofms_33.txt",
 channels, 18816, 14, 14);
pw_conv(off_chip_weights, channels, result2, 34, layer_34_pw_depth,
    layer_34_pw_num_fils, layer_34_pw_num_of_tiles_in_d,
    layer_34_pw_num_of_tiles_out_d, layer_34_pw_num_of_tiles_h,
    layer_34_pw_num_of_tiles_w, tmp_channels, 0,
    layer_34_pw_num_of_weight_groups_for_one_pass,
    0, layer_34_pw_weights_offset, layer_34_relu);
dumb_layer_output("/media/SSD2TB/wd/my_repos/DL_Benchmarking/tflite_scripts_imgnt_accuracy_and_weight_extraction/scratch_out/ofms_34.txt",
 result2, 112896, 14, 14);
fill_dw_layer_weights(dw_weights_35, dw_weights_buffer, layer_35_dw_depth, layer_35_dw_filter_size, layer_35_dw_filter_size);
    dw_conv_3x3(dw_weights_buffer, channels, result2, 35, layer_35_dw_depth,
    layer_35_dw_ifm_width, layer_35_dw_ifm_height, layer_35_dw_num_of_tiles_in_d,
    layer_35_dw_num_of_tiles_h, layer_35_dw_num_of_tiles_w,
    layer_35_dw_strides, layer_35_dw_padding_left,layer_35_dw_padding_top,
    1);
dumb_layer_output("/media/SSD2TB/wd/my_repos/DL_Benchmarking/tflite_scripts_imgnt_accuracy_and_weight_extraction/scratch_out/ofms_35.txt",
 channels, 112896, 14, 14);
pw_conv(off_chip_weights, channels, result2, 36, layer_36_pw_depth,
    layer_36_pw_num_fils, layer_36_pw_num_of_tiles_in_d,
    layer_36_pw_num_of_tiles_out_d, layer_36_pw_num_of_tiles_h,
    layer_36_pw_num_of_tiles_w, tmp_channels, 1,
    layer_36_pw_num_of_weight_groups_for_one_pass,
    0, layer_36_pw_weights_offset, layer_36_relu);
dumb_layer_output("/media/SSD2TB/wd/my_repos/DL_Benchmarking/tflite_scripts_imgnt_accuracy_and_weight_extraction/scratch_out/ofms_36.txt",
 result2, 18816, 14, 14);
pw_conv(off_chip_weights, channels, result2, 37, layer_37_pw_depth,
    layer_37_pw_num_fils, layer_37_pw_num_of_tiles_in_d,
    layer_37_pw_num_of_tiles_out_d, layer_37_pw_num_of_tiles_h,
    layer_37_pw_num_of_tiles_w, tmp_channels, 0,
    layer_37_pw_num_of_weight_groups_for_one_pass,
    1, layer_37_pw_weights_offset, layer_37_relu);
dumb_layer_output("/media/SSD2TB/wd/my_repos/DL_Benchmarking/tflite_scripts_imgnt_accuracy_and_weight_extraction/scratch_out/ofms_37.txt",
 channels, 112896, 14, 14);
fill_dw_layer_weights(dw_weights_38, dw_weights_buffer, layer_38_dw_depth, layer_38_dw_filter_size, layer_38_dw_filter_size);
    dw_conv_3x3(dw_weights_buffer, channels, result2, 38, layer_38_dw_depth,
    layer_38_dw_ifm_width, layer_38_dw_ifm_height, layer_38_dw_num_of_tiles_in_d,
    layer_38_dw_num_of_tiles_h, layer_38_dw_num_of_tiles_w,
    layer_38_dw_strides, layer_38_dw_padding_left,layer_38_dw_padding_top,
    0);
dumb_layer_output("/media/SSD2TB/wd/my_repos/DL_Benchmarking/tflite_scripts_imgnt_accuracy_and_weight_extraction/scratch_out/ofms_38.txt",
 result2, 112896, 14, 14);
pw_conv(off_chip_weights, channels, result2, 39, layer_39_pw_depth,
    layer_39_pw_num_fils, layer_39_pw_num_of_tiles_in_d,
    layer_39_pw_num_of_tiles_out_d, layer_39_pw_num_of_tiles_h,
    layer_39_pw_num_of_tiles_w, tmp_channels, 0,
    layer_39_pw_num_of_weight_groups_for_one_pass,
    1, layer_39_pw_weights_offset, layer_39_relu);
dumb_layer_output("/media/SSD2TB/wd/my_repos/DL_Benchmarking/tflite_scripts_imgnt_accuracy_and_weight_extraction/scratch_out/ofms_39.txt",
 channels, 18816, 14, 14);
pw_conv(off_chip_weights, channels, result2, 40, layer_40_pw_depth,
    layer_40_pw_num_fils, layer_40_pw_num_of_tiles_in_d,
    layer_40_pw_num_of_tiles_out_d, layer_40_pw_num_of_tiles_h,
    layer_40_pw_num_of_tiles_w, tmp_channels, 0,
    layer_40_pw_num_of_weight_groups_for_one_pass,
    0, layer_40_pw_weights_offset, layer_40_relu);
dumb_layer_output("/media/SSD2TB/wd/my_repos/DL_Benchmarking/tflite_scripts_imgnt_accuracy_and_weight_extraction/scratch_out/ofms_40.txt",
 result2, 112896, 14, 14);
fill_dw_layer_weights(dw_weights_41, dw_weights_buffer, layer_41_dw_depth, layer_41_dw_filter_size, layer_41_dw_filter_size);
    dw_conv_3x3(dw_weights_buffer, channels, result2, 41, layer_41_dw_depth,
    layer_41_dw_ifm_width, layer_41_dw_ifm_height, layer_41_dw_num_of_tiles_in_d,
    layer_41_dw_num_of_tiles_h, layer_41_dw_num_of_tiles_w,
    layer_41_dw_strides, layer_41_dw_padding_left,layer_41_dw_padding_top,
    1);
dumb_layer_output("/media/SSD2TB/wd/my_repos/DL_Benchmarking/tflite_scripts_imgnt_accuracy_and_weight_extraction/scratch_out/ofms_41.txt",
 channels, 28224, 7, 7);
pw_conv(off_chip_weights, channels, result2, 42, layer_42_pw_depth,
    layer_42_pw_num_fils, layer_42_pw_num_of_tiles_in_d,
    layer_42_pw_num_of_tiles_out_d, layer_42_pw_num_of_tiles_h,
    layer_42_pw_num_of_tiles_w, tmp_channels, 2,
    layer_42_pw_num_of_weight_groups_for_one_pass,
    0, layer_42_pw_weights_offset, layer_42_relu);
dumb_layer_output("/media/SSD2TB/wd/my_repos/DL_Benchmarking/tflite_scripts_imgnt_accuracy_and_weight_extraction/scratch_out/ofms_42.txt",
 result2, 7840, 7, 7);
pw_conv(off_chip_weights, channels, result2, 43, layer_43_pw_depth,
    layer_43_pw_num_fils, layer_43_pw_num_of_tiles_in_d,
    layer_43_pw_num_of_tiles_out_d, layer_43_pw_num_of_tiles_h,
    layer_43_pw_num_of_tiles_w, tmp_channels, 0,
    layer_43_pw_num_of_weight_groups_for_one_pass,
    1, layer_43_pw_weights_offset, layer_43_relu);
dumb_layer_output("/media/SSD2TB/wd/my_repos/DL_Benchmarking/tflite_scripts_imgnt_accuracy_and_weight_extraction/scratch_out/ofms_43.txt",
 channels, 47040, 7, 7);
// 	//end_code_generation
// 	avgpool(result2, fc_input);
	//fc_layer(fc_weights, fc_input, fc_output);

}
