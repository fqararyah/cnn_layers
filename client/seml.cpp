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
	//begin_code_generation[2:]
fill_layer_input("/media/SSD2TB/wd/my_repos/DL_Benchmarking/tflite_scripts_imgnt_accuracy_and_weight_extraction/fms/fms_2_32_112_112.txt",
 result2, 112, 112);
verify_fill_layer_input("/media/SSD2TB/wd/my_repos/DL_Benchmarking/tflite_scripts_imgnt_accuracy_and_weight_extraction/scratch_out/verify_2.txt",
 result2, 401408, 112, 112);
fill_dw_layer_weights(dw_weights_2, dw_weights_buffer, layer_2_dw_depth, layer_2_dw_filter_size, layer_2_dw_filter_size);
    dw_conv_3x3(dw_weights_buffer, channels, result2, 2, layer_2_dw_depth,
    layer_2_dw_ifm_width, layer_2_dw_ifm_height, layer_2_dw_num_of_tiles_in_d,
    layer_2_dw_num_of_tiles_h, layer_2_dw_num_of_tiles_w,
    layer_2_dw_strides, layer_2_dw_padding_left,layer_2_dw_padding_top,
    1);
dumb_layer_output("/media/SSD2TB/wd/my_repos/DL_Benchmarking/tflite_scripts_imgnt_accuracy_and_weight_extraction/scratch_out/ofms_2.txt",
 channels, 401408, 112, 112);
pw_conv(off_chip_weights, channels, result2, 3, layer_3_pw_depth,
    layer_3_pw_num_fils, layer_3_pw_num_of_tiles_in_d,
    layer_3_pw_num_of_tiles_out_d, layer_3_pw_num_of_tiles_h,
    layer_3_pw_num_of_tiles_w, tmp_channels, 0,
    layer_3_pw_num_of_weight_groups_for_one_pass,
    0, layer_3_pw_weights_offset, layer_3_relu);
dumb_layer_output("/media/SSD2TB/wd/my_repos/DL_Benchmarking/tflite_scripts_imgnt_accuracy_and_weight_extraction/scratch_out/ofms_3.txt",
 result2, 200704, 112, 112);
pw_conv(off_chip_weights, channels, result2, 4, layer_4_pw_depth,
    layer_4_pw_num_fils, layer_4_pw_num_of_tiles_in_d,
    layer_4_pw_num_of_tiles_out_d, layer_4_pw_num_of_tiles_h,
    layer_4_pw_num_of_tiles_w, tmp_channels, 0,
    layer_4_pw_num_of_weight_groups_for_one_pass,
    1, layer_4_pw_weights_offset, layer_4_relu);
dumb_layer_output("/media/SSD2TB/wd/my_repos/DL_Benchmarking/tflite_scripts_imgnt_accuracy_and_weight_extraction/scratch_out/ofms_4.txt",
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
    layer_6_pw_num_of_tiles_w, tmp_channels, 2,
    layer_6_pw_num_of_weight_groups_for_one_pass,
    1, layer_6_pw_weights_offset, layer_6_relu);
dumb_layer_output("/media/SSD2TB/wd/my_repos/DL_Benchmarking/tflite_scripts_imgnt_accuracy_and_weight_extraction/scratch_out/ofms_6.txt",
 channels, 75264, 56, 56);
pw_conv(off_chip_weights, channels, result2, 7, layer_7_pw_depth,
    layer_7_pw_num_fils, layer_7_pw_num_of_tiles_in_d,
    layer_7_pw_num_of_tiles_out_d, layer_7_pw_num_of_tiles_h,
    layer_7_pw_num_of_tiles_w, tmp_channels, 0,
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
dumb_layer_output("/media/SSD2TB/wd/my_repos/DL_Benchmarking/tflite_scripts_imgnt_accuracy_and_weight_extraction/scratch_out/ofms_8.txt",
 channels, 451584, 56, 56);
pw_conv(off_chip_weights, channels, result2, 9, layer_9_pw_depth,
    layer_9_pw_num_fils, layer_9_pw_num_of_tiles_in_d,
    layer_9_pw_num_of_tiles_out_d, layer_9_pw_num_of_tiles_h,
    layer_9_pw_num_of_tiles_w, tmp_channels, 1,
    layer_9_pw_num_of_weight_groups_for_one_pass,
    0, layer_9_pw_weights_offset, layer_9_relu);
dumb_layer_output("/media/SSD2TB/wd/my_repos/DL_Benchmarking/tflite_scripts_imgnt_accuracy_and_weight_extraction/scratch_out/ofms_9.txt",
 result2, 75264, 56, 56);
pw_conv(off_chip_weights, channels, result2, 10, layer_10_pw_depth,
    layer_10_pw_num_fils, layer_10_pw_num_of_tiles_in_d,
    layer_10_pw_num_of_tiles_out_d, layer_10_pw_num_of_tiles_h,
    layer_10_pw_num_of_tiles_w, tmp_channels, 0,
    layer_10_pw_num_of_weight_groups_for_one_pass,
    1, layer_10_pw_weights_offset, layer_10_relu);
dumb_layer_output("/media/SSD2TB/wd/my_repos/DL_Benchmarking/tflite_scripts_imgnt_accuracy_and_weight_extraction/scratch_out/ofms_10.txt",
 channels, 451584, 56, 56);
fill_dw_layer_weights(dw_weights_11, dw_weights_buffer, layer_11_dw_depth, layer_11_dw_filter_size, layer_11_dw_filter_size);
    dw_conv_3x3(dw_weights_buffer, channels, result2, 11, layer_11_dw_depth,
    layer_11_dw_ifm_width, layer_11_dw_ifm_height, layer_11_dw_num_of_tiles_in_d,
    layer_11_dw_num_of_tiles_h, layer_11_dw_num_of_tiles_w,
    layer_11_dw_strides, layer_11_dw_padding_left,layer_11_dw_padding_top,
    0);
dumb_layer_output("/media/SSD2TB/wd/my_repos/DL_Benchmarking/tflite_scripts_imgnt_accuracy_and_weight_extraction/scratch_out/ofms_11.txt",
 result2, 112896, 28, 28);
pw_conv(off_chip_weights, channels, result2, 12, layer_12_pw_depth,
    layer_12_pw_num_fils, layer_12_pw_num_of_tiles_in_d,
    layer_12_pw_num_of_tiles_out_d, layer_12_pw_num_of_tiles_h,
    layer_12_pw_num_of_tiles_w, tmp_channels, 2,
    layer_12_pw_num_of_weight_groups_for_one_pass,
    1, layer_12_pw_weights_offset, layer_12_relu);
dumb_layer_output("/media/SSD2TB/wd/my_repos/DL_Benchmarking/tflite_scripts_imgnt_accuracy_and_weight_extraction/scratch_out/ofms_12.txt",
 channels, 25088, 28, 28);
pw_conv(off_chip_weights, channels, result2, 13, layer_13_pw_depth,
    layer_13_pw_num_fils, layer_13_pw_num_of_tiles_in_d,
    layer_13_pw_num_of_tiles_out_d, layer_13_pw_num_of_tiles_h,
    layer_13_pw_num_of_tiles_w, tmp_channels, 0,
    layer_13_pw_num_of_weight_groups_for_one_pass,
    0, layer_13_pw_weights_offset, layer_13_relu);
dumb_layer_output("/media/SSD2TB/wd/my_repos/DL_Benchmarking/tflite_scripts_imgnt_accuracy_and_weight_extraction/scratch_out/ofms_13.txt",
 result2, 150528, 28, 28);
fill_dw_layer_weights(dw_weights_14, dw_weights_buffer, layer_14_dw_depth, layer_14_dw_filter_size, layer_14_dw_filter_size);
    dw_conv_3x3(dw_weights_buffer, channels, result2, 14, layer_14_dw_depth,
    layer_14_dw_ifm_width, layer_14_dw_ifm_height, layer_14_dw_num_of_tiles_in_d,
    layer_14_dw_num_of_tiles_h, layer_14_dw_num_of_tiles_w,
    layer_14_dw_strides, layer_14_dw_padding_left,layer_14_dw_padding_top,
    1);
dumb_layer_output("/media/SSD2TB/wd/my_repos/DL_Benchmarking/tflite_scripts_imgnt_accuracy_and_weight_extraction/scratch_out/ofms_14.txt",
 channels, 150528, 28, 28);
pw_conv(off_chip_weights, channels, result2, 15, layer_15_pw_depth,
    layer_15_pw_num_fils, layer_15_pw_num_of_tiles_in_d,
    layer_15_pw_num_of_tiles_out_d, layer_15_pw_num_of_tiles_h,
    layer_15_pw_num_of_tiles_w, tmp_channels, 3,
    layer_15_pw_num_of_weight_groups_for_one_pass,
    0, layer_15_pw_weights_offset, layer_15_relu);
dumb_layer_output("/media/SSD2TB/wd/my_repos/DL_Benchmarking/tflite_scripts_imgnt_accuracy_and_weight_extraction/scratch_out/ofms_15.txt",
 result2, 25088, 28, 28);
pw_conv(off_chip_weights, channels, result2, 16, layer_16_pw_depth,
    layer_16_pw_num_fils, layer_16_pw_num_of_tiles_in_d,
    layer_16_pw_num_of_tiles_out_d, layer_16_pw_num_of_tiles_h,
    layer_16_pw_num_of_tiles_w, tmp_channels, 0,
    layer_16_pw_num_of_weight_groups_for_one_pass,
    1, layer_16_pw_weights_offset, layer_16_relu);
dumb_layer_output("/media/SSD2TB/wd/my_repos/DL_Benchmarking/tflite_scripts_imgnt_accuracy_and_weight_extraction/scratch_out/ofms_16.txt",
 channels, 150528, 28, 28);
fill_dw_layer_weights(dw_weights_17, dw_weights_buffer, layer_17_dw_depth, layer_17_dw_filter_size, layer_17_dw_filter_size);
    dw_conv_3x3(dw_weights_buffer, channels, result2, 17, layer_17_dw_depth,
    layer_17_dw_ifm_width, layer_17_dw_ifm_height, layer_17_dw_num_of_tiles_in_d,
    layer_17_dw_num_of_tiles_h, layer_17_dw_num_of_tiles_w,
    layer_17_dw_strides, layer_17_dw_padding_left,layer_17_dw_padding_top,
    0);
dumb_layer_output("/media/SSD2TB/wd/my_repos/DL_Benchmarking/tflite_scripts_imgnt_accuracy_and_weight_extraction/scratch_out/ofms_17.txt",
 result2, 150528, 28, 28);
pw_conv(off_chip_weights, channels, result2, 18, layer_18_pw_depth,
    layer_18_pw_num_fils, layer_18_pw_num_of_tiles_in_d,
    layer_18_pw_num_of_tiles_out_d, layer_18_pw_num_of_tiles_h,
    layer_18_pw_num_of_tiles_w, tmp_channels, 1,
    layer_18_pw_num_of_weight_groups_for_one_pass,
    1, layer_18_pw_weights_offset, layer_18_relu);
dumb_layer_output("/media/SSD2TB/wd/my_repos/DL_Benchmarking/tflite_scripts_imgnt_accuracy_and_weight_extraction/scratch_out/ofms_18.txt",
 channels, 25088, 28, 28);
pw_conv(off_chip_weights, channels, result2, 19, layer_19_pw_depth,
    layer_19_pw_num_fils, layer_19_pw_num_of_tiles_in_d,
    layer_19_pw_num_of_tiles_out_d, layer_19_pw_num_of_tiles_h,
    layer_19_pw_num_of_tiles_w, tmp_channels, 0,
    layer_19_pw_num_of_weight_groups_for_one_pass,
    0, layer_19_pw_weights_offset, layer_19_relu);
dumb_layer_output("/media/SSD2TB/wd/my_repos/DL_Benchmarking/tflite_scripts_imgnt_accuracy_and_weight_extraction/scratch_out/ofms_19.txt",
 result2, 150528, 28, 28);
fill_dw_layer_weights(dw_weights_20, dw_weights_buffer, layer_20_dw_depth, layer_20_dw_filter_size, layer_20_dw_filter_size);
    dw_conv_3x3(dw_weights_buffer, channels, result2, 20, layer_20_dw_depth,
    layer_20_dw_ifm_width, layer_20_dw_ifm_height, layer_20_dw_num_of_tiles_in_d,
    layer_20_dw_num_of_tiles_h, layer_20_dw_num_of_tiles_w,
    layer_20_dw_strides, layer_20_dw_padding_left,layer_20_dw_padding_top,
    1);
dumb_layer_output("/media/SSD2TB/wd/my_repos/DL_Benchmarking/tflite_scripts_imgnt_accuracy_and_weight_extraction/scratch_out/ofms_20.txt",
 channels, 37632, 14, 14);
pw_conv(off_chip_weights, channels, result2, 21, layer_21_pw_depth,
    layer_21_pw_num_fils, layer_21_pw_num_of_tiles_in_d,
    layer_21_pw_num_of_tiles_out_d, layer_21_pw_num_of_tiles_h,
    layer_21_pw_num_of_tiles_w, tmp_channels, 2,
    layer_21_pw_num_of_weight_groups_for_one_pass,
    0, layer_21_pw_weights_offset, layer_21_relu);
dumb_layer_output("/media/SSD2TB/wd/my_repos/DL_Benchmarking/tflite_scripts_imgnt_accuracy_and_weight_extraction/scratch_out/ofms_21.txt",
 result2, 12544, 14, 14);
pw_conv(off_chip_weights, channels, result2, 22, layer_22_pw_depth,
    layer_22_pw_num_fils, layer_22_pw_num_of_tiles_in_d,
    layer_22_pw_num_of_tiles_out_d, layer_22_pw_num_of_tiles_h,
    layer_22_pw_num_of_tiles_w, tmp_channels, 0,
    layer_22_pw_num_of_weight_groups_for_one_pass,
    1, layer_22_pw_weights_offset, layer_22_relu);
dumb_layer_output("/media/SSD2TB/wd/my_repos/DL_Benchmarking/tflite_scripts_imgnt_accuracy_and_weight_extraction/scratch_out/ofms_22.txt",
 channels, 75264, 14, 14);
fill_dw_layer_weights(dw_weights_23, dw_weights_buffer, layer_23_dw_depth, layer_23_dw_filter_size, layer_23_dw_filter_size);
    dw_conv_3x3(dw_weights_buffer, channels, result2, 23, layer_23_dw_depth,
    layer_23_dw_ifm_width, layer_23_dw_ifm_height, layer_23_dw_num_of_tiles_in_d,
    layer_23_dw_num_of_tiles_h, layer_23_dw_num_of_tiles_w,
    layer_23_dw_strides, layer_23_dw_padding_left,layer_23_dw_padding_top,
    0);
dumb_layer_output("/media/SSD2TB/wd/my_repos/DL_Benchmarking/tflite_scripts_imgnt_accuracy_and_weight_extraction/scratch_out/ofms_23.txt",
 result2, 75264, 14, 14);
pw_conv(off_chip_weights, channels, result2, 24, layer_24_pw_depth,
    layer_24_pw_num_fils, layer_24_pw_num_of_tiles_in_d,
    layer_24_pw_num_of_tiles_out_d, layer_24_pw_num_of_tiles_h,
    layer_24_pw_num_of_tiles_w, tmp_channels, 3,
    layer_24_pw_num_of_weight_groups_for_one_pass,
    1, layer_24_pw_weights_offset, layer_24_relu);
dumb_layer_output("/media/SSD2TB/wd/my_repos/DL_Benchmarking/tflite_scripts_imgnt_accuracy_and_weight_extraction/scratch_out/ofms_24.txt",
 channels, 12544, 14, 14);
pw_conv(off_chip_weights, channels, result2, 25, layer_25_pw_depth,
    layer_25_pw_num_fils, layer_25_pw_num_of_tiles_in_d,
    layer_25_pw_num_of_tiles_out_d, layer_25_pw_num_of_tiles_h,
    layer_25_pw_num_of_tiles_w, tmp_channels, 0,
    layer_25_pw_num_of_weight_groups_for_one_pass,
    0, layer_25_pw_weights_offset, layer_25_relu);
dumb_layer_output("/media/SSD2TB/wd/my_repos/DL_Benchmarking/tflite_scripts_imgnt_accuracy_and_weight_extraction/scratch_out/ofms_25.txt",
 result2, 75264, 14, 14);
fill_dw_layer_weights(dw_weights_26, dw_weights_buffer, layer_26_dw_depth, layer_26_dw_filter_size, layer_26_dw_filter_size);
    dw_conv_3x3(dw_weights_buffer, channels, result2, 26, layer_26_dw_depth,
    layer_26_dw_ifm_width, layer_26_dw_ifm_height, layer_26_dw_num_of_tiles_in_d,
    layer_26_dw_num_of_tiles_h, layer_26_dw_num_of_tiles_w,
    layer_26_dw_strides, layer_26_dw_padding_left,layer_26_dw_padding_top,
    1);
dumb_layer_output("/media/SSD2TB/wd/my_repos/DL_Benchmarking/tflite_scripts_imgnt_accuracy_and_weight_extraction/scratch_out/ofms_26.txt",
 channels, 75264, 14, 14);
pw_conv(off_chip_weights, channels, result2, 27, layer_27_pw_depth,
    layer_27_pw_num_fils, layer_27_pw_num_of_tiles_in_d,
    layer_27_pw_num_of_tiles_out_d, layer_27_pw_num_of_tiles_h,
    layer_27_pw_num_of_tiles_w, tmp_channels, 3,
    layer_27_pw_num_of_weight_groups_for_one_pass,
    0, layer_27_pw_weights_offset, layer_27_relu);
dumb_layer_output("/media/SSD2TB/wd/my_repos/DL_Benchmarking/tflite_scripts_imgnt_accuracy_and_weight_extraction/scratch_out/ofms_27.txt",
 result2, 12544, 14, 14);
pw_conv(off_chip_weights, channels, result2, 28, layer_28_pw_depth,
    layer_28_pw_num_fils, layer_28_pw_num_of_tiles_in_d,
    layer_28_pw_num_of_tiles_out_d, layer_28_pw_num_of_tiles_h,
    layer_28_pw_num_of_tiles_w, tmp_channels, 0,
    layer_28_pw_num_of_weight_groups_for_one_pass,
    1, layer_28_pw_weights_offset, layer_28_relu);
dumb_layer_output("/media/SSD2TB/wd/my_repos/DL_Benchmarking/tflite_scripts_imgnt_accuracy_and_weight_extraction/scratch_out/ofms_28.txt",
 channels, 75264, 14, 14);
fill_dw_layer_weights(dw_weights_29, dw_weights_buffer, layer_29_dw_depth, layer_29_dw_filter_size, layer_29_dw_filter_size);
    dw_conv_3x3(dw_weights_buffer, channels, result2, 29, layer_29_dw_depth,
    layer_29_dw_ifm_width, layer_29_dw_ifm_height, layer_29_dw_num_of_tiles_in_d,
    layer_29_dw_num_of_tiles_h, layer_29_dw_num_of_tiles_w,
    layer_29_dw_strides, layer_29_dw_padding_left,layer_29_dw_padding_top,
    0);
dumb_layer_output("/media/SSD2TB/wd/my_repos/DL_Benchmarking/tflite_scripts_imgnt_accuracy_and_weight_extraction/scratch_out/ofms_29.txt",
 result2, 75264, 14, 14);
pw_conv(off_chip_weights, channels, result2, 30, layer_30_pw_depth,
    layer_30_pw_num_fils, layer_30_pw_num_of_tiles_in_d,
    layer_30_pw_num_of_tiles_out_d, layer_30_pw_num_of_tiles_h,
    layer_30_pw_num_of_tiles_w, tmp_channels, 1,
    layer_30_pw_num_of_weight_groups_for_one_pass,
    1, layer_30_pw_weights_offset, layer_30_relu);
dumb_layer_output("/media/SSD2TB/wd/my_repos/DL_Benchmarking/tflite_scripts_imgnt_accuracy_and_weight_extraction/scratch_out/ofms_30.txt",
 channels, 12544, 14, 14);
pw_conv(off_chip_weights, channels, result2, 31, layer_31_pw_depth,
    layer_31_pw_num_fils, layer_31_pw_num_of_tiles_in_d,
    layer_31_pw_num_of_tiles_out_d, layer_31_pw_num_of_tiles_h,
    layer_31_pw_num_of_tiles_w, tmp_channels, 0,
    layer_31_pw_num_of_weight_groups_for_one_pass,
    0, layer_31_pw_weights_offset, layer_31_relu);
fill_dw_layer_weights(dw_weights_32, dw_weights_buffer, layer_32_dw_depth, layer_32_dw_filter_size, layer_32_dw_filter_size);
    dw_conv_3x3(dw_weights_buffer, channels, result2, 32, layer_32_dw_depth,
    layer_32_dw_ifm_width, layer_32_dw_ifm_height, layer_32_dw_num_of_tiles_in_d,
    layer_32_dw_num_of_tiles_h, layer_32_dw_num_of_tiles_w,
    layer_32_dw_strides, layer_32_dw_padding_left,layer_32_dw_padding_top,
    1);
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
fill_dw_layer_weights(dw_weights_35, dw_weights_buffer, layer_35_dw_depth, layer_35_dw_filter_size, layer_35_dw_filter_size);
    dw_conv_3x3(dw_weights_buffer, channels, result2, 35, layer_35_dw_depth,
    layer_35_dw_ifm_width, layer_35_dw_ifm_height, layer_35_dw_num_of_tiles_in_d,
    layer_35_dw_num_of_tiles_h, layer_35_dw_num_of_tiles_w,
    layer_35_dw_strides, layer_35_dw_padding_left,layer_35_dw_padding_top,
    0);
pw_conv(off_chip_weights, channels, result2, 36, layer_36_pw_depth,
    layer_36_pw_num_fils, layer_36_pw_num_of_tiles_in_d,
    layer_36_pw_num_of_tiles_out_d, layer_36_pw_num_of_tiles_h,
    layer_36_pw_num_of_tiles_w, tmp_channels, 3,
    layer_36_pw_num_of_weight_groups_for_one_pass,
    1, layer_36_pw_weights_offset, layer_36_relu);
pw_conv(off_chip_weights, channels, result2, 37, layer_37_pw_depth,
    layer_37_pw_num_fils, layer_37_pw_num_of_tiles_in_d,
    layer_37_pw_num_of_tiles_out_d, layer_37_pw_num_of_tiles_h,
    layer_37_pw_num_of_tiles_w, tmp_channels, 0,
    layer_37_pw_num_of_weight_groups_for_one_pass,
    0, layer_37_pw_weights_offset, layer_37_relu);
dumb_layer_output("/media/SSD2TB/wd/my_repos/DL_Benchmarking/tflite_scripts_imgnt_accuracy_and_weight_extraction/scratch_out/ofms_37.txt",
 result2, 112896, 14, 14);
fill_dw_layer_weights(dw_weights_38, dw_weights_buffer, layer_38_dw_depth, layer_38_dw_filter_size, layer_38_dw_filter_size);
    dw_conv_3x3(dw_weights_buffer, channels, result2, 38, layer_38_dw_depth,
    layer_38_dw_ifm_width, layer_38_dw_ifm_height, layer_38_dw_num_of_tiles_in_d,
    layer_38_dw_num_of_tiles_h, layer_38_dw_num_of_tiles_w,
    layer_38_dw_strides, layer_38_dw_padding_left,layer_38_dw_padding_top,
    1);
pw_conv(off_chip_weights, channels, result2, 39, layer_39_pw_depth,
    layer_39_pw_num_fils, layer_39_pw_num_of_tiles_in_d,
    layer_39_pw_num_of_tiles_out_d, layer_39_pw_num_of_tiles_h,
    layer_39_pw_num_of_tiles_w, tmp_channels, 1,
    layer_39_pw_num_of_weight_groups_for_one_pass,
    0, layer_39_pw_weights_offset, layer_39_relu);
dumb_layer_output("/media/SSD2TB/wd/my_repos/DL_Benchmarking/tflite_scripts_imgnt_accuracy_and_weight_extraction/scratch_out/ofms_39.txt",
 result2, 18816, 14, 14);
pw_conv(off_chip_weights, channels, result2, 40, layer_40_pw_depth,
    layer_40_pw_num_fils, layer_40_pw_num_of_tiles_in_d,
    layer_40_pw_num_of_tiles_out_d, layer_40_pw_num_of_tiles_h,
    layer_40_pw_num_of_tiles_w, tmp_channels, 0,
    layer_40_pw_num_of_weight_groups_for_one_pass,
    1, layer_40_pw_weights_offset, layer_40_relu);
dumb_layer_output("/media/SSD2TB/wd/my_repos/DL_Benchmarking/tflite_scripts_imgnt_accuracy_and_weight_extraction/scratch_out/ofms_40.txt",
 channels, 112896, 14, 14);
fill_dw_layer_weights(dw_weights_41, dw_weights_buffer, layer_41_dw_depth, layer_41_dw_filter_size, layer_41_dw_filter_size);
    dw_conv_3x3(dw_weights_buffer, channels, result2, 41, layer_41_dw_depth,
    layer_41_dw_ifm_width, layer_41_dw_ifm_height, layer_41_dw_num_of_tiles_in_d,
    layer_41_dw_num_of_tiles_h, layer_41_dw_num_of_tiles_w,
    layer_41_dw_strides, layer_41_dw_padding_left,layer_41_dw_padding_top,
    0);
dumb_layer_output("/media/SSD2TB/wd/my_repos/DL_Benchmarking/tflite_scripts_imgnt_accuracy_and_weight_extraction/scratch_out/ofms_41.txt",
 result2, 28224, 7, 7);
pw_conv(off_chip_weights, channels, result2, 42, layer_42_pw_depth,
    layer_42_pw_num_fils, layer_42_pw_num_of_tiles_in_d,
    layer_42_pw_num_of_tiles_out_d, layer_42_pw_num_of_tiles_h,
    layer_42_pw_num_of_tiles_w, tmp_channels, 2,
    layer_42_pw_num_of_weight_groups_for_one_pass,
    1, layer_42_pw_weights_offset, layer_42_relu);
dumb_layer_output("/media/SSD2TB/wd/my_repos/DL_Benchmarking/tflite_scripts_imgnt_accuracy_and_weight_extraction/scratch_out/ofms_42.txt",
 channels, 7840, 7, 7);
pw_conv(off_chip_weights, channels, result2, 43, layer_43_pw_depth,
    layer_43_pw_num_fils, layer_43_pw_num_of_tiles_in_d,
    layer_43_pw_num_of_tiles_out_d, layer_43_pw_num_of_tiles_h,
    layer_43_pw_num_of_tiles_w, tmp_channels, 0,
    layer_43_pw_num_of_weight_groups_for_one_pass,
    0, layer_43_pw_weights_offset, layer_43_relu);
dumb_layer_output("/media/SSD2TB/wd/my_repos/DL_Benchmarking/tflite_scripts_imgnt_accuracy_and_weight_extraction/scratch_out/ofms_43.txt",
 result2, 47040, 7, 7);
fill_dw_layer_weights(dw_weights_44, dw_weights_buffer, layer_44_dw_depth, layer_44_dw_filter_size, layer_44_dw_filter_size);
    dw_conv_3x3(dw_weights_buffer, channels, result2, 44, layer_44_dw_depth,
    layer_44_dw_ifm_width, layer_44_dw_ifm_height, layer_44_dw_num_of_tiles_in_d,
    layer_44_dw_num_of_tiles_h, layer_44_dw_num_of_tiles_w,
    layer_44_dw_strides, layer_44_dw_padding_left,layer_44_dw_padding_top,
    1);
dumb_layer_output("/media/SSD2TB/wd/my_repos/DL_Benchmarking/tflite_scripts_imgnt_accuracy_and_weight_extraction/scratch_out/ofms_44.txt",
 channels, 47040, 7, 7);
pw_conv(off_chip_weights, channels, result2, 45, layer_45_pw_depth,
    layer_45_pw_num_fils, layer_45_pw_num_of_tiles_in_d,
    layer_45_pw_num_of_tiles_out_d, layer_45_pw_num_of_tiles_h,
    layer_45_pw_num_of_tiles_w, tmp_channels, 3,
    layer_45_pw_num_of_weight_groups_for_one_pass,
    0, layer_45_pw_weights_offset, layer_45_relu);
dumb_layer_output("/media/SSD2TB/wd/my_repos/DL_Benchmarking/tflite_scripts_imgnt_accuracy_and_weight_extraction/scratch_out/ofms_45.txt",
 result2, 7840, 7, 7);
pw_conv(off_chip_weights, channels, result2, 46, layer_46_pw_depth,
    layer_46_pw_num_fils, layer_46_pw_num_of_tiles_in_d,
    layer_46_pw_num_of_tiles_out_d, layer_46_pw_num_of_tiles_h,
    layer_46_pw_num_of_tiles_w, tmp_channels, 0,
    layer_46_pw_num_of_weight_groups_for_one_pass,
    1, layer_46_pw_weights_offset, layer_46_relu);
dumb_layer_output("/media/SSD2TB/wd/my_repos/DL_Benchmarking/tflite_scripts_imgnt_accuracy_and_weight_extraction/scratch_out/ofms_46.txt",
 channels, 47040, 7, 7);
fill_dw_layer_weights(dw_weights_47, dw_weights_buffer, layer_47_dw_depth, layer_47_dw_filter_size, layer_47_dw_filter_size);
    dw_conv_3x3(dw_weights_buffer, channels, result2, 47, layer_47_dw_depth,
    layer_47_dw_ifm_width, layer_47_dw_ifm_height, layer_47_dw_num_of_tiles_in_d,
    layer_47_dw_num_of_tiles_h, layer_47_dw_num_of_tiles_w,
    layer_47_dw_strides, layer_47_dw_padding_left,layer_47_dw_padding_top,
    0);
dumb_layer_output("/media/SSD2TB/wd/my_repos/DL_Benchmarking/tflite_scripts_imgnt_accuracy_and_weight_extraction/scratch_out/ofms_47.txt",
 result2, 47040, 7, 7);
pw_conv(off_chip_weights, channels, result2, 48, layer_48_pw_depth,
    layer_48_pw_num_fils, layer_48_pw_num_of_tiles_in_d,
    layer_48_pw_num_of_tiles_out_d, layer_48_pw_num_of_tiles_h,
    layer_48_pw_num_of_tiles_w, tmp_channels, 1,
    layer_48_pw_num_of_weight_groups_for_one_pass,
    1, layer_48_pw_weights_offset, layer_48_relu);
dumb_layer_output("/media/SSD2TB/wd/my_repos/DL_Benchmarking/tflite_scripts_imgnt_accuracy_and_weight_extraction/scratch_out/ofms_48.txt",
 channels, 7840, 7, 7);
pw_conv(off_chip_weights, channels, result2, 49, layer_49_pw_depth,
    layer_49_pw_num_fils, layer_49_pw_num_of_tiles_in_d,
    layer_49_pw_num_of_tiles_out_d, layer_49_pw_num_of_tiles_h,
    layer_49_pw_num_of_tiles_w, tmp_channels, 0,
    layer_49_pw_num_of_weight_groups_for_one_pass,
    0, layer_49_pw_weights_offset, layer_49_relu);
dumb_layer_output("/media/SSD2TB/wd/my_repos/DL_Benchmarking/tflite_scripts_imgnt_accuracy_and_weight_extraction/scratch_out/ofms_49.txt",
 result2, 47040, 7, 7);
fill_dw_layer_weights(dw_weights_50, dw_weights_buffer, layer_50_dw_depth, layer_50_dw_filter_size, layer_50_dw_filter_size);
    dw_conv_3x3(dw_weights_buffer, channels, result2, 50, layer_50_dw_depth,
    layer_50_dw_ifm_width, layer_50_dw_ifm_height, layer_50_dw_num_of_tiles_in_d,
    layer_50_dw_num_of_tiles_h, layer_50_dw_num_of_tiles_w,
    layer_50_dw_strides, layer_50_dw_padding_left,layer_50_dw_padding_top,
    1);
dumb_layer_output("/media/SSD2TB/wd/my_repos/DL_Benchmarking/tflite_scripts_imgnt_accuracy_and_weight_extraction/scratch_out/ofms_50.txt",
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
