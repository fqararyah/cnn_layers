#include "../../basic_defs/basic_defs_glue.h"
#ifndef LAYERS_SPECS
#define LAYERS_SPECS
//****************************
 const int layer_0_s_num_fils = 32 / alpha;
const int layer_0_s_depth = input_image_depth;
const int layer_0_s_ifm_height = input_image_height;
const int layer_0_s_ifm_width = input_image_width;
const int layer_0_s_strides = 2;
const int layer_0_s_ofm_height = layer_0_s_ifm_height / layer_0_s_strides;
const int layer_0_s_ofm_width = layer_0_s_ifm_width / layer_0_s_strides;
const int layer_0_s_num_of_tiles_out_d = int(0.99 + ((float) layer_0_s_num_fils) / pw_conv_parallelism_out);
const int layer_0_s_padding_left = 0;
const int layer_0_s_padding_right = 1;
const int layer_0_s_padding_top = 0;
 const int layer_0_s_padding_bottom = 1;
 const int layer_0_s_filter_dim = 3;
 const int layer_0_s_num_of_tiles_w = layer_0_s_ofm_width / pw_tile_w; 
 const int layer_0_s_num_of_tiles_h = layer_0_s_ofm_height / pw_tile_h; 
 const int layer_0_s_num_of_tiles_d_in = layer_0_s_depth / pw_tile_d; 
 //****************************
const int layer_1_dw_num_fils = layer_0_s_num_fils / alpha;
 const int layer_1_dw_depth = layer_1_dw_num_fils;
 const int layer_1_dw_strides = 1;
 const int layer_1_dw_ifm_height = layer_0_s_ofm_height;
 const int layer_1_dw_ifm_width = layer_0_s_ofm_width;
 const int layer_1_dw_ofm_height = layer_1_dw_ifm_height / layer_1_dw_strides;
 const int layer_1_dw_ofm_width = layer_1_dw_ifm_width / layer_1_dw_strides;
 const int layer_1_dw_padding_left = 1;
 const int layer_1_dw_padding_right = 1;
 const int layer_1_dw_padding_top = 1;
 const int layer_1_dw_padding_bottom = 1;
 const int layer_1_dw_filter_size = 3;
 const int layer_1_dw_num_of_tiles_in_d = (int)(0.99 + (float)layer_1_dw_depth / dw_tile_d);
 const int layer_1_dw_ifm_num_of_tiles_w = (int)(0.99 + (float)layer_1_dw_ifm_width / dw_tile_w); 
 const int layer_1_dw_ifm_num_of_tiles_h = (int)(0.99 + (float)layer_1_dw_ifm_height / dw_tile_h); 
 const int layer_1_dw_num_of_tiles_w = (int)(0.99 + (float)layer_1_dw_ofm_width / dw_tile_w); 
 const int layer_1_dw_num_of_tiles_h = (int)(0.99 + (float)layer_1_dw_ofm_height / dw_tile_h); 
 //****************************
//****************************
 const int layer_2_pw_num_fils = 16 / alpha;
 const int layer_2_pw_depth = layer_1_dw_num_fils;
 const int layer_2_pw_ifm_height = layer_1_dw_ofm_height;
 const int layer_2_pw_ifm_width = layer_1_dw_ofm_width;
 const int layer_2_pw_ofm_height = layer_2_pw_ifm_height;
 const int layer_2_pw_ofm_width = layer_2_pw_ifm_width;
 const int layer_2_pw_num_of_tiles_in_d = (int)(0.99 + (float)layer_2_pw_depth / pw_tile_d);
 const int layer_2_pw_num_of_tiles_out_d = (int)(0.99 + (float)layer_2_pw_num_fils / pw_conv_parallelism_out);
 const int layer_2_pw_num_of_tiles_w = (int)(0.99 + (float)layer_2_pw_ofm_width / pw_tile_w); 
 const int layer_2_pw_num_of_tiles_h = (int)(0.99 + (float)layer_2_pw_ofm_height / pw_tile_h); 
 const int layer_2_pw_num_of_weight_groups_for_one_pass = layer_2_pw_depth * pw_conv_parallelism_out / weights_group_items; 
 const int layer_2_pw_weights_offset = 0; 
 const int layer_2_activation = 0;
//****************************
//****************************
 const int layer_3_pw_num_fils = 96 / alpha;
 const int layer_3_pw_depth = layer_2_pw_num_fils;
 const int layer_3_pw_ifm_height = layer_2_pw_ofm_height;
 const int layer_3_pw_ifm_width = layer_2_pw_ofm_width;
 const int layer_3_pw_ofm_height = layer_3_pw_ifm_height;
 const int layer_3_pw_ofm_width = layer_3_pw_ifm_width;
 const int layer_3_pw_num_of_tiles_in_d = (int)(0.99 + (float)layer_3_pw_depth / pw_tile_d);
 const int layer_3_pw_num_of_tiles_out_d = (int)(0.99 + (float)layer_3_pw_num_fils / pw_conv_parallelism_out);
 const int layer_3_pw_num_of_tiles_w = (int)(0.99 + (float)layer_3_pw_ofm_width / pw_tile_w); 
 const int layer_3_pw_num_of_tiles_h = (int)(0.99 + (float)layer_3_pw_ofm_height / pw_tile_h); 
 const int layer_3_pw_num_of_weight_groups_for_one_pass = layer_3_pw_depth * pw_conv_parallelism_out / weights_group_items; 
 const int layer_3_pw_weights_offset = 8; 
 const int layer_3_activation = 6;
//****************************
const int layer_4_dw_num_fils = layer_3_pw_num_fils / alpha;
 const int layer_4_dw_depth = layer_4_dw_num_fils;
 const int layer_4_dw_strides = 2;
 const int layer_4_dw_ifm_height = layer_3_pw_ofm_height;
 const int layer_4_dw_ifm_width = layer_3_pw_ofm_width;
 const int layer_4_dw_ofm_height = layer_4_dw_ifm_height / layer_4_dw_strides;
 const int layer_4_dw_ofm_width = layer_4_dw_ifm_width / layer_4_dw_strides;
 const int layer_4_dw_padding_left = 0;
 const int layer_4_dw_padding_right = 1;
 const int layer_4_dw_padding_top = 0;
 const int layer_4_dw_padding_bottom = 1;
 const int layer_4_dw_filter_size = 3;
 const int layer_4_dw_num_of_tiles_in_d = (int)(0.99 + (float)layer_4_dw_depth / dw_tile_d);
 const int layer_4_dw_ifm_num_of_tiles_w = (int)(0.99 + (float)layer_4_dw_ifm_width / dw_tile_w); 
 const int layer_4_dw_ifm_num_of_tiles_h = (int)(0.99 + (float)layer_4_dw_ifm_height / dw_tile_h); 
 const int layer_4_dw_num_of_tiles_w = (int)(0.99 + (float)layer_4_dw_ofm_width / dw_tile_w); 
 const int layer_4_dw_num_of_tiles_h = (int)(0.99 + (float)layer_4_dw_ofm_height / dw_tile_h); 
 //****************************
//****************************
 const int layer_5_pw_num_fils = 24 / alpha;
 const int layer_5_pw_depth = layer_4_dw_num_fils;
 const int layer_5_pw_ifm_height = layer_4_dw_ofm_height;
 const int layer_5_pw_ifm_width = layer_4_dw_ofm_width;
 const int layer_5_pw_ofm_height = layer_5_pw_ifm_height;
 const int layer_5_pw_ofm_width = layer_5_pw_ifm_width;
 const int layer_5_pw_num_of_tiles_in_d = (int)(0.99 + (float)layer_5_pw_depth / pw_tile_d);
 const int layer_5_pw_num_of_tiles_out_d = (int)(0.99 + (float)layer_5_pw_num_fils / pw_conv_parallelism_out);
 const int layer_5_pw_num_of_tiles_w = (int)(0.99 + (float)layer_5_pw_ofm_width / pw_tile_w); 
 const int layer_5_pw_num_of_tiles_h = (int)(0.99 + (float)layer_5_pw_ofm_height / pw_tile_h); 
 const int layer_5_pw_num_of_weight_groups_for_one_pass = layer_5_pw_depth * pw_conv_parallelism_out / weights_group_items; 
 const int layer_5_pw_weights_offset = 32; 
 const int layer_5_activation = 0;
//****************************
//****************************
 const int layer_6_pw_num_fils = 144 / alpha;
 const int layer_6_pw_depth = layer_5_pw_num_fils;
 const int layer_6_pw_ifm_height = layer_5_pw_ofm_height;
 const int layer_6_pw_ifm_width = layer_5_pw_ofm_width;
 const int layer_6_pw_ofm_height = layer_6_pw_ifm_height;
 const int layer_6_pw_ofm_width = layer_6_pw_ifm_width;
 const int layer_6_pw_num_of_tiles_in_d = (int)(0.99 + (float)layer_6_pw_depth / pw_tile_d);
 const int layer_6_pw_num_of_tiles_out_d = (int)(0.99 + (float)layer_6_pw_num_fils / pw_conv_parallelism_out);
 const int layer_6_pw_num_of_tiles_w = (int)(0.99 + (float)layer_6_pw_ofm_width / pw_tile_w); 
 const int layer_6_pw_num_of_tiles_h = (int)(0.99 + (float)layer_6_pw_ofm_height / pw_tile_h); 
 const int layer_6_pw_num_of_weight_groups_for_one_pass = layer_6_pw_depth * pw_conv_parallelism_out / weights_group_items; 
 const int layer_6_pw_weights_offset = 68; 
 const int layer_6_activation = 6;
//****************************
const int layer_7_dw_num_fils = layer_6_pw_num_fils / alpha;
 const int layer_7_dw_depth = layer_7_dw_num_fils;
 const int layer_7_dw_strides = 1;
 const int layer_7_dw_ifm_height = layer_6_pw_ofm_height;
 const int layer_7_dw_ifm_width = layer_6_pw_ofm_width;
 const int layer_7_dw_ofm_height = layer_7_dw_ifm_height / layer_7_dw_strides;
 const int layer_7_dw_ofm_width = layer_7_dw_ifm_width / layer_7_dw_strides;
 const int layer_7_dw_padding_left = 1;
 const int layer_7_dw_padding_right = 1;
 const int layer_7_dw_padding_top = 1;
 const int layer_7_dw_padding_bottom = 1;
 const int layer_7_dw_filter_size = 3;
 const int layer_7_dw_num_of_tiles_in_d = (int)(0.99 + (float)layer_7_dw_depth / dw_tile_d);
 const int layer_7_dw_ifm_num_of_tiles_w = (int)(0.99 + (float)layer_7_dw_ifm_width / dw_tile_w); 
 const int layer_7_dw_ifm_num_of_tiles_h = (int)(0.99 + (float)layer_7_dw_ifm_height / dw_tile_h); 
 const int layer_7_dw_num_of_tiles_w = (int)(0.99 + (float)layer_7_dw_ofm_width / dw_tile_w); 
 const int layer_7_dw_num_of_tiles_h = (int)(0.99 + (float)layer_7_dw_ofm_height / dw_tile_h); 
 //****************************
//****************************
 const int layer_8_pw_num_fils = 24 / alpha;
 const int layer_8_pw_depth = layer_7_dw_num_fils;
 const int layer_8_pw_ifm_height = layer_7_dw_ofm_height;
 const int layer_8_pw_ifm_width = layer_7_dw_ofm_width;
 const int layer_8_pw_ofm_height = layer_8_pw_ifm_height;
 const int layer_8_pw_ofm_width = layer_8_pw_ifm_width;
 const int layer_8_pw_num_of_tiles_in_d = (int)(0.99 + (float)layer_8_pw_depth / pw_tile_d);
 const int layer_8_pw_num_of_tiles_out_d = (int)(0.99 + (float)layer_8_pw_num_fils / pw_conv_parallelism_out);
 const int layer_8_pw_num_of_tiles_w = (int)(0.99 + (float)layer_8_pw_ofm_width / pw_tile_w); 
 const int layer_8_pw_num_of_tiles_h = (int)(0.99 + (float)layer_8_pw_ofm_height / pw_tile_h); 
 const int layer_8_pw_num_of_weight_groups_for_one_pass = layer_8_pw_depth * pw_conv_parallelism_out / weights_group_items; 
 const int layer_8_pw_weights_offset = 122; 
 const int layer_8_activation = 0;
//****************************
//****************************
 const int layer_9_pw_num_fils = 144 / alpha;
 const int layer_9_pw_depth = layer_8_pw_num_fils;
 const int layer_9_pw_ifm_height = layer_8_pw_ofm_height;
 const int layer_9_pw_ifm_width = layer_8_pw_ofm_width;
 const int layer_9_pw_ofm_height = layer_9_pw_ifm_height;
 const int layer_9_pw_ofm_width = layer_9_pw_ifm_width;
 const int layer_9_pw_num_of_tiles_in_d = (int)(0.99 + (float)layer_9_pw_depth / pw_tile_d);
 const int layer_9_pw_num_of_tiles_out_d = (int)(0.99 + (float)layer_9_pw_num_fils / pw_conv_parallelism_out);
 const int layer_9_pw_num_of_tiles_w = (int)(0.99 + (float)layer_9_pw_ofm_width / pw_tile_w); 
 const int layer_9_pw_num_of_tiles_h = (int)(0.99 + (float)layer_9_pw_ofm_height / pw_tile_h); 
 const int layer_9_pw_num_of_weight_groups_for_one_pass = layer_9_pw_depth * pw_conv_parallelism_out / weights_group_items; 
 const int layer_9_pw_weights_offset = 176; 
 const int layer_9_activation = 6;
//****************************
const int layer_10_dw_num_fils = layer_9_pw_num_fils / alpha;
 const int layer_10_dw_depth = layer_10_dw_num_fils;
 const int layer_10_dw_strides = 2;
 const int layer_10_dw_ifm_height = layer_9_pw_ofm_height;
 const int layer_10_dw_ifm_width = layer_9_pw_ofm_width;
 const int layer_10_dw_ofm_height = layer_10_dw_ifm_height / layer_10_dw_strides;
 const int layer_10_dw_ofm_width = layer_10_dw_ifm_width / layer_10_dw_strides;
 const int layer_10_dw_padding_left = 0;
 const int layer_10_dw_padding_right = 1;
 const int layer_10_dw_padding_top = 0;
 const int layer_10_dw_padding_bottom = 1;
 const int layer_10_dw_filter_size = 3;
 const int layer_10_dw_num_of_tiles_in_d = (int)(0.99 + (float)layer_10_dw_depth / dw_tile_d);
 const int layer_10_dw_ifm_num_of_tiles_w = (int)(0.99 + (float)layer_10_dw_ifm_width / dw_tile_w); 
 const int layer_10_dw_ifm_num_of_tiles_h = (int)(0.99 + (float)layer_10_dw_ifm_height / dw_tile_h); 
 const int layer_10_dw_num_of_tiles_w = (int)(0.99 + (float)layer_10_dw_ofm_width / dw_tile_w); 
 const int layer_10_dw_num_of_tiles_h = (int)(0.99 + (float)layer_10_dw_ofm_height / dw_tile_h); 
 //****************************
//****************************
 const int layer_11_pw_num_fils = 32 / alpha;
 const int layer_11_pw_depth = layer_10_dw_num_fils;
 const int layer_11_pw_ifm_height = layer_10_dw_ofm_height;
 const int layer_11_pw_ifm_width = layer_10_dw_ofm_width;
 const int layer_11_pw_ofm_height = layer_11_pw_ifm_height;
 const int layer_11_pw_ofm_width = layer_11_pw_ifm_width;
 const int layer_11_pw_num_of_tiles_in_d = (int)(0.99 + (float)layer_11_pw_depth / pw_tile_d);
 const int layer_11_pw_num_of_tiles_out_d = (int)(0.99 + (float)layer_11_pw_num_fils / pw_conv_parallelism_out);
 const int layer_11_pw_num_of_tiles_w = (int)(0.99 + (float)layer_11_pw_ofm_width / pw_tile_w); 
 const int layer_11_pw_num_of_tiles_h = (int)(0.99 + (float)layer_11_pw_ofm_height / pw_tile_h); 
 const int layer_11_pw_num_of_weight_groups_for_one_pass = layer_11_pw_depth * pw_conv_parallelism_out / weights_group_items; 
 const int layer_11_pw_weights_offset = 230; 
 const int layer_11_activation = 0;
//****************************
//****************************
 const int layer_12_pw_num_fils = 192 / alpha;
 const int layer_12_pw_depth = layer_11_pw_num_fils;
 const int layer_12_pw_ifm_height = layer_11_pw_ofm_height;
 const int layer_12_pw_ifm_width = layer_11_pw_ofm_width;
 const int layer_12_pw_ofm_height = layer_12_pw_ifm_height;
 const int layer_12_pw_ofm_width = layer_12_pw_ifm_width;
 const int layer_12_pw_num_of_tiles_in_d = (int)(0.99 + (float)layer_12_pw_depth / pw_tile_d);
 const int layer_12_pw_num_of_tiles_out_d = (int)(0.99 + (float)layer_12_pw_num_fils / pw_conv_parallelism_out);
 const int layer_12_pw_num_of_tiles_w = (int)(0.99 + (float)layer_12_pw_ofm_width / pw_tile_w); 
 const int layer_12_pw_num_of_tiles_h = (int)(0.99 + (float)layer_12_pw_ofm_height / pw_tile_h); 
 const int layer_12_pw_num_of_weight_groups_for_one_pass = layer_12_pw_depth * pw_conv_parallelism_out / weights_group_items; 
 const int layer_12_pw_weights_offset = 302; 
 const int layer_12_activation = 6;
//****************************
const int layer_13_dw_num_fils = layer_12_pw_num_fils / alpha;
 const int layer_13_dw_depth = layer_13_dw_num_fils;
 const int layer_13_dw_strides = 1;
 const int layer_13_dw_ifm_height = layer_12_pw_ofm_height;
 const int layer_13_dw_ifm_width = layer_12_pw_ofm_width;
 const int layer_13_dw_ofm_height = layer_13_dw_ifm_height / layer_13_dw_strides;
 const int layer_13_dw_ofm_width = layer_13_dw_ifm_width / layer_13_dw_strides;
 const int layer_13_dw_padding_left = 1;
 const int layer_13_dw_padding_right = 1;
 const int layer_13_dw_padding_top = 1;
 const int layer_13_dw_padding_bottom = 1;
 const int layer_13_dw_filter_size = 3;
 const int layer_13_dw_num_of_tiles_in_d = (int)(0.99 + (float)layer_13_dw_depth / dw_tile_d);
 const int layer_13_dw_ifm_num_of_tiles_w = (int)(0.99 + (float)layer_13_dw_ifm_width / dw_tile_w); 
 const int layer_13_dw_ifm_num_of_tiles_h = (int)(0.99 + (float)layer_13_dw_ifm_height / dw_tile_h); 
 const int layer_13_dw_num_of_tiles_w = (int)(0.99 + (float)layer_13_dw_ofm_width / dw_tile_w); 
 const int layer_13_dw_num_of_tiles_h = (int)(0.99 + (float)layer_13_dw_ofm_height / dw_tile_h); 
 //****************************
//****************************
 const int layer_14_pw_num_fils = 32 / alpha;
 const int layer_14_pw_depth = layer_13_dw_num_fils;
 const int layer_14_pw_ifm_height = layer_13_dw_ofm_height;
 const int layer_14_pw_ifm_width = layer_13_dw_ofm_width;
 const int layer_14_pw_ofm_height = layer_14_pw_ifm_height;
 const int layer_14_pw_ofm_width = layer_14_pw_ifm_width;
 const int layer_14_pw_num_of_tiles_in_d = (int)(0.99 + (float)layer_14_pw_depth / pw_tile_d);
 const int layer_14_pw_num_of_tiles_out_d = (int)(0.99 + (float)layer_14_pw_num_fils / pw_conv_parallelism_out);
 const int layer_14_pw_num_of_tiles_w = (int)(0.99 + (float)layer_14_pw_ofm_width / pw_tile_w); 
 const int layer_14_pw_num_of_tiles_h = (int)(0.99 + (float)layer_14_pw_ofm_height / pw_tile_h); 
 const int layer_14_pw_num_of_weight_groups_for_one_pass = layer_14_pw_depth * pw_conv_parallelism_out / weights_group_items; 
 const int layer_14_pw_weights_offset = 398; 
 const int layer_14_activation = 0;
//****************************
//****************************
 const int layer_15_pw_num_fils = 192 / alpha;
 const int layer_15_pw_depth = layer_14_pw_num_fils;
 const int layer_15_pw_ifm_height = layer_14_pw_ofm_height;
 const int layer_15_pw_ifm_width = layer_14_pw_ofm_width;
 const int layer_15_pw_ofm_height = layer_15_pw_ifm_height;
 const int layer_15_pw_ofm_width = layer_15_pw_ifm_width;
 const int layer_15_pw_num_of_tiles_in_d = (int)(0.99 + (float)layer_15_pw_depth / pw_tile_d);
 const int layer_15_pw_num_of_tiles_out_d = (int)(0.99 + (float)layer_15_pw_num_fils / pw_conv_parallelism_out);
 const int layer_15_pw_num_of_tiles_w = (int)(0.99 + (float)layer_15_pw_ofm_width / pw_tile_w); 
 const int layer_15_pw_num_of_tiles_h = (int)(0.99 + (float)layer_15_pw_ofm_height / pw_tile_h); 
 const int layer_15_pw_num_of_weight_groups_for_one_pass = layer_15_pw_depth * pw_conv_parallelism_out / weights_group_items; 
 const int layer_15_pw_weights_offset = 494; 
 const int layer_15_activation = 6;
//****************************
const int layer_16_dw_num_fils = layer_15_pw_num_fils / alpha;
 const int layer_16_dw_depth = layer_16_dw_num_fils;
 const int layer_16_dw_strides = 1;
 const int layer_16_dw_ifm_height = layer_15_pw_ofm_height;
 const int layer_16_dw_ifm_width = layer_15_pw_ofm_width;
 const int layer_16_dw_ofm_height = layer_16_dw_ifm_height / layer_16_dw_strides;
 const int layer_16_dw_ofm_width = layer_16_dw_ifm_width / layer_16_dw_strides;
 const int layer_16_dw_padding_left = 1;
 const int layer_16_dw_padding_right = 1;
 const int layer_16_dw_padding_top = 1;
 const int layer_16_dw_padding_bottom = 1;
 const int layer_16_dw_filter_size = 3;
 const int layer_16_dw_num_of_tiles_in_d = (int)(0.99 + (float)layer_16_dw_depth / dw_tile_d);
 const int layer_16_dw_ifm_num_of_tiles_w = (int)(0.99 + (float)layer_16_dw_ifm_width / dw_tile_w); 
 const int layer_16_dw_ifm_num_of_tiles_h = (int)(0.99 + (float)layer_16_dw_ifm_height / dw_tile_h); 
 const int layer_16_dw_num_of_tiles_w = (int)(0.99 + (float)layer_16_dw_ofm_width / dw_tile_w); 
 const int layer_16_dw_num_of_tiles_h = (int)(0.99 + (float)layer_16_dw_ofm_height / dw_tile_h); 
 //****************************
//****************************
 const int layer_17_pw_num_fils = 32 / alpha;
 const int layer_17_pw_depth = layer_16_dw_num_fils;
 const int layer_17_pw_ifm_height = layer_16_dw_ofm_height;
 const int layer_17_pw_ifm_width = layer_16_dw_ofm_width;
 const int layer_17_pw_ofm_height = layer_17_pw_ifm_height;
 const int layer_17_pw_ofm_width = layer_17_pw_ifm_width;
 const int layer_17_pw_num_of_tiles_in_d = (int)(0.99 + (float)layer_17_pw_depth / pw_tile_d);
 const int layer_17_pw_num_of_tiles_out_d = (int)(0.99 + (float)layer_17_pw_num_fils / pw_conv_parallelism_out);
 const int layer_17_pw_num_of_tiles_w = (int)(0.99 + (float)layer_17_pw_ofm_width / pw_tile_w); 
 const int layer_17_pw_num_of_tiles_h = (int)(0.99 + (float)layer_17_pw_ofm_height / pw_tile_h); 
 const int layer_17_pw_num_of_weight_groups_for_one_pass = layer_17_pw_depth * pw_conv_parallelism_out / weights_group_items; 
 const int layer_17_pw_weights_offset = 590; 
 const int layer_17_activation = 0;
//****************************
//****************************
 const int layer_18_pw_num_fils = 192 / alpha;
 const int layer_18_pw_depth = layer_17_pw_num_fils;
 const int layer_18_pw_ifm_height = layer_17_pw_ofm_height;
 const int layer_18_pw_ifm_width = layer_17_pw_ofm_width;
 const int layer_18_pw_ofm_height = layer_18_pw_ifm_height;
 const int layer_18_pw_ofm_width = layer_18_pw_ifm_width;
 const int layer_18_pw_num_of_tiles_in_d = (int)(0.99 + (float)layer_18_pw_depth / pw_tile_d);
 const int layer_18_pw_num_of_tiles_out_d = (int)(0.99 + (float)layer_18_pw_num_fils / pw_conv_parallelism_out);
 const int layer_18_pw_num_of_tiles_w = (int)(0.99 + (float)layer_18_pw_ofm_width / pw_tile_w); 
 const int layer_18_pw_num_of_tiles_h = (int)(0.99 + (float)layer_18_pw_ofm_height / pw_tile_h); 
 const int layer_18_pw_num_of_weight_groups_for_one_pass = layer_18_pw_depth * pw_conv_parallelism_out / weights_group_items; 
 const int layer_18_pw_weights_offset = 686; 
 const int layer_18_activation = 6;
//****************************
const int layer_19_dw_num_fils = layer_18_pw_num_fils / alpha;
 const int layer_19_dw_depth = layer_19_dw_num_fils;
 const int layer_19_dw_strides = 2;
 const int layer_19_dw_ifm_height = layer_18_pw_ofm_height;
 const int layer_19_dw_ifm_width = layer_18_pw_ofm_width;
 const int layer_19_dw_ofm_height = layer_19_dw_ifm_height / layer_19_dw_strides;
 const int layer_19_dw_ofm_width = layer_19_dw_ifm_width / layer_19_dw_strides;
 const int layer_19_dw_padding_left = 0;
 const int layer_19_dw_padding_right = 1;
 const int layer_19_dw_padding_top = 0;
 const int layer_19_dw_padding_bottom = 1;
 const int layer_19_dw_filter_size = 3;
 const int layer_19_dw_num_of_tiles_in_d = (int)(0.99 + (float)layer_19_dw_depth / dw_tile_d);
 const int layer_19_dw_ifm_num_of_tiles_w = (int)(0.99 + (float)layer_19_dw_ifm_width / dw_tile_w); 
 const int layer_19_dw_ifm_num_of_tiles_h = (int)(0.99 + (float)layer_19_dw_ifm_height / dw_tile_h); 
 const int layer_19_dw_num_of_tiles_w = (int)(0.99 + (float)layer_19_dw_ofm_width / dw_tile_w); 
 const int layer_19_dw_num_of_tiles_h = (int)(0.99 + (float)layer_19_dw_ofm_height / dw_tile_h); 
 //****************************
//****************************
 const int layer_20_pw_num_fils = 64 / alpha;
 const int layer_20_pw_depth = layer_19_dw_num_fils;
 const int layer_20_pw_ifm_height = layer_19_dw_ofm_height;
 const int layer_20_pw_ifm_width = layer_19_dw_ofm_width;
 const int layer_20_pw_ofm_height = layer_20_pw_ifm_height;
 const int layer_20_pw_ofm_width = layer_20_pw_ifm_width;
 const int layer_20_pw_num_of_tiles_in_d = (int)(0.99 + (float)layer_20_pw_depth / pw_tile_d);
 const int layer_20_pw_num_of_tiles_out_d = (int)(0.99 + (float)layer_20_pw_num_fils / pw_conv_parallelism_out);
 const int layer_20_pw_num_of_tiles_w = (int)(0.99 + (float)layer_20_pw_ofm_width / pw_tile_w); 
 const int layer_20_pw_num_of_tiles_h = (int)(0.99 + (float)layer_20_pw_ofm_height / pw_tile_h); 
 const int layer_20_pw_num_of_weight_groups_for_one_pass = layer_20_pw_depth * pw_conv_parallelism_out / weights_group_items; 
 const int layer_20_pw_weights_offset = 782; 
 const int layer_20_activation = 0;
//****************************
//****************************
 const int layer_21_pw_num_fils = 384 / alpha;
 const int layer_21_pw_depth = layer_20_pw_num_fils;
 const int layer_21_pw_ifm_height = layer_20_pw_ofm_height;
 const int layer_21_pw_ifm_width = layer_20_pw_ofm_width;
 const int layer_21_pw_ofm_height = layer_21_pw_ifm_height;
 const int layer_21_pw_ofm_width = layer_21_pw_ifm_width;
 const int layer_21_pw_num_of_tiles_in_d = (int)(0.99 + (float)layer_21_pw_depth / pw_tile_d);
 const int layer_21_pw_num_of_tiles_out_d = (int)(0.99 + (float)layer_21_pw_num_fils / pw_conv_parallelism_out);
 const int layer_21_pw_num_of_tiles_w = (int)(0.99 + (float)layer_21_pw_ofm_width / pw_tile_w); 
 const int layer_21_pw_num_of_tiles_h = (int)(0.99 + (float)layer_21_pw_ofm_height / pw_tile_h); 
 const int layer_21_pw_num_of_weight_groups_for_one_pass = layer_21_pw_depth * pw_conv_parallelism_out / weights_group_items; 
 const int layer_21_pw_weights_offset = 974; 
 const int layer_21_activation = 6;
//****************************
const int layer_22_dw_num_fils = layer_21_pw_num_fils / alpha;
 const int layer_22_dw_depth = layer_22_dw_num_fils;
 const int layer_22_dw_strides = 1;
 const int layer_22_dw_ifm_height = layer_21_pw_ofm_height;
 const int layer_22_dw_ifm_width = layer_21_pw_ofm_width;
 const int layer_22_dw_ofm_height = layer_22_dw_ifm_height / layer_22_dw_strides;
 const int layer_22_dw_ofm_width = layer_22_dw_ifm_width / layer_22_dw_strides;
 const int layer_22_dw_padding_left = 1;
 const int layer_22_dw_padding_right = 1;
 const int layer_22_dw_padding_top = 1;
 const int layer_22_dw_padding_bottom = 1;
 const int layer_22_dw_filter_size = 3;
 const int layer_22_dw_num_of_tiles_in_d = (int)(0.99 + (float)layer_22_dw_depth / dw_tile_d);
 const int layer_22_dw_ifm_num_of_tiles_w = (int)(0.99 + (float)layer_22_dw_ifm_width / dw_tile_w); 
 const int layer_22_dw_ifm_num_of_tiles_h = (int)(0.99 + (float)layer_22_dw_ifm_height / dw_tile_h); 
 const int layer_22_dw_num_of_tiles_w = (int)(0.99 + (float)layer_22_dw_ofm_width / dw_tile_w); 
 const int layer_22_dw_num_of_tiles_h = (int)(0.99 + (float)layer_22_dw_ofm_height / dw_tile_h); 
 //****************************
//****************************
 const int layer_23_pw_num_fils = 64 / alpha;
 const int layer_23_pw_depth = layer_22_dw_num_fils;
 const int layer_23_pw_ifm_height = layer_22_dw_ofm_height;
 const int layer_23_pw_ifm_width = layer_22_dw_ofm_width;
 const int layer_23_pw_ofm_height = layer_23_pw_ifm_height;
 const int layer_23_pw_ofm_width = layer_23_pw_ifm_width;
 const int layer_23_pw_num_of_tiles_in_d = (int)(0.99 + (float)layer_23_pw_depth / pw_tile_d);
 const int layer_23_pw_num_of_tiles_out_d = (int)(0.99 + (float)layer_23_pw_num_fils / pw_conv_parallelism_out);
 const int layer_23_pw_num_of_tiles_w = (int)(0.99 + (float)layer_23_pw_ofm_width / pw_tile_w); 
 const int layer_23_pw_num_of_tiles_h = (int)(0.99 + (float)layer_23_pw_ofm_height / pw_tile_h); 
 const int layer_23_pw_num_of_weight_groups_for_one_pass = layer_23_pw_depth * pw_conv_parallelism_out / weights_group_items; 
 const int layer_23_pw_weights_offset = 1358; 
 const int layer_23_activation = 0;
//****************************
//****************************
 const int layer_24_pw_num_fils = 384 / alpha;
 const int layer_24_pw_depth = layer_23_pw_num_fils;
 const int layer_24_pw_ifm_height = layer_23_pw_ofm_height;
 const int layer_24_pw_ifm_width = layer_23_pw_ofm_width;
 const int layer_24_pw_ofm_height = layer_24_pw_ifm_height;
 const int layer_24_pw_ofm_width = layer_24_pw_ifm_width;
 const int layer_24_pw_num_of_tiles_in_d = (int)(0.99 + (float)layer_24_pw_depth / pw_tile_d);
 const int layer_24_pw_num_of_tiles_out_d = (int)(0.99 + (float)layer_24_pw_num_fils / pw_conv_parallelism_out);
 const int layer_24_pw_num_of_tiles_w = (int)(0.99 + (float)layer_24_pw_ofm_width / pw_tile_w); 
 const int layer_24_pw_num_of_tiles_h = (int)(0.99 + (float)layer_24_pw_ofm_height / pw_tile_h); 
 const int layer_24_pw_num_of_weight_groups_for_one_pass = layer_24_pw_depth * pw_conv_parallelism_out / weights_group_items; 
 const int layer_24_pw_weights_offset = 1742; 
 const int layer_24_activation = 6;
//****************************
const int layer_25_dw_num_fils = layer_24_pw_num_fils / alpha;
 const int layer_25_dw_depth = layer_25_dw_num_fils;
 const int layer_25_dw_strides = 1;
 const int layer_25_dw_ifm_height = layer_24_pw_ofm_height;
 const int layer_25_dw_ifm_width = layer_24_pw_ofm_width;
 const int layer_25_dw_ofm_height = layer_25_dw_ifm_height / layer_25_dw_strides;
 const int layer_25_dw_ofm_width = layer_25_dw_ifm_width / layer_25_dw_strides;
 const int layer_25_dw_padding_left = 1;
 const int layer_25_dw_padding_right = 1;
 const int layer_25_dw_padding_top = 1;
 const int layer_25_dw_padding_bottom = 1;
 const int layer_25_dw_filter_size = 3;
 const int layer_25_dw_num_of_tiles_in_d = (int)(0.99 + (float)layer_25_dw_depth / dw_tile_d);
 const int layer_25_dw_ifm_num_of_tiles_w = (int)(0.99 + (float)layer_25_dw_ifm_width / dw_tile_w); 
 const int layer_25_dw_ifm_num_of_tiles_h = (int)(0.99 + (float)layer_25_dw_ifm_height / dw_tile_h); 
 const int layer_25_dw_num_of_tiles_w = (int)(0.99 + (float)layer_25_dw_ofm_width / dw_tile_w); 
 const int layer_25_dw_num_of_tiles_h = (int)(0.99 + (float)layer_25_dw_ofm_height / dw_tile_h); 
 //****************************
//****************************
 const int layer_26_pw_num_fils = 64 / alpha;
 const int layer_26_pw_depth = layer_25_dw_num_fils;
 const int layer_26_pw_ifm_height = layer_25_dw_ofm_height;
 const int layer_26_pw_ifm_width = layer_25_dw_ofm_width;
 const int layer_26_pw_ofm_height = layer_26_pw_ifm_height;
 const int layer_26_pw_ofm_width = layer_26_pw_ifm_width;
 const int layer_26_pw_num_of_tiles_in_d = (int)(0.99 + (float)layer_26_pw_depth / pw_tile_d);
 const int layer_26_pw_num_of_tiles_out_d = (int)(0.99 + (float)layer_26_pw_num_fils / pw_conv_parallelism_out);
 const int layer_26_pw_num_of_tiles_w = (int)(0.99 + (float)layer_26_pw_ofm_width / pw_tile_w); 
 const int layer_26_pw_num_of_tiles_h = (int)(0.99 + (float)layer_26_pw_ofm_height / pw_tile_h); 
 const int layer_26_pw_num_of_weight_groups_for_one_pass = layer_26_pw_depth * pw_conv_parallelism_out / weights_group_items; 
 const int layer_26_pw_weights_offset = 2126; 
 const int layer_26_activation = 0;
//****************************
//****************************
 const int layer_27_pw_num_fils = 384 / alpha;
 const int layer_27_pw_depth = layer_26_pw_num_fils;
 const int layer_27_pw_ifm_height = layer_26_pw_ofm_height;
 const int layer_27_pw_ifm_width = layer_26_pw_ofm_width;
 const int layer_27_pw_ofm_height = layer_27_pw_ifm_height;
 const int layer_27_pw_ofm_width = layer_27_pw_ifm_width;
 const int layer_27_pw_num_of_tiles_in_d = (int)(0.99 + (float)layer_27_pw_depth / pw_tile_d);
 const int layer_27_pw_num_of_tiles_out_d = (int)(0.99 + (float)layer_27_pw_num_fils / pw_conv_parallelism_out);
 const int layer_27_pw_num_of_tiles_w = (int)(0.99 + (float)layer_27_pw_ofm_width / pw_tile_w); 
 const int layer_27_pw_num_of_tiles_h = (int)(0.99 + (float)layer_27_pw_ofm_height / pw_tile_h); 
 const int layer_27_pw_num_of_weight_groups_for_one_pass = layer_27_pw_depth * pw_conv_parallelism_out / weights_group_items; 
 const int layer_27_pw_weights_offset = 2510; 
 const int layer_27_activation = 6;
//****************************
const int layer_28_dw_num_fils = layer_27_pw_num_fils / alpha;
 const int layer_28_dw_depth = layer_28_dw_num_fils;
 const int layer_28_dw_strides = 1;
 const int layer_28_dw_ifm_height = layer_27_pw_ofm_height;
 const int layer_28_dw_ifm_width = layer_27_pw_ofm_width;
 const int layer_28_dw_ofm_height = layer_28_dw_ifm_height / layer_28_dw_strides;
 const int layer_28_dw_ofm_width = layer_28_dw_ifm_width / layer_28_dw_strides;
 const int layer_28_dw_padding_left = 1;
 const int layer_28_dw_padding_right = 1;
 const int layer_28_dw_padding_top = 1;
 const int layer_28_dw_padding_bottom = 1;
 const int layer_28_dw_filter_size = 3;
 const int layer_28_dw_num_of_tiles_in_d = (int)(0.99 + (float)layer_28_dw_depth / dw_tile_d);
 const int layer_28_dw_ifm_num_of_tiles_w = (int)(0.99 + (float)layer_28_dw_ifm_width / dw_tile_w); 
 const int layer_28_dw_ifm_num_of_tiles_h = (int)(0.99 + (float)layer_28_dw_ifm_height / dw_tile_h); 
 const int layer_28_dw_num_of_tiles_w = (int)(0.99 + (float)layer_28_dw_ofm_width / dw_tile_w); 
 const int layer_28_dw_num_of_tiles_h = (int)(0.99 + (float)layer_28_dw_ofm_height / dw_tile_h); 
 //****************************
//****************************
 const int layer_29_pw_num_fils = 64 / alpha;
 const int layer_29_pw_depth = layer_28_dw_num_fils;
 const int layer_29_pw_ifm_height = layer_28_dw_ofm_height;
 const int layer_29_pw_ifm_width = layer_28_dw_ofm_width;
 const int layer_29_pw_ofm_height = layer_29_pw_ifm_height;
 const int layer_29_pw_ofm_width = layer_29_pw_ifm_width;
 const int layer_29_pw_num_of_tiles_in_d = (int)(0.99 + (float)layer_29_pw_depth / pw_tile_d);
 const int layer_29_pw_num_of_tiles_out_d = (int)(0.99 + (float)layer_29_pw_num_fils / pw_conv_parallelism_out);
 const int layer_29_pw_num_of_tiles_w = (int)(0.99 + (float)layer_29_pw_ofm_width / pw_tile_w); 
 const int layer_29_pw_num_of_tiles_h = (int)(0.99 + (float)layer_29_pw_ofm_height / pw_tile_h); 
 const int layer_29_pw_num_of_weight_groups_for_one_pass = layer_29_pw_depth * pw_conv_parallelism_out / weights_group_items; 
 const int layer_29_pw_weights_offset = 2894; 
 const int layer_29_activation = 0;
//****************************
//****************************
 const int layer_30_pw_num_fils = 384 / alpha;
 const int layer_30_pw_depth = layer_29_pw_num_fils;
 const int layer_30_pw_ifm_height = layer_29_pw_ofm_height;
 const int layer_30_pw_ifm_width = layer_29_pw_ofm_width;
 const int layer_30_pw_ofm_height = layer_30_pw_ifm_height;
 const int layer_30_pw_ofm_width = layer_30_pw_ifm_width;
 const int layer_30_pw_num_of_tiles_in_d = (int)(0.99 + (float)layer_30_pw_depth / pw_tile_d);
 const int layer_30_pw_num_of_tiles_out_d = (int)(0.99 + (float)layer_30_pw_num_fils / pw_conv_parallelism_out);
 const int layer_30_pw_num_of_tiles_w = (int)(0.99 + (float)layer_30_pw_ofm_width / pw_tile_w); 
 const int layer_30_pw_num_of_tiles_h = (int)(0.99 + (float)layer_30_pw_ofm_height / pw_tile_h); 
 const int layer_30_pw_num_of_weight_groups_for_one_pass = layer_30_pw_depth * pw_conv_parallelism_out / weights_group_items; 
 const int layer_30_pw_weights_offset = 3278; 
 const int layer_30_activation = 6;
//****************************
const int layer_31_dw_num_fils = layer_30_pw_num_fils / alpha;
 const int layer_31_dw_depth = layer_31_dw_num_fils;
 const int layer_31_dw_strides = 1;
 const int layer_31_dw_ifm_height = layer_30_pw_ofm_height;
 const int layer_31_dw_ifm_width = layer_30_pw_ofm_width;
 const int layer_31_dw_ofm_height = layer_31_dw_ifm_height / layer_31_dw_strides;
 const int layer_31_dw_ofm_width = layer_31_dw_ifm_width / layer_31_dw_strides;
 const int layer_31_dw_padding_left = 1;
 const int layer_31_dw_padding_right = 1;
 const int layer_31_dw_padding_top = 1;
 const int layer_31_dw_padding_bottom = 1;
 const int layer_31_dw_filter_size = 3;
 const int layer_31_dw_num_of_tiles_in_d = (int)(0.99 + (float)layer_31_dw_depth / dw_tile_d);
 const int layer_31_dw_ifm_num_of_tiles_w = (int)(0.99 + (float)layer_31_dw_ifm_width / dw_tile_w); 
 const int layer_31_dw_ifm_num_of_tiles_h = (int)(0.99 + (float)layer_31_dw_ifm_height / dw_tile_h); 
 const int layer_31_dw_num_of_tiles_w = (int)(0.99 + (float)layer_31_dw_ofm_width / dw_tile_w); 
 const int layer_31_dw_num_of_tiles_h = (int)(0.99 + (float)layer_31_dw_ofm_height / dw_tile_h); 
 //****************************
//****************************
 const int layer_32_pw_num_fils = 96 / alpha;
 const int layer_32_pw_depth = layer_31_dw_num_fils;
 const int layer_32_pw_ifm_height = layer_31_dw_ofm_height;
 const int layer_32_pw_ifm_width = layer_31_dw_ofm_width;
 const int layer_32_pw_ofm_height = layer_32_pw_ifm_height;
 const int layer_32_pw_ofm_width = layer_32_pw_ifm_width;
 const int layer_32_pw_num_of_tiles_in_d = (int)(0.99 + (float)layer_32_pw_depth / pw_tile_d);
 const int layer_32_pw_num_of_tiles_out_d = (int)(0.99 + (float)layer_32_pw_num_fils / pw_conv_parallelism_out);
 const int layer_32_pw_num_of_tiles_w = (int)(0.99 + (float)layer_32_pw_ofm_width / pw_tile_w); 
 const int layer_32_pw_num_of_tiles_h = (int)(0.99 + (float)layer_32_pw_ofm_height / pw_tile_h); 
 const int layer_32_pw_num_of_weight_groups_for_one_pass = layer_32_pw_depth * pw_conv_parallelism_out / weights_group_items; 
 const int layer_32_pw_weights_offset = 3662; 
 const int layer_32_activation = 0;
//****************************
//****************************
 const int layer_33_pw_num_fils = 576 / alpha;
 const int layer_33_pw_depth = layer_32_pw_num_fils;
 const int layer_33_pw_ifm_height = layer_32_pw_ofm_height;
 const int layer_33_pw_ifm_width = layer_32_pw_ofm_width;
 const int layer_33_pw_ofm_height = layer_33_pw_ifm_height;
 const int layer_33_pw_ofm_width = layer_33_pw_ifm_width;
 const int layer_33_pw_num_of_tiles_in_d = (int)(0.99 + (float)layer_33_pw_depth / pw_tile_d);
 const int layer_33_pw_num_of_tiles_out_d = (int)(0.99 + (float)layer_33_pw_num_fils / pw_conv_parallelism_out);
 const int layer_33_pw_num_of_tiles_w = (int)(0.99 + (float)layer_33_pw_ofm_width / pw_tile_w); 
 const int layer_33_pw_num_of_tiles_h = (int)(0.99 + (float)layer_33_pw_ofm_height / pw_tile_h); 
 const int layer_33_pw_num_of_weight_groups_for_one_pass = layer_33_pw_depth * pw_conv_parallelism_out / weights_group_items; 
 const int layer_33_pw_weights_offset = 4238; 
 const int layer_33_activation = 6;
//****************************
const int layer_34_dw_num_fils = layer_33_pw_num_fils / alpha;
 const int layer_34_dw_depth = layer_34_dw_num_fils;
 const int layer_34_dw_strides = 1;
 const int layer_34_dw_ifm_height = layer_33_pw_ofm_height;
 const int layer_34_dw_ifm_width = layer_33_pw_ofm_width;
 const int layer_34_dw_ofm_height = layer_34_dw_ifm_height / layer_34_dw_strides;
 const int layer_34_dw_ofm_width = layer_34_dw_ifm_width / layer_34_dw_strides;
 const int layer_34_dw_padding_left = 1;
 const int layer_34_dw_padding_right = 1;
 const int layer_34_dw_padding_top = 1;
 const int layer_34_dw_padding_bottom = 1;
 const int layer_34_dw_filter_size = 3;
 const int layer_34_dw_num_of_tiles_in_d = (int)(0.99 + (float)layer_34_dw_depth / dw_tile_d);
 const int layer_34_dw_ifm_num_of_tiles_w = (int)(0.99 + (float)layer_34_dw_ifm_width / dw_tile_w); 
 const int layer_34_dw_ifm_num_of_tiles_h = (int)(0.99 + (float)layer_34_dw_ifm_height / dw_tile_h); 
 const int layer_34_dw_num_of_tiles_w = (int)(0.99 + (float)layer_34_dw_ofm_width / dw_tile_w); 
 const int layer_34_dw_num_of_tiles_h = (int)(0.99 + (float)layer_34_dw_ofm_height / dw_tile_h); 
 //****************************
//****************************
 const int layer_35_pw_num_fils = 96 / alpha;
 const int layer_35_pw_depth = layer_34_dw_num_fils;
 const int layer_35_pw_ifm_height = layer_34_dw_ofm_height;
 const int layer_35_pw_ifm_width = layer_34_dw_ofm_width;
 const int layer_35_pw_ofm_height = layer_35_pw_ifm_height;
 const int layer_35_pw_ofm_width = layer_35_pw_ifm_width;
 const int layer_35_pw_num_of_tiles_in_d = (int)(0.99 + (float)layer_35_pw_depth / pw_tile_d);
 const int layer_35_pw_num_of_tiles_out_d = (int)(0.99 + (float)layer_35_pw_num_fils / pw_conv_parallelism_out);
 const int layer_35_pw_num_of_tiles_w = (int)(0.99 + (float)layer_35_pw_ofm_width / pw_tile_w); 
 const int layer_35_pw_num_of_tiles_h = (int)(0.99 + (float)layer_35_pw_ofm_height / pw_tile_h); 
 const int layer_35_pw_num_of_weight_groups_for_one_pass = layer_35_pw_depth * pw_conv_parallelism_out / weights_group_items; 
 const int layer_35_pw_weights_offset = 5102; 
 const int layer_35_activation = 0;
//****************************
//****************************
 const int layer_36_pw_num_fils = 576 / alpha;
 const int layer_36_pw_depth = layer_35_pw_num_fils;
 const int layer_36_pw_ifm_height = layer_35_pw_ofm_height;
 const int layer_36_pw_ifm_width = layer_35_pw_ofm_width;
 const int layer_36_pw_ofm_height = layer_36_pw_ifm_height;
 const int layer_36_pw_ofm_width = layer_36_pw_ifm_width;
 const int layer_36_pw_num_of_tiles_in_d = (int)(0.99 + (float)layer_36_pw_depth / pw_tile_d);
 const int layer_36_pw_num_of_tiles_out_d = (int)(0.99 + (float)layer_36_pw_num_fils / pw_conv_parallelism_out);
 const int layer_36_pw_num_of_tiles_w = (int)(0.99 + (float)layer_36_pw_ofm_width / pw_tile_w); 
 const int layer_36_pw_num_of_tiles_h = (int)(0.99 + (float)layer_36_pw_ofm_height / pw_tile_h); 
 const int layer_36_pw_num_of_weight_groups_for_one_pass = layer_36_pw_depth * pw_conv_parallelism_out / weights_group_items; 
 const int layer_36_pw_weights_offset = 5966; 
 const int layer_36_activation = 6;
//****************************
const int layer_37_dw_num_fils = layer_36_pw_num_fils / alpha;
 const int layer_37_dw_depth = layer_37_dw_num_fils;
 const int layer_37_dw_strides = 1;
 const int layer_37_dw_ifm_height = layer_36_pw_ofm_height;
 const int layer_37_dw_ifm_width = layer_36_pw_ofm_width;
 const int layer_37_dw_ofm_height = layer_37_dw_ifm_height / layer_37_dw_strides;
 const int layer_37_dw_ofm_width = layer_37_dw_ifm_width / layer_37_dw_strides;
 const int layer_37_dw_padding_left = 1;
 const int layer_37_dw_padding_right = 1;
 const int layer_37_dw_padding_top = 1;
 const int layer_37_dw_padding_bottom = 1;
 const int layer_37_dw_filter_size = 3;
 const int layer_37_dw_num_of_tiles_in_d = (int)(0.99 + (float)layer_37_dw_depth / dw_tile_d);
 const int layer_37_dw_ifm_num_of_tiles_w = (int)(0.99 + (float)layer_37_dw_ifm_width / dw_tile_w); 
 const int layer_37_dw_ifm_num_of_tiles_h = (int)(0.99 + (float)layer_37_dw_ifm_height / dw_tile_h); 
 const int layer_37_dw_num_of_tiles_w = (int)(0.99 + (float)layer_37_dw_ofm_width / dw_tile_w); 
 const int layer_37_dw_num_of_tiles_h = (int)(0.99 + (float)layer_37_dw_ofm_height / dw_tile_h); 
 //****************************
//****************************
 const int layer_38_pw_num_fils = 96 / alpha;
 const int layer_38_pw_depth = layer_37_dw_num_fils;
 const int layer_38_pw_ifm_height = layer_37_dw_ofm_height;
 const int layer_38_pw_ifm_width = layer_37_dw_ofm_width;
 const int layer_38_pw_ofm_height = layer_38_pw_ifm_height;
 const int layer_38_pw_ofm_width = layer_38_pw_ifm_width;
 const int layer_38_pw_num_of_tiles_in_d = (int)(0.99 + (float)layer_38_pw_depth / pw_tile_d);
 const int layer_38_pw_num_of_tiles_out_d = (int)(0.99 + (float)layer_38_pw_num_fils / pw_conv_parallelism_out);
 const int layer_38_pw_num_of_tiles_w = (int)(0.99 + (float)layer_38_pw_ofm_width / pw_tile_w); 
 const int layer_38_pw_num_of_tiles_h = (int)(0.99 + (float)layer_38_pw_ofm_height / pw_tile_h); 
 const int layer_38_pw_num_of_weight_groups_for_one_pass = layer_38_pw_depth * pw_conv_parallelism_out / weights_group_items; 
 const int layer_38_pw_weights_offset = 6830; 
 const int layer_38_activation = 0;
//****************************
//****************************
 const int layer_39_pw_num_fils = 576 / alpha;
 const int layer_39_pw_depth = layer_38_pw_num_fils;
 const int layer_39_pw_ifm_height = layer_38_pw_ofm_height;
 const int layer_39_pw_ifm_width = layer_38_pw_ofm_width;
 const int layer_39_pw_ofm_height = layer_39_pw_ifm_height;
 const int layer_39_pw_ofm_width = layer_39_pw_ifm_width;
 const int layer_39_pw_num_of_tiles_in_d = (int)(0.99 + (float)layer_39_pw_depth / pw_tile_d);
 const int layer_39_pw_num_of_tiles_out_d = (int)(0.99 + (float)layer_39_pw_num_fils / pw_conv_parallelism_out);
 const int layer_39_pw_num_of_tiles_w = (int)(0.99 + (float)layer_39_pw_ofm_width / pw_tile_w); 
 const int layer_39_pw_num_of_tiles_h = (int)(0.99 + (float)layer_39_pw_ofm_height / pw_tile_h); 
 const int layer_39_pw_num_of_weight_groups_for_one_pass = layer_39_pw_depth * pw_conv_parallelism_out / weights_group_items; 
 const int layer_39_pw_weights_offset = 7694; 
 const int layer_39_activation = 6;
//****************************
const int layer_40_dw_num_fils = layer_39_pw_num_fils / alpha;
 const int layer_40_dw_depth = layer_40_dw_num_fils;
 const int layer_40_dw_strides = 2;
 const int layer_40_dw_ifm_height = layer_39_pw_ofm_height;
 const int layer_40_dw_ifm_width = layer_39_pw_ofm_width;
 const int layer_40_dw_ofm_height = layer_40_dw_ifm_height / layer_40_dw_strides;
 const int layer_40_dw_ofm_width = layer_40_dw_ifm_width / layer_40_dw_strides;
 const int layer_40_dw_padding_left = 0;
 const int layer_40_dw_padding_right = 1;
 const int layer_40_dw_padding_top = 0;
 const int layer_40_dw_padding_bottom = 1;
 const int layer_40_dw_filter_size = 3;
 const int layer_40_dw_num_of_tiles_in_d = (int)(0.99 + (float)layer_40_dw_depth / dw_tile_d);
 const int layer_40_dw_ifm_num_of_tiles_w = (int)(0.99 + (float)layer_40_dw_ifm_width / dw_tile_w); 
 const int layer_40_dw_ifm_num_of_tiles_h = (int)(0.99 + (float)layer_40_dw_ifm_height / dw_tile_h); 
 const int layer_40_dw_num_of_tiles_w = (int)(0.99 + (float)layer_40_dw_ofm_width / dw_tile_w); 
 const int layer_40_dw_num_of_tiles_h = (int)(0.99 + (float)layer_40_dw_ofm_height / dw_tile_h); 
 //****************************
//****************************
 const int layer_41_pw_num_fils = 160 / alpha;
 const int layer_41_pw_depth = layer_40_dw_num_fils;
 const int layer_41_pw_ifm_height = layer_40_dw_ofm_height;
 const int layer_41_pw_ifm_width = layer_40_dw_ofm_width;
 const int layer_41_pw_ofm_height = layer_41_pw_ifm_height;
 const int layer_41_pw_ofm_width = layer_41_pw_ifm_width;
 const int layer_41_pw_num_of_tiles_in_d = (int)(0.99 + (float)layer_41_pw_depth / pw_tile_d);
 const int layer_41_pw_num_of_tiles_out_d = (int)(0.99 + (float)layer_41_pw_num_fils / pw_conv_parallelism_out);
 const int layer_41_pw_num_of_tiles_w = (int)(0.99 + (float)layer_41_pw_ofm_width / pw_tile_w); 
 const int layer_41_pw_num_of_tiles_h = (int)(0.99 + (float)layer_41_pw_ofm_height / pw_tile_h); 
 const int layer_41_pw_num_of_weight_groups_for_one_pass = layer_41_pw_depth * pw_conv_parallelism_out / weights_group_items; 
 const int layer_41_pw_weights_offset = 8558; 
 const int layer_41_activation = 0;
//****************************
//****************************
 const int layer_42_pw_num_fils = 960 / alpha;
 const int layer_42_pw_depth = layer_41_pw_num_fils;
 const int layer_42_pw_ifm_height = layer_41_pw_ofm_height;
 const int layer_42_pw_ifm_width = layer_41_pw_ofm_width;
 const int layer_42_pw_ofm_height = layer_42_pw_ifm_height;
 const int layer_42_pw_ofm_width = layer_42_pw_ifm_width;
 const int layer_42_pw_num_of_tiles_in_d = (int)(0.99 + (float)layer_42_pw_depth / pw_tile_d);
 const int layer_42_pw_num_of_tiles_out_d = (int)(0.99 + (float)layer_42_pw_num_fils / pw_conv_parallelism_out);
 const int layer_42_pw_num_of_tiles_w = (int)(0.99 + (float)layer_42_pw_ofm_width / pw_tile_w); 
 const int layer_42_pw_num_of_tiles_h = (int)(0.99 + (float)layer_42_pw_ofm_height / pw_tile_h); 
 const int layer_42_pw_num_of_weight_groups_for_one_pass = layer_42_pw_depth * pw_conv_parallelism_out / weights_group_items; 
 const int layer_42_pw_weights_offset = 9998; 
 const int layer_42_activation = 6;
//****************************
const int layer_43_dw_num_fils = layer_42_pw_num_fils / alpha;
 const int layer_43_dw_depth = layer_43_dw_num_fils;
 const int layer_43_dw_strides = 1;
 const int layer_43_dw_ifm_height = layer_42_pw_ofm_height;
 const int layer_43_dw_ifm_width = layer_42_pw_ofm_width;
 const int layer_43_dw_ofm_height = layer_43_dw_ifm_height / layer_43_dw_strides;
 const int layer_43_dw_ofm_width = layer_43_dw_ifm_width / layer_43_dw_strides;
 const int layer_43_dw_padding_left = 1;
 const int layer_43_dw_padding_right = 1;
 const int layer_43_dw_padding_top = 1;
 const int layer_43_dw_padding_bottom = 1;
 const int layer_43_dw_filter_size = 3;
 const int layer_43_dw_num_of_tiles_in_d = (int)(0.99 + (float)layer_43_dw_depth / dw_tile_d);
 const int layer_43_dw_ifm_num_of_tiles_w = (int)(0.99 + (float)layer_43_dw_ifm_width / dw_tile_w); 
 const int layer_43_dw_ifm_num_of_tiles_h = (int)(0.99 + (float)layer_43_dw_ifm_height / dw_tile_h); 
 const int layer_43_dw_num_of_tiles_w = (int)(0.99 + (float)layer_43_dw_ofm_width / dw_tile_w); 
 const int layer_43_dw_num_of_tiles_h = (int)(0.99 + (float)layer_43_dw_ofm_height / dw_tile_h); 
 //****************************
//****************************
 const int layer_44_pw_num_fils = 160 / alpha;
 const int layer_44_pw_depth = layer_43_dw_num_fils;
 const int layer_44_pw_ifm_height = layer_43_dw_ofm_height;
 const int layer_44_pw_ifm_width = layer_43_dw_ofm_width;
 const int layer_44_pw_ofm_height = layer_44_pw_ifm_height;
 const int layer_44_pw_ofm_width = layer_44_pw_ifm_width;
 const int layer_44_pw_num_of_tiles_in_d = (int)(0.99 + (float)layer_44_pw_depth / pw_tile_d);
 const int layer_44_pw_num_of_tiles_out_d = (int)(0.99 + (float)layer_44_pw_num_fils / pw_conv_parallelism_out);
 const int layer_44_pw_num_of_tiles_w = (int)(0.99 + (float)layer_44_pw_ofm_width / pw_tile_w); 
 const int layer_44_pw_num_of_tiles_h = (int)(0.99 + (float)layer_44_pw_ofm_height / pw_tile_h); 
 const int layer_44_pw_num_of_weight_groups_for_one_pass = layer_44_pw_depth * pw_conv_parallelism_out / weights_group_items; 
 const int layer_44_pw_weights_offset = 12398; 
 const int layer_44_activation = 0;
//****************************
//****************************
 const int layer_45_pw_num_fils = 960 / alpha;
 const int layer_45_pw_depth = layer_44_pw_num_fils;
 const int layer_45_pw_ifm_height = layer_44_pw_ofm_height;
 const int layer_45_pw_ifm_width = layer_44_pw_ofm_width;
 const int layer_45_pw_ofm_height = layer_45_pw_ifm_height;
 const int layer_45_pw_ofm_width = layer_45_pw_ifm_width;
 const int layer_45_pw_num_of_tiles_in_d = (int)(0.99 + (float)layer_45_pw_depth / pw_tile_d);
 const int layer_45_pw_num_of_tiles_out_d = (int)(0.99 + (float)layer_45_pw_num_fils / pw_conv_parallelism_out);
 const int layer_45_pw_num_of_tiles_w = (int)(0.99 + (float)layer_45_pw_ofm_width / pw_tile_w); 
 const int layer_45_pw_num_of_tiles_h = (int)(0.99 + (float)layer_45_pw_ofm_height / pw_tile_h); 
 const int layer_45_pw_num_of_weight_groups_for_one_pass = layer_45_pw_depth * pw_conv_parallelism_out / weights_group_items; 
 const int layer_45_pw_weights_offset = 14798; 
 const int layer_45_activation = 6;
//****************************
const int layer_46_dw_num_fils = layer_45_pw_num_fils / alpha;
 const int layer_46_dw_depth = layer_46_dw_num_fils;
 const int layer_46_dw_strides = 1;
 const int layer_46_dw_ifm_height = layer_45_pw_ofm_height;
 const int layer_46_dw_ifm_width = layer_45_pw_ofm_width;
 const int layer_46_dw_ofm_height = layer_46_dw_ifm_height / layer_46_dw_strides;
 const int layer_46_dw_ofm_width = layer_46_dw_ifm_width / layer_46_dw_strides;
 const int layer_46_dw_padding_left = 1;
 const int layer_46_dw_padding_right = 1;
 const int layer_46_dw_padding_top = 1;
 const int layer_46_dw_padding_bottom = 1;
 const int layer_46_dw_filter_size = 3;
 const int layer_46_dw_num_of_tiles_in_d = (int)(0.99 + (float)layer_46_dw_depth / dw_tile_d);
 const int layer_46_dw_ifm_num_of_tiles_w = (int)(0.99 + (float)layer_46_dw_ifm_width / dw_tile_w); 
 const int layer_46_dw_ifm_num_of_tiles_h = (int)(0.99 + (float)layer_46_dw_ifm_height / dw_tile_h); 
 const int layer_46_dw_num_of_tiles_w = (int)(0.99 + (float)layer_46_dw_ofm_width / dw_tile_w); 
 const int layer_46_dw_num_of_tiles_h = (int)(0.99 + (float)layer_46_dw_ofm_height / dw_tile_h); 
 //****************************
//****************************
 const int layer_47_pw_num_fils = 160 / alpha;
 const int layer_47_pw_depth = layer_46_dw_num_fils;
 const int layer_47_pw_ifm_height = layer_46_dw_ofm_height;
 const int layer_47_pw_ifm_width = layer_46_dw_ofm_width;
 const int layer_47_pw_ofm_height = layer_47_pw_ifm_height;
 const int layer_47_pw_ofm_width = layer_47_pw_ifm_width;
 const int layer_47_pw_num_of_tiles_in_d = (int)(0.99 + (float)layer_47_pw_depth / pw_tile_d);
 const int layer_47_pw_num_of_tiles_out_d = (int)(0.99 + (float)layer_47_pw_num_fils / pw_conv_parallelism_out);
 const int layer_47_pw_num_of_tiles_w = (int)(0.99 + (float)layer_47_pw_ofm_width / pw_tile_w); 
 const int layer_47_pw_num_of_tiles_h = (int)(0.99 + (float)layer_47_pw_ofm_height / pw_tile_h); 
 const int layer_47_pw_num_of_weight_groups_for_one_pass = layer_47_pw_depth * pw_conv_parallelism_out / weights_group_items; 
 const int layer_47_pw_weights_offset = 17198; 
 const int layer_47_activation = 0;
//****************************
//****************************
 const int layer_48_pw_num_fils = 960 / alpha;
 const int layer_48_pw_depth = layer_47_pw_num_fils;
 const int layer_48_pw_ifm_height = layer_47_pw_ofm_height;
 const int layer_48_pw_ifm_width = layer_47_pw_ofm_width;
 const int layer_48_pw_ofm_height = layer_48_pw_ifm_height;
 const int layer_48_pw_ofm_width = layer_48_pw_ifm_width;
 const int layer_48_pw_num_of_tiles_in_d = (int)(0.99 + (float)layer_48_pw_depth / pw_tile_d);
 const int layer_48_pw_num_of_tiles_out_d = (int)(0.99 + (float)layer_48_pw_num_fils / pw_conv_parallelism_out);
 const int layer_48_pw_num_of_tiles_w = (int)(0.99 + (float)layer_48_pw_ofm_width / pw_tile_w); 
 const int layer_48_pw_num_of_tiles_h = (int)(0.99 + (float)layer_48_pw_ofm_height / pw_tile_h); 
 const int layer_48_pw_num_of_weight_groups_for_one_pass = layer_48_pw_depth * pw_conv_parallelism_out / weights_group_items; 
 const int layer_48_pw_weights_offset = 19598; 
 const int layer_48_activation = 6;
//****************************
const int layer_49_dw_num_fils = layer_48_pw_num_fils / alpha;
 const int layer_49_dw_depth = layer_49_dw_num_fils;
 const int layer_49_dw_strides = 1;
 const int layer_49_dw_ifm_height = layer_48_pw_ofm_height;
 const int layer_49_dw_ifm_width = layer_48_pw_ofm_width;
 const int layer_49_dw_ofm_height = layer_49_dw_ifm_height / layer_49_dw_strides;
 const int layer_49_dw_ofm_width = layer_49_dw_ifm_width / layer_49_dw_strides;
 const int layer_49_dw_padding_left = 1;
 const int layer_49_dw_padding_right = 1;
 const int layer_49_dw_padding_top = 1;
 const int layer_49_dw_padding_bottom = 1;
 const int layer_49_dw_filter_size = 3;
 const int layer_49_dw_num_of_tiles_in_d = (int)(0.99 + (float)layer_49_dw_depth / dw_tile_d);
 const int layer_49_dw_ifm_num_of_tiles_w = (int)(0.99 + (float)layer_49_dw_ifm_width / dw_tile_w); 
 const int layer_49_dw_ifm_num_of_tiles_h = (int)(0.99 + (float)layer_49_dw_ifm_height / dw_tile_h); 
 const int layer_49_dw_num_of_tiles_w = (int)(0.99 + (float)layer_49_dw_ofm_width / dw_tile_w); 
 const int layer_49_dw_num_of_tiles_h = (int)(0.99 + (float)layer_49_dw_ofm_height / dw_tile_h); 
 //****************************
//****************************
 const int layer_50_pw_num_fils = 320 / alpha;
 const int layer_50_pw_depth = layer_49_dw_num_fils;
 const int layer_50_pw_ifm_height = layer_49_dw_ofm_height;
 const int layer_50_pw_ifm_width = layer_49_dw_ofm_width;
 const int layer_50_pw_ofm_height = layer_50_pw_ifm_height;
 const int layer_50_pw_ofm_width = layer_50_pw_ifm_width;
 const int layer_50_pw_num_of_tiles_in_d = (int)(0.99 + (float)layer_50_pw_depth / pw_tile_d);
 const int layer_50_pw_num_of_tiles_out_d = (int)(0.99 + (float)layer_50_pw_num_fils / pw_conv_parallelism_out);
 const int layer_50_pw_num_of_tiles_w = (int)(0.99 + (float)layer_50_pw_ofm_width / pw_tile_w); 
 const int layer_50_pw_num_of_tiles_h = (int)(0.99 + (float)layer_50_pw_ofm_height / pw_tile_h); 
 const int layer_50_pw_num_of_weight_groups_for_one_pass = layer_50_pw_depth * pw_conv_parallelism_out / weights_group_items; 
 const int layer_50_pw_weights_offset = 21998; 
 const int layer_50_activation = 0;
//****************************
//****************************
 const int layer_51_pw_num_fils = 1280 / alpha;
 const int layer_51_pw_depth = layer_50_pw_num_fils;
 const int layer_51_pw_ifm_height = layer_50_pw_ofm_height;
 const int layer_51_pw_ifm_width = layer_50_pw_ofm_width;
 const int layer_51_pw_ofm_height = layer_51_pw_ifm_height;
 const int layer_51_pw_ofm_width = layer_51_pw_ifm_width;
 const int layer_51_pw_num_of_tiles_in_d = (int)(0.99 + (float)layer_51_pw_depth / pw_tile_d);
 const int layer_51_pw_num_of_tiles_out_d = (int)(0.99 + (float)layer_51_pw_num_fils / pw_conv_parallelism_out);
 const int layer_51_pw_num_of_tiles_w = (int)(0.99 + (float)layer_51_pw_ofm_width / pw_tile_w); 
 const int layer_51_pw_num_of_tiles_h = (int)(0.99 + (float)layer_51_pw_ofm_height / pw_tile_h); 
 const int layer_51_pw_num_of_weight_groups_for_one_pass = layer_51_pw_depth * pw_conv_parallelism_out / weights_group_items; 
 const int layer_51_pw_weights_offset = 26798; 
 const int layer_51_activation = 6;
//****************************
#endif
