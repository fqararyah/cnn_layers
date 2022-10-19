#include "../basic_defs/basic_defs_glue.h"
#ifndef LAYERS_SPECS
#define LAYERS_SPECS
//****************************
 const int layer_0_num_fils = 32 / alpha;
const int layer_0_depth = input_image_depth;
const int layer_0_ifm_height = input_image_height;
const int layer_0_ifm_width = input_image_width;
const int layer_0_strides = 2;
const int layer_0_ofm_height = layer_0_ifm_height / layer_0_strides;
const int layer_0_ofm_width = layer_0_ifm_width / layer_0_strides;
const int layer_0_num_of_tiles_out_d = int(0.99 + ((float) layer_0_num_fils) / pw_conv_parallelism_out);
const int layer_0_padding_left = 0;
const int layer_0_padding_right = 1;
const int layer_0_filter_size = 3;
 const int layer_0_num_of_tiles_w = layer_0_ofm_width / pw_tile_w; 
 const int layer_0_num_of_tiles_h = layer_0_ofm_height / pw_tile_h; 
 const int layer_0_num_of_tiles_d_in = layer_0_depth / pw_tile_d; 
 //****************************
//****************************
 const int layer_1_pw_num_fils = 32 / alpha;
 const int layer_1_pw_depth = layer_0_num_fils;
 const int layer_1_pw_ifm_height = layer_0_ofm_height;
 const int layer_1_pw_ifm_width = layer_0_ofm_width;
 const int layer_1_pw_ofm_height = layer_1_pw_ifm_height;
 const int layer_1_pw_ofm_width = layer_1_pw_ifm_width;
 const int layer_1_pw_num_of_tiles_in_d = (int)(0.99 + (float)layer_1_pw_depth / pw_tile_d);
 const int layer_1_pw_num_of_tiles_out_d = (int)(0.99 + (float)layer_1_pw_num_fils / pw_conv_parallelism_out);
 const int layer_1_pw_num_of_tiles_w = (int)(0.99 + (float)layer_1_pw_ofm_width / pw_tile_w); 
 const int layer_1_pw_num_of_tiles_h = (int)(0.99 + (float)layer_1_pw_ofm_height / pw_tile_h); 
 const int layer_1_pw_num_of_weight_groups_for_one_pass = layer_1_pw_depth * pw_conv_parallelism_out / weights_group_items; 
 const int layer_1_pw_weights_offset = 0; 
 const int layer_1_relu = 6;
//****************************
const int layer_2_dw_num_fils = layer_1_pw_num_fils / alpha;
 const int layer_2_dw_depth = layer_2_dw_num_fils;
 const int layer_2_dw_strides = 1;
 const int layer_2_dw_ifm_height = layer_1_pw_ofm_height;
 const int layer_2_dw_ifm_width = layer_1_pw_ofm_width;
 const int layer_2_dw_ofm_height = layer_2_dw_ifm_height / layer_2_dw_strides;
 const int layer_2_dw_ofm_width = layer_2_dw_ifm_width / layer_2_dw_strides;
 const int layer_2_dw_padding_left = 1.0;
 const int layer_2_dw_padding_right = 1.0;
 const int layer_2_dw_filter_size = 3;
 const int layer_2_dw_num_of_tiles_in_d = (int)(((float)layer_2_dw_depth / dw_tile_d) + 0.5);
 const int layer_2_dw_num_of_tiles_w = layer_2_dw_ofm_width / dw_tile_w; 
 const int layer_2_dw_num_of_tiles_h = layer_2_dw_ofm_height / dw_tile_h; 
 //****************************
//****************************
 const int layer_3_pw_num_fils = 16 / alpha;
 const int layer_3_pw_depth = layer_2_dw_depth;
 const int layer_3_pw_ifm_height = layer_2_dw_ofm_height;
 const int layer_3_pw_ifm_width = layer_2_dw_ofm_width;
 const int layer_3_pw_ofm_height = layer_3_pw_ifm_height;
 const int layer_3_pw_ofm_width = layer_3_pw_ifm_width;
 const int layer_3_pw_num_of_tiles_in_d = (int)(0.99 + (float)layer_3_pw_depth / pw_tile_d);
 const int layer_3_pw_num_of_tiles_out_d = (int)(0.99 + (float)layer_3_pw_num_fils / pw_conv_parallelism_out);
 const int layer_3_pw_num_of_tiles_w = (int)(0.99 + (float)layer_3_pw_ofm_width / pw_tile_w); 
 const int layer_3_pw_num_of_tiles_h = (int)(0.99 + (float)layer_3_pw_ofm_height / pw_tile_h); 
 const int layer_3_pw_num_of_weight_groups_for_one_pass = layer_3_pw_depth * pw_conv_parallelism_out / weights_group_items; 
 const int layer_3_pw_weights_offset = 0; 
 const int layer_3_relu = 0;
//****************************
//****************************
 const int layer_4_pw_num_fils = 96 / alpha;
 const int layer_4_pw_depth = layer_3_pw_num_fils;
 const int layer_4_pw_ifm_height = layer_3_pw_ofm_height;
 const int layer_4_pw_ifm_width = layer_3_pw_ofm_width;
 const int layer_4_pw_ofm_height = layer_4_pw_ifm_height;
 const int layer_4_pw_ofm_width = layer_4_pw_ifm_width;
 const int layer_4_pw_num_of_tiles_in_d = (int)(0.99 + (float)layer_4_pw_depth / pw_tile_d);
 const int layer_4_pw_num_of_tiles_out_d = (int)(0.99 + (float)layer_4_pw_num_fils / pw_conv_parallelism_out);
 const int layer_4_pw_num_of_tiles_w = (int)(0.99 + (float)layer_4_pw_ofm_width / pw_tile_w); 
 const int layer_4_pw_num_of_tiles_h = (int)(0.99 + (float)layer_4_pw_ofm_height / pw_tile_h); 
 const int layer_4_pw_num_of_weight_groups_for_one_pass = layer_4_pw_depth * pw_conv_parallelism_out / weights_group_items; 
 const int layer_4_pw_weights_offset = 512; 
 const int layer_4_relu = 6;
//****************************
const int layer_5_dw_num_fils = layer_4_pw_num_fils / alpha;
 const int layer_5_dw_depth = layer_5_dw_num_fils;
 const int layer_5_dw_strides = 2;
 const int layer_5_dw_ifm_height = layer_4_pw_ofm_height;
 const int layer_5_dw_ifm_width = layer_4_pw_ofm_width;
 const int layer_5_dw_ofm_height = layer_5_dw_ifm_height / layer_5_dw_strides;
 const int layer_5_dw_ofm_width = layer_5_dw_ifm_width / layer_5_dw_strides;
 const int layer_5_dw_padding_left = 0;
 const int layer_5_dw_padding_right = 1;
 const int layer_5_dw_filter_size = 3;
 const int layer_5_dw_num_of_tiles_in_d = (int)(((float)layer_5_dw_depth / dw_tile_d) + 0.5);
 const int layer_5_dw_num_of_tiles_w = layer_5_dw_ofm_width / dw_tile_w; 
 const int layer_5_dw_num_of_tiles_h = layer_5_dw_ofm_height / dw_tile_h; 
 //****************************
//****************************
 const int layer_6_pw_num_fils = 24 / alpha;
 const int layer_6_pw_depth = layer_5_dw_depth;
 const int layer_6_pw_ifm_height = layer_5_dw_ofm_height;
 const int layer_6_pw_ifm_width = layer_5_dw_ofm_width;
 const int layer_6_pw_ofm_height = layer_6_pw_ifm_height;
 const int layer_6_pw_ofm_width = layer_6_pw_ifm_width;
 const int layer_6_pw_num_of_tiles_in_d = (int)(0.99 + (float)layer_6_pw_depth / pw_tile_d);
 const int layer_6_pw_num_of_tiles_out_d = (int)(0.99 + (float)layer_6_pw_num_fils / pw_conv_parallelism_out);
 const int layer_6_pw_num_of_tiles_w = (int)(0.99 + (float)layer_6_pw_ofm_width / pw_tile_w); 
 const int layer_6_pw_num_of_tiles_h = (int)(0.99 + (float)layer_6_pw_ofm_height / pw_tile_h); 
 const int layer_6_pw_num_of_weight_groups_for_one_pass = layer_6_pw_depth * pw_conv_parallelism_out / weights_group_items; 
 const int layer_6_pw_weights_offset = 2048; 
 const int layer_6_relu = 0;
//****************************
//****************************
 const int layer_7_pw_num_fils = 144 / alpha;
 const int layer_7_pw_depth = layer_6_pw_num_fils;
 const int layer_7_pw_ifm_height = layer_6_pw_ofm_height;
 const int layer_7_pw_ifm_width = layer_6_pw_ofm_width;
 const int layer_7_pw_ofm_height = layer_7_pw_ifm_height;
 const int layer_7_pw_ofm_width = layer_7_pw_ifm_width;
 const int layer_7_pw_num_of_tiles_in_d = (int)(0.99 + (float)layer_7_pw_depth / pw_tile_d);
 const int layer_7_pw_num_of_tiles_out_d = (int)(0.99 + (float)layer_7_pw_num_fils / pw_conv_parallelism_out);
 const int layer_7_pw_num_of_tiles_w = (int)(0.99 + (float)layer_7_pw_ofm_width / pw_tile_w); 
 const int layer_7_pw_num_of_tiles_h = (int)(0.99 + (float)layer_7_pw_ofm_height / pw_tile_h); 
 const int layer_7_pw_num_of_weight_groups_for_one_pass = layer_7_pw_depth * pw_conv_parallelism_out / weights_group_items; 
 const int layer_7_pw_weights_offset = 4352; 
 const int layer_7_relu = 6;
//****************************
const int layer_8_dw_num_fils = layer_7_pw_num_fils / alpha;
 const int layer_8_dw_depth = layer_8_dw_num_fils;
 const int layer_8_dw_strides = 1;
 const int layer_8_dw_ifm_height = layer_7_pw_ofm_height;
 const int layer_8_dw_ifm_width = layer_7_pw_ofm_width;
 const int layer_8_dw_ofm_height = layer_8_dw_ifm_height / layer_8_dw_strides;
 const int layer_8_dw_ofm_width = layer_8_dw_ifm_width / layer_8_dw_strides;
 const int layer_8_dw_padding_left = 1.0;
 const int layer_8_dw_padding_right = 1.0;
 const int layer_8_dw_filter_size = 3;
 const int layer_8_dw_num_of_tiles_in_d = (int)(((float)layer_8_dw_depth / dw_tile_d) + 0.5);
 const int layer_8_dw_num_of_tiles_w = layer_8_dw_ofm_width / dw_tile_w; 
 const int layer_8_dw_num_of_tiles_h = layer_8_dw_ofm_height / dw_tile_h; 
 //****************************
//****************************
 const int layer_9_pw_num_fils = 24 / alpha;
 const int layer_9_pw_depth = layer_8_dw_depth;
 const int layer_9_pw_ifm_height = layer_8_dw_ofm_height;
 const int layer_9_pw_ifm_width = layer_8_dw_ofm_width;
 const int layer_9_pw_ofm_height = layer_9_pw_ifm_height;
 const int layer_9_pw_ofm_width = layer_9_pw_ifm_width;
 const int layer_9_pw_num_of_tiles_in_d = (int)(0.99 + (float)layer_9_pw_depth / pw_tile_d);
 const int layer_9_pw_num_of_tiles_out_d = (int)(0.99 + (float)layer_9_pw_num_fils / pw_conv_parallelism_out);
 const int layer_9_pw_num_of_tiles_w = (int)(0.99 + (float)layer_9_pw_ofm_width / pw_tile_w); 
 const int layer_9_pw_num_of_tiles_h = (int)(0.99 + (float)layer_9_pw_ofm_height / pw_tile_h); 
 const int layer_9_pw_num_of_weight_groups_for_one_pass = layer_9_pw_depth * pw_conv_parallelism_out / weights_group_items; 
 const int layer_9_pw_weights_offset = 7808; 
 const int layer_9_relu = 0;
//****************************
//****************************
 const int layer_10_pw_num_fils = 144 / alpha;
 const int layer_10_pw_depth = layer_9_pw_num_fils;
 const int layer_10_pw_ifm_height = layer_9_pw_ofm_height;
 const int layer_10_pw_ifm_width = layer_9_pw_ofm_width;
 const int layer_10_pw_ofm_height = layer_10_pw_ifm_height;
 const int layer_10_pw_ofm_width = layer_10_pw_ifm_width;
 const int layer_10_pw_num_of_tiles_in_d = (int)(0.99 + (float)layer_10_pw_depth / pw_tile_d);
 const int layer_10_pw_num_of_tiles_out_d = (int)(0.99 + (float)layer_10_pw_num_fils / pw_conv_parallelism_out);
 const int layer_10_pw_num_of_tiles_w = (int)(0.99 + (float)layer_10_pw_ofm_width / pw_tile_w); 
 const int layer_10_pw_num_of_tiles_h = (int)(0.99 + (float)layer_10_pw_ofm_height / pw_tile_h); 
 const int layer_10_pw_num_of_weight_groups_for_one_pass = layer_10_pw_depth * pw_conv_parallelism_out / weights_group_items; 
 const int layer_10_pw_weights_offset = 11264; 
 const int layer_10_relu = 6;
//****************************
const int layer_11_dw_num_fils = layer_10_pw_num_fils / alpha;
 const int layer_11_dw_depth = layer_11_dw_num_fils;
 const int layer_11_dw_strides = 2;
 const int layer_11_dw_ifm_height = layer_10_pw_ofm_height;
 const int layer_11_dw_ifm_width = layer_10_pw_ofm_width;
 const int layer_11_dw_ofm_height = layer_11_dw_ifm_height / layer_11_dw_strides;
 const int layer_11_dw_ofm_width = layer_11_dw_ifm_width / layer_11_dw_strides;
 const int layer_11_dw_padding_left = 0;
 const int layer_11_dw_padding_right = 1;
 const int layer_11_dw_filter_size = 3;
 const int layer_11_dw_num_of_tiles_in_d = (int)(((float)layer_11_dw_depth / dw_tile_d) + 0.5);
 const int layer_11_dw_num_of_tiles_w = layer_11_dw_ofm_width / dw_tile_w; 
 const int layer_11_dw_num_of_tiles_h = layer_11_dw_ofm_height / dw_tile_h; 
 //****************************
//****************************
 const int layer_12_pw_num_fils = 32 / alpha;
 const int layer_12_pw_depth = layer_11_dw_depth;
 const int layer_12_pw_ifm_height = layer_11_dw_ofm_height;
 const int layer_12_pw_ifm_width = layer_11_dw_ofm_width;
 const int layer_12_pw_ofm_height = layer_12_pw_ifm_height;
 const int layer_12_pw_ofm_width = layer_12_pw_ifm_width;
 const int layer_12_pw_num_of_tiles_in_d = (int)(0.99 + (float)layer_12_pw_depth / pw_tile_d);
 const int layer_12_pw_num_of_tiles_out_d = (int)(0.99 + (float)layer_12_pw_num_fils / pw_conv_parallelism_out);
 const int layer_12_pw_num_of_tiles_w = (int)(0.99 + (float)layer_12_pw_ofm_width / pw_tile_w); 
 const int layer_12_pw_num_of_tiles_h = (int)(0.99 + (float)layer_12_pw_ofm_height / pw_tile_h); 
 const int layer_12_pw_num_of_weight_groups_for_one_pass = layer_12_pw_depth * pw_conv_parallelism_out / weights_group_items; 
 const int layer_12_pw_weights_offset = 14720; 
 const int layer_12_relu = 0;
//****************************
//****************************
 const int layer_13_pw_num_fils = 192 / alpha;
 const int layer_13_pw_depth = layer_12_pw_num_fils;
 const int layer_13_pw_ifm_height = layer_12_pw_ofm_height;
 const int layer_13_pw_ifm_width = layer_12_pw_ofm_width;
 const int layer_13_pw_ofm_height = layer_13_pw_ifm_height;
 const int layer_13_pw_ofm_width = layer_13_pw_ifm_width;
 const int layer_13_pw_num_of_tiles_in_d = (int)(0.99 + (float)layer_13_pw_depth / pw_tile_d);
 const int layer_13_pw_num_of_tiles_out_d = (int)(0.99 + (float)layer_13_pw_num_fils / pw_conv_parallelism_out);
 const int layer_13_pw_num_of_tiles_w = (int)(0.99 + (float)layer_13_pw_ofm_width / pw_tile_w); 
 const int layer_13_pw_num_of_tiles_h = (int)(0.99 + (float)layer_13_pw_ofm_height / pw_tile_h); 
 const int layer_13_pw_num_of_weight_groups_for_one_pass = layer_13_pw_depth * pw_conv_parallelism_out / weights_group_items; 
 const int layer_13_pw_weights_offset = 19328; 
 const int layer_13_relu = 6;
//****************************
const int layer_14_dw_num_fils = layer_13_pw_num_fils / alpha;
 const int layer_14_dw_depth = layer_14_dw_num_fils;
 const int layer_14_dw_strides = 1;
 const int layer_14_dw_ifm_height = layer_13_pw_ofm_height;
 const int layer_14_dw_ifm_width = layer_13_pw_ofm_width;
 const int layer_14_dw_ofm_height = layer_14_dw_ifm_height / layer_14_dw_strides;
 const int layer_14_dw_ofm_width = layer_14_dw_ifm_width / layer_14_dw_strides;
 const int layer_14_dw_padding_left = 1.0;
 const int layer_14_dw_padding_right = 1.0;
 const int layer_14_dw_filter_size = 3;
 const int layer_14_dw_num_of_tiles_in_d = (int)(((float)layer_14_dw_depth / dw_tile_d) + 0.5);
 const int layer_14_dw_num_of_tiles_w = layer_14_dw_ofm_width / dw_tile_w; 
 const int layer_14_dw_num_of_tiles_h = layer_14_dw_ofm_height / dw_tile_h; 
 //****************************
//****************************
 const int layer_15_pw_num_fils = 32 / alpha;
 const int layer_15_pw_depth = layer_14_dw_depth;
 const int layer_15_pw_ifm_height = layer_14_dw_ofm_height;
 const int layer_15_pw_ifm_width = layer_14_dw_ofm_width;
 const int layer_15_pw_ofm_height = layer_15_pw_ifm_height;
 const int layer_15_pw_ofm_width = layer_15_pw_ifm_width;
 const int layer_15_pw_num_of_tiles_in_d = (int)(0.99 + (float)layer_15_pw_depth / pw_tile_d);
 const int layer_15_pw_num_of_tiles_out_d = (int)(0.99 + (float)layer_15_pw_num_fils / pw_conv_parallelism_out);
 const int layer_15_pw_num_of_tiles_w = (int)(0.99 + (float)layer_15_pw_ofm_width / pw_tile_w); 
 const int layer_15_pw_num_of_tiles_h = (int)(0.99 + (float)layer_15_pw_ofm_height / pw_tile_h); 
 const int layer_15_pw_num_of_weight_groups_for_one_pass = layer_15_pw_depth * pw_conv_parallelism_out / weights_group_items; 
 const int layer_15_pw_weights_offset = 25472; 
 const int layer_15_relu = 0;
//****************************
//****************************
 const int layer_16_pw_num_fils = 192 / alpha;
 const int layer_16_pw_depth = layer_15_pw_num_fils;
 const int layer_16_pw_ifm_height = layer_15_pw_ofm_height;
 const int layer_16_pw_ifm_width = layer_15_pw_ofm_width;
 const int layer_16_pw_ofm_height = layer_16_pw_ifm_height;
 const int layer_16_pw_ofm_width = layer_16_pw_ifm_width;
 const int layer_16_pw_num_of_tiles_in_d = (int)(0.99 + (float)layer_16_pw_depth / pw_tile_d);
 const int layer_16_pw_num_of_tiles_out_d = (int)(0.99 + (float)layer_16_pw_num_fils / pw_conv_parallelism_out);
 const int layer_16_pw_num_of_tiles_w = (int)(0.99 + (float)layer_16_pw_ofm_width / pw_tile_w); 
 const int layer_16_pw_num_of_tiles_h = (int)(0.99 + (float)layer_16_pw_ofm_height / pw_tile_h); 
 const int layer_16_pw_num_of_weight_groups_for_one_pass = layer_16_pw_depth * pw_conv_parallelism_out / weights_group_items; 
 const int layer_16_pw_weights_offset = 31616; 
 const int layer_16_relu = 6;
//****************************
const int layer_17_dw_num_fils = layer_16_pw_num_fils / alpha;
 const int layer_17_dw_depth = layer_17_dw_num_fils;
 const int layer_17_dw_strides = 1;
 const int layer_17_dw_ifm_height = layer_16_pw_ofm_height;
 const int layer_17_dw_ifm_width = layer_16_pw_ofm_width;
 const int layer_17_dw_ofm_height = layer_17_dw_ifm_height / layer_17_dw_strides;
 const int layer_17_dw_ofm_width = layer_17_dw_ifm_width / layer_17_dw_strides;
 const int layer_17_dw_padding_left = 1.0;
 const int layer_17_dw_padding_right = 1.0;
 const int layer_17_dw_filter_size = 3;
 const int layer_17_dw_num_of_tiles_in_d = (int)(((float)layer_17_dw_depth / dw_tile_d) + 0.5);
 const int layer_17_dw_num_of_tiles_w = layer_17_dw_ofm_width / dw_tile_w; 
 const int layer_17_dw_num_of_tiles_h = layer_17_dw_ofm_height / dw_tile_h; 
 //****************************
//****************************
 const int layer_18_pw_num_fils = 32 / alpha;
 const int layer_18_pw_depth = layer_17_dw_depth;
 const int layer_18_pw_ifm_height = layer_17_dw_ofm_height;
 const int layer_18_pw_ifm_width = layer_17_dw_ofm_width;
 const int layer_18_pw_ofm_height = layer_18_pw_ifm_height;
 const int layer_18_pw_ofm_width = layer_18_pw_ifm_width;
 const int layer_18_pw_num_of_tiles_in_d = (int)(0.99 + (float)layer_18_pw_depth / pw_tile_d);
 const int layer_18_pw_num_of_tiles_out_d = (int)(0.99 + (float)layer_18_pw_num_fils / pw_conv_parallelism_out);
 const int layer_18_pw_num_of_tiles_w = (int)(0.99 + (float)layer_18_pw_ofm_width / pw_tile_w); 
 const int layer_18_pw_num_of_tiles_h = (int)(0.99 + (float)layer_18_pw_ofm_height / pw_tile_h); 
 const int layer_18_pw_num_of_weight_groups_for_one_pass = layer_18_pw_depth * pw_conv_parallelism_out / weights_group_items; 
 const int layer_18_pw_weights_offset = 37760; 
 const int layer_18_relu = 0;
//****************************
//****************************
 const int layer_19_pw_num_fils = 192 / alpha;
 const int layer_19_pw_depth = layer_18_pw_num_fils;
 const int layer_19_pw_ifm_height = layer_18_pw_ofm_height;
 const int layer_19_pw_ifm_width = layer_18_pw_ofm_width;
 const int layer_19_pw_ofm_height = layer_19_pw_ifm_height;
 const int layer_19_pw_ofm_width = layer_19_pw_ifm_width;
 const int layer_19_pw_num_of_tiles_in_d = (int)(0.99 + (float)layer_19_pw_depth / pw_tile_d);
 const int layer_19_pw_num_of_tiles_out_d = (int)(0.99 + (float)layer_19_pw_num_fils / pw_conv_parallelism_out);
 const int layer_19_pw_num_of_tiles_w = (int)(0.99 + (float)layer_19_pw_ofm_width / pw_tile_w); 
 const int layer_19_pw_num_of_tiles_h = (int)(0.99 + (float)layer_19_pw_ofm_height / pw_tile_h); 
 const int layer_19_pw_num_of_weight_groups_for_one_pass = layer_19_pw_depth * pw_conv_parallelism_out / weights_group_items; 
 const int layer_19_pw_weights_offset = 43904; 
 const int layer_19_relu = 6;
//****************************
const int layer_20_dw_num_fils = layer_19_pw_num_fils / alpha;
 const int layer_20_dw_depth = layer_20_dw_num_fils;
 const int layer_20_dw_strides = 2;
 const int layer_20_dw_ifm_height = layer_19_pw_ofm_height;
 const int layer_20_dw_ifm_width = layer_19_pw_ofm_width;
 const int layer_20_dw_ofm_height = layer_20_dw_ifm_height / layer_20_dw_strides;
 const int layer_20_dw_ofm_width = layer_20_dw_ifm_width / layer_20_dw_strides;
 const int layer_20_dw_padding_left = 0;
 const int layer_20_dw_padding_right = 1;
 const int layer_20_dw_filter_size = 3;
 const int layer_20_dw_num_of_tiles_in_d = (int)(((float)layer_20_dw_depth / dw_tile_d) + 0.5);
 const int layer_20_dw_num_of_tiles_w = layer_20_dw_ofm_width / dw_tile_w; 
 const int layer_20_dw_num_of_tiles_h = layer_20_dw_ofm_height / dw_tile_h; 
 //****************************
//****************************
 const int layer_21_pw_num_fils = 64 / alpha;
 const int layer_21_pw_depth = layer_20_dw_depth;
 const int layer_21_pw_ifm_height = layer_20_dw_ofm_height;
 const int layer_21_pw_ifm_width = layer_20_dw_ofm_width;
 const int layer_21_pw_ofm_height = layer_21_pw_ifm_height;
 const int layer_21_pw_ofm_width = layer_21_pw_ifm_width;
 const int layer_21_pw_num_of_tiles_in_d = (int)(0.99 + (float)layer_21_pw_depth / pw_tile_d);
 const int layer_21_pw_num_of_tiles_out_d = (int)(0.99 + (float)layer_21_pw_num_fils / pw_conv_parallelism_out);
 const int layer_21_pw_num_of_tiles_w = (int)(0.99 + (float)layer_21_pw_ofm_width / pw_tile_w); 
 const int layer_21_pw_num_of_tiles_h = (int)(0.99 + (float)layer_21_pw_ofm_height / pw_tile_h); 
 const int layer_21_pw_num_of_weight_groups_for_one_pass = layer_21_pw_depth * pw_conv_parallelism_out / weights_group_items; 
 const int layer_21_pw_weights_offset = 50048; 
 const int layer_21_relu = 0;
//****************************
//****************************
 const int layer_22_pw_num_fils = 384 / alpha;
 const int layer_22_pw_depth = layer_21_pw_num_fils;
 const int layer_22_pw_ifm_height = layer_21_pw_ofm_height;
 const int layer_22_pw_ifm_width = layer_21_pw_ofm_width;
 const int layer_22_pw_ofm_height = layer_22_pw_ifm_height;
 const int layer_22_pw_ofm_width = layer_22_pw_ifm_width;
 const int layer_22_pw_num_of_tiles_in_d = (int)(0.99 + (float)layer_22_pw_depth / pw_tile_d);
 const int layer_22_pw_num_of_tiles_out_d = (int)(0.99 + (float)layer_22_pw_num_fils / pw_conv_parallelism_out);
 const int layer_22_pw_num_of_tiles_w = (int)(0.99 + (float)layer_22_pw_ofm_width / pw_tile_w); 
 const int layer_22_pw_num_of_tiles_h = (int)(0.99 + (float)layer_22_pw_ofm_height / pw_tile_h); 
 const int layer_22_pw_num_of_weight_groups_for_one_pass = layer_22_pw_depth * pw_conv_parallelism_out / weights_group_items; 
 const int layer_22_pw_weights_offset = 62336; 
 const int layer_22_relu = 6;
//****************************
const int layer_23_dw_num_fils = layer_22_pw_num_fils / alpha;
 const int layer_23_dw_depth = layer_23_dw_num_fils;
 const int layer_23_dw_strides = 1;
 const int layer_23_dw_ifm_height = layer_22_pw_ofm_height;
 const int layer_23_dw_ifm_width = layer_22_pw_ofm_width;
 const int layer_23_dw_ofm_height = layer_23_dw_ifm_height / layer_23_dw_strides;
 const int layer_23_dw_ofm_width = layer_23_dw_ifm_width / layer_23_dw_strides;
 const int layer_23_dw_padding_left = 1.0;
 const int layer_23_dw_padding_right = 1.0;
 const int layer_23_dw_filter_size = 3;
 const int layer_23_dw_num_of_tiles_in_d = (int)(((float)layer_23_dw_depth / dw_tile_d) + 0.5);
 const int layer_23_dw_num_of_tiles_w = layer_23_dw_ofm_width / dw_tile_w; 
 const int layer_23_dw_num_of_tiles_h = layer_23_dw_ofm_height / dw_tile_h; 
 //****************************
//****************************
 const int layer_24_pw_num_fils = 64 / alpha;
 const int layer_24_pw_depth = layer_23_dw_depth;
 const int layer_24_pw_ifm_height = layer_23_dw_ofm_height;
 const int layer_24_pw_ifm_width = layer_23_dw_ofm_width;
 const int layer_24_pw_ofm_height = layer_24_pw_ifm_height;
 const int layer_24_pw_ofm_width = layer_24_pw_ifm_width;
 const int layer_24_pw_num_of_tiles_in_d = (int)(0.99 + (float)layer_24_pw_depth / pw_tile_d);
 const int layer_24_pw_num_of_tiles_out_d = (int)(0.99 + (float)layer_24_pw_num_fils / pw_conv_parallelism_out);
 const int layer_24_pw_num_of_tiles_w = (int)(0.99 + (float)layer_24_pw_ofm_width / pw_tile_w); 
 const int layer_24_pw_num_of_tiles_h = (int)(0.99 + (float)layer_24_pw_ofm_height / pw_tile_h); 
 const int layer_24_pw_num_of_weight_groups_for_one_pass = layer_24_pw_depth * pw_conv_parallelism_out / weights_group_items; 
 const int layer_24_pw_weights_offset = 86912; 
 const int layer_24_relu = 0;
//****************************
//****************************
 const int layer_25_pw_num_fils = 384 / alpha;
 const int layer_25_pw_depth = layer_24_pw_num_fils;
 const int layer_25_pw_ifm_height = layer_24_pw_ofm_height;
 const int layer_25_pw_ifm_width = layer_24_pw_ofm_width;
 const int layer_25_pw_ofm_height = layer_25_pw_ifm_height;
 const int layer_25_pw_ofm_width = layer_25_pw_ifm_width;
 const int layer_25_pw_num_of_tiles_in_d = (int)(0.99 + (float)layer_25_pw_depth / pw_tile_d);
 const int layer_25_pw_num_of_tiles_out_d = (int)(0.99 + (float)layer_25_pw_num_fils / pw_conv_parallelism_out);
 const int layer_25_pw_num_of_tiles_w = (int)(0.99 + (float)layer_25_pw_ofm_width / pw_tile_w); 
 const int layer_25_pw_num_of_tiles_h = (int)(0.99 + (float)layer_25_pw_ofm_height / pw_tile_h); 
 const int layer_25_pw_num_of_weight_groups_for_one_pass = layer_25_pw_depth * pw_conv_parallelism_out / weights_group_items; 
 const int layer_25_pw_weights_offset = 111488; 
 const int layer_25_relu = 6;
//****************************
const int layer_26_dw_num_fils = layer_25_pw_num_fils / alpha;
 const int layer_26_dw_depth = layer_26_dw_num_fils;
 const int layer_26_dw_strides = 1;
 const int layer_26_dw_ifm_height = layer_25_pw_ofm_height;
 const int layer_26_dw_ifm_width = layer_25_pw_ofm_width;
 const int layer_26_dw_ofm_height = layer_26_dw_ifm_height / layer_26_dw_strides;
 const int layer_26_dw_ofm_width = layer_26_dw_ifm_width / layer_26_dw_strides;
 const int layer_26_dw_padding_left = 1.0;
 const int layer_26_dw_padding_right = 1.0;
 const int layer_26_dw_filter_size = 3;
 const int layer_26_dw_num_of_tiles_in_d = (int)(((float)layer_26_dw_depth / dw_tile_d) + 0.5);
 const int layer_26_dw_num_of_tiles_w = layer_26_dw_ofm_width / dw_tile_w; 
 const int layer_26_dw_num_of_tiles_h = layer_26_dw_ofm_height / dw_tile_h; 
 //****************************
//****************************
 const int layer_27_pw_num_fils = 64 / alpha;
 const int layer_27_pw_depth = layer_26_dw_depth;
 const int layer_27_pw_ifm_height = layer_26_dw_ofm_height;
 const int layer_27_pw_ifm_width = layer_26_dw_ofm_width;
 const int layer_27_pw_ofm_height = layer_27_pw_ifm_height;
 const int layer_27_pw_ofm_width = layer_27_pw_ifm_width;
 const int layer_27_pw_num_of_tiles_in_d = (int)(0.99 + (float)layer_27_pw_depth / pw_tile_d);
 const int layer_27_pw_num_of_tiles_out_d = (int)(0.99 + (float)layer_27_pw_num_fils / pw_conv_parallelism_out);
 const int layer_27_pw_num_of_tiles_w = (int)(0.99 + (float)layer_27_pw_ofm_width / pw_tile_w); 
 const int layer_27_pw_num_of_tiles_h = (int)(0.99 + (float)layer_27_pw_ofm_height / pw_tile_h); 
 const int layer_27_pw_num_of_weight_groups_for_one_pass = layer_27_pw_depth * pw_conv_parallelism_out / weights_group_items; 
 const int layer_27_pw_weights_offset = 136064; 
 const int layer_27_relu = 0;
//****************************
//****************************
 const int layer_28_pw_num_fils = 384 / alpha;
 const int layer_28_pw_depth = layer_27_pw_num_fils;
 const int layer_28_pw_ifm_height = layer_27_pw_ofm_height;
 const int layer_28_pw_ifm_width = layer_27_pw_ofm_width;
 const int layer_28_pw_ofm_height = layer_28_pw_ifm_height;
 const int layer_28_pw_ofm_width = layer_28_pw_ifm_width;
 const int layer_28_pw_num_of_tiles_in_d = (int)(0.99 + (float)layer_28_pw_depth / pw_tile_d);
 const int layer_28_pw_num_of_tiles_out_d = (int)(0.99 + (float)layer_28_pw_num_fils / pw_conv_parallelism_out);
 const int layer_28_pw_num_of_tiles_w = (int)(0.99 + (float)layer_28_pw_ofm_width / pw_tile_w); 
 const int layer_28_pw_num_of_tiles_h = (int)(0.99 + (float)layer_28_pw_ofm_height / pw_tile_h); 
 const int layer_28_pw_num_of_weight_groups_for_one_pass = layer_28_pw_depth * pw_conv_parallelism_out / weights_group_items; 
 const int layer_28_pw_weights_offset = 160640; 
 const int layer_28_relu = 6;
//****************************
const int layer_29_dw_num_fils = layer_28_pw_num_fils / alpha;
 const int layer_29_dw_depth = layer_29_dw_num_fils;
 const int layer_29_dw_strides = 1;
 const int layer_29_dw_ifm_height = layer_28_pw_ofm_height;
 const int layer_29_dw_ifm_width = layer_28_pw_ofm_width;
 const int layer_29_dw_ofm_height = layer_29_dw_ifm_height / layer_29_dw_strides;
 const int layer_29_dw_ofm_width = layer_29_dw_ifm_width / layer_29_dw_strides;
 const int layer_29_dw_padding_left = 1.0;
 const int layer_29_dw_padding_right = 1.0;
 const int layer_29_dw_filter_size = 3;
 const int layer_29_dw_num_of_tiles_in_d = (int)(((float)layer_29_dw_depth / dw_tile_d) + 0.5);
 const int layer_29_dw_num_of_tiles_w = layer_29_dw_ofm_width / dw_tile_w; 
 const int layer_29_dw_num_of_tiles_h = layer_29_dw_ofm_height / dw_tile_h; 
 //****************************
//****************************
 const int layer_30_pw_num_fils = 64 / alpha;
 const int layer_30_pw_depth = layer_29_dw_depth;
 const int layer_30_pw_ifm_height = layer_29_dw_ofm_height;
 const int layer_30_pw_ifm_width = layer_29_dw_ofm_width;
 const int layer_30_pw_ofm_height = layer_30_pw_ifm_height;
 const int layer_30_pw_ofm_width = layer_30_pw_ifm_width;
 const int layer_30_pw_num_of_tiles_in_d = (int)(0.99 + (float)layer_30_pw_depth / pw_tile_d);
 const int layer_30_pw_num_of_tiles_out_d = (int)(0.99 + (float)layer_30_pw_num_fils / pw_conv_parallelism_out);
 const int layer_30_pw_num_of_tiles_w = (int)(0.99 + (float)layer_30_pw_ofm_width / pw_tile_w); 
 const int layer_30_pw_num_of_tiles_h = (int)(0.99 + (float)layer_30_pw_ofm_height / pw_tile_h); 
 const int layer_30_pw_num_of_weight_groups_for_one_pass = layer_30_pw_depth * pw_conv_parallelism_out / weights_group_items; 
 const int layer_30_pw_weights_offset = 185216; 
 const int layer_30_relu = 0;
//****************************
//****************************
 const int layer_31_pw_num_fils = 384 / alpha;
 const int layer_31_pw_depth = layer_30_pw_num_fils;
 const int layer_31_pw_ifm_height = layer_30_pw_ofm_height;
 const int layer_31_pw_ifm_width = layer_30_pw_ofm_width;
 const int layer_31_pw_ofm_height = layer_31_pw_ifm_height;
 const int layer_31_pw_ofm_width = layer_31_pw_ifm_width;
 const int layer_31_pw_num_of_tiles_in_d = (int)(0.99 + (float)layer_31_pw_depth / pw_tile_d);
 const int layer_31_pw_num_of_tiles_out_d = (int)(0.99 + (float)layer_31_pw_num_fils / pw_conv_parallelism_out);
 const int layer_31_pw_num_of_tiles_w = (int)(0.99 + (float)layer_31_pw_ofm_width / pw_tile_w); 
 const int layer_31_pw_num_of_tiles_h = (int)(0.99 + (float)layer_31_pw_ofm_height / pw_tile_h); 
 const int layer_31_pw_num_of_weight_groups_for_one_pass = layer_31_pw_depth * pw_conv_parallelism_out / weights_group_items; 
 const int layer_31_pw_weights_offset = 209792; 
 const int layer_31_relu = 6;
//****************************
const int layer_32_dw_num_fils = layer_31_pw_num_fils / alpha;
 const int layer_32_dw_depth = layer_32_dw_num_fils;
 const int layer_32_dw_strides = 1;
 const int layer_32_dw_ifm_height = layer_31_pw_ofm_height;
 const int layer_32_dw_ifm_width = layer_31_pw_ofm_width;
 const int layer_32_dw_ofm_height = layer_32_dw_ifm_height / layer_32_dw_strides;
 const int layer_32_dw_ofm_width = layer_32_dw_ifm_width / layer_32_dw_strides;
 const int layer_32_dw_padding_left = 1.0;
 const int layer_32_dw_padding_right = 1.0;
 const int layer_32_dw_filter_size = 3;
 const int layer_32_dw_num_of_tiles_in_d = (int)(((float)layer_32_dw_depth / dw_tile_d) + 0.5);
 const int layer_32_dw_num_of_tiles_w = layer_32_dw_ofm_width / dw_tile_w; 
 const int layer_32_dw_num_of_tiles_h = layer_32_dw_ofm_height / dw_tile_h; 
 //****************************
//****************************
 const int layer_33_pw_num_fils = 96 / alpha;
 const int layer_33_pw_depth = layer_32_dw_depth;
 const int layer_33_pw_ifm_height = layer_32_dw_ofm_height;
 const int layer_33_pw_ifm_width = layer_32_dw_ofm_width;
 const int layer_33_pw_ofm_height = layer_33_pw_ifm_height;
 const int layer_33_pw_ofm_width = layer_33_pw_ifm_width;
 const int layer_33_pw_num_of_tiles_in_d = (int)(0.99 + (float)layer_33_pw_depth / pw_tile_d);
 const int layer_33_pw_num_of_tiles_out_d = (int)(0.99 + (float)layer_33_pw_num_fils / pw_conv_parallelism_out);
 const int layer_33_pw_num_of_tiles_w = (int)(0.99 + (float)layer_33_pw_ofm_width / pw_tile_w); 
 const int layer_33_pw_num_of_tiles_h = (int)(0.99 + (float)layer_33_pw_ofm_height / pw_tile_h); 
 const int layer_33_pw_num_of_weight_groups_for_one_pass = layer_33_pw_depth * pw_conv_parallelism_out / weights_group_items; 
 const int layer_33_pw_weights_offset = 234368; 
 const int layer_33_relu = 0;
//****************************
//****************************
 const int layer_34_pw_num_fils = 576 / alpha;
 const int layer_34_pw_depth = layer_33_pw_num_fils;
 const int layer_34_pw_ifm_height = layer_33_pw_ofm_height;
 const int layer_34_pw_ifm_width = layer_33_pw_ofm_width;
 const int layer_34_pw_ofm_height = layer_34_pw_ifm_height;
 const int layer_34_pw_ofm_width = layer_34_pw_ifm_width;
 const int layer_34_pw_num_of_tiles_in_d = (int)(0.99 + (float)layer_34_pw_depth / pw_tile_d);
 const int layer_34_pw_num_of_tiles_out_d = (int)(0.99 + (float)layer_34_pw_num_fils / pw_conv_parallelism_out);
 const int layer_34_pw_num_of_tiles_w = (int)(0.99 + (float)layer_34_pw_ofm_width / pw_tile_w); 
 const int layer_34_pw_num_of_tiles_h = (int)(0.99 + (float)layer_34_pw_ofm_height / pw_tile_h); 
 const int layer_34_pw_num_of_weight_groups_for_one_pass = layer_34_pw_depth * pw_conv_parallelism_out / weights_group_items; 
 const int layer_34_pw_weights_offset = 271232; 
 const int layer_34_relu = 6;
//****************************
const int layer_35_dw_num_fils = layer_34_pw_num_fils / alpha;
 const int layer_35_dw_depth = layer_35_dw_num_fils;
 const int layer_35_dw_strides = 1;
 const int layer_35_dw_ifm_height = layer_34_pw_ofm_height;
 const int layer_35_dw_ifm_width = layer_34_pw_ofm_width;
 const int layer_35_dw_ofm_height = layer_35_dw_ifm_height / layer_35_dw_strides;
 const int layer_35_dw_ofm_width = layer_35_dw_ifm_width / layer_35_dw_strides;
 const int layer_35_dw_padding_left = 1.0;
 const int layer_35_dw_padding_right = 1.0;
 const int layer_35_dw_filter_size = 3;
 const int layer_35_dw_num_of_tiles_in_d = (int)(((float)layer_35_dw_depth / dw_tile_d) + 0.5);
 const int layer_35_dw_num_of_tiles_w = layer_35_dw_ofm_width / dw_tile_w; 
 const int layer_35_dw_num_of_tiles_h = layer_35_dw_ofm_height / dw_tile_h; 
 //****************************
//****************************
 const int layer_36_pw_num_fils = 96 / alpha;
 const int layer_36_pw_depth = layer_35_dw_depth;
 const int layer_36_pw_ifm_height = layer_35_dw_ofm_height;
 const int layer_36_pw_ifm_width = layer_35_dw_ofm_width;
 const int layer_36_pw_ofm_height = layer_36_pw_ifm_height;
 const int layer_36_pw_ofm_width = layer_36_pw_ifm_width;
 const int layer_36_pw_num_of_tiles_in_d = (int)(0.99 + (float)layer_36_pw_depth / pw_tile_d);
 const int layer_36_pw_num_of_tiles_out_d = (int)(0.99 + (float)layer_36_pw_num_fils / pw_conv_parallelism_out);
 const int layer_36_pw_num_of_tiles_w = (int)(0.99 + (float)layer_36_pw_ofm_width / pw_tile_w); 
 const int layer_36_pw_num_of_tiles_h = (int)(0.99 + (float)layer_36_pw_ofm_height / pw_tile_h); 
 const int layer_36_pw_num_of_weight_groups_for_one_pass = layer_36_pw_depth * pw_conv_parallelism_out / weights_group_items; 
 const int layer_36_pw_weights_offset = 326528; 
 const int layer_36_relu = 0;
//****************************
//****************************
 const int layer_37_pw_num_fils = 576 / alpha;
 const int layer_37_pw_depth = layer_36_pw_num_fils;
 const int layer_37_pw_ifm_height = layer_36_pw_ofm_height;
 const int layer_37_pw_ifm_width = layer_36_pw_ofm_width;
 const int layer_37_pw_ofm_height = layer_37_pw_ifm_height;
 const int layer_37_pw_ofm_width = layer_37_pw_ifm_width;
 const int layer_37_pw_num_of_tiles_in_d = (int)(0.99 + (float)layer_37_pw_depth / pw_tile_d);
 const int layer_37_pw_num_of_tiles_out_d = (int)(0.99 + (float)layer_37_pw_num_fils / pw_conv_parallelism_out);
 const int layer_37_pw_num_of_tiles_w = (int)(0.99 + (float)layer_37_pw_ofm_width / pw_tile_w); 
 const int layer_37_pw_num_of_tiles_h = (int)(0.99 + (float)layer_37_pw_ofm_height / pw_tile_h); 
 const int layer_37_pw_num_of_weight_groups_for_one_pass = layer_37_pw_depth * pw_conv_parallelism_out / weights_group_items; 
 const int layer_37_pw_weights_offset = 381824; 
 const int layer_37_relu = 6;
//****************************
const int layer_38_dw_num_fils = layer_37_pw_num_fils / alpha;
 const int layer_38_dw_depth = layer_38_dw_num_fils;
 const int layer_38_dw_strides = 1;
 const int layer_38_dw_ifm_height = layer_37_pw_ofm_height;
 const int layer_38_dw_ifm_width = layer_37_pw_ofm_width;
 const int layer_38_dw_ofm_height = layer_38_dw_ifm_height / layer_38_dw_strides;
 const int layer_38_dw_ofm_width = layer_38_dw_ifm_width / layer_38_dw_strides;
 const int layer_38_dw_padding_left = 1.0;
 const int layer_38_dw_padding_right = 1.0;
 const int layer_38_dw_filter_size = 3;
 const int layer_38_dw_num_of_tiles_in_d = (int)(((float)layer_38_dw_depth / dw_tile_d) + 0.5);
 const int layer_38_dw_num_of_tiles_w = layer_38_dw_ofm_width / dw_tile_w; 
 const int layer_38_dw_num_of_tiles_h = layer_38_dw_ofm_height / dw_tile_h; 
 //****************************
//****************************
 const int layer_39_pw_num_fils = 96 / alpha;
 const int layer_39_pw_depth = layer_38_dw_depth;
 const int layer_39_pw_ifm_height = layer_38_dw_ofm_height;
 const int layer_39_pw_ifm_width = layer_38_dw_ofm_width;
 const int layer_39_pw_ofm_height = layer_39_pw_ifm_height;
 const int layer_39_pw_ofm_width = layer_39_pw_ifm_width;
 const int layer_39_pw_num_of_tiles_in_d = (int)(0.99 + (float)layer_39_pw_depth / pw_tile_d);
 const int layer_39_pw_num_of_tiles_out_d = (int)(0.99 + (float)layer_39_pw_num_fils / pw_conv_parallelism_out);
 const int layer_39_pw_num_of_tiles_w = (int)(0.99 + (float)layer_39_pw_ofm_width / pw_tile_w); 
 const int layer_39_pw_num_of_tiles_h = (int)(0.99 + (float)layer_39_pw_ofm_height / pw_tile_h); 
 const int layer_39_pw_num_of_weight_groups_for_one_pass = layer_39_pw_depth * pw_conv_parallelism_out / weights_group_items; 
 const int layer_39_pw_weights_offset = 437120; 
 const int layer_39_relu = 0;
//****************************
//****************************
 const int layer_40_pw_num_fils = 576 / alpha;
 const int layer_40_pw_depth = layer_39_pw_num_fils;
 const int layer_40_pw_ifm_height = layer_39_pw_ofm_height;
 const int layer_40_pw_ifm_width = layer_39_pw_ofm_width;
 const int layer_40_pw_ofm_height = layer_40_pw_ifm_height;
 const int layer_40_pw_ofm_width = layer_40_pw_ifm_width;
 const int layer_40_pw_num_of_tiles_in_d = (int)(0.99 + (float)layer_40_pw_depth / pw_tile_d);
 const int layer_40_pw_num_of_tiles_out_d = (int)(0.99 + (float)layer_40_pw_num_fils / pw_conv_parallelism_out);
 const int layer_40_pw_num_of_tiles_w = (int)(0.99 + (float)layer_40_pw_ofm_width / pw_tile_w); 
 const int layer_40_pw_num_of_tiles_h = (int)(0.99 + (float)layer_40_pw_ofm_height / pw_tile_h); 
 const int layer_40_pw_num_of_weight_groups_for_one_pass = layer_40_pw_depth * pw_conv_parallelism_out / weights_group_items; 
 const int layer_40_pw_weights_offset = 492416; 
 const int layer_40_relu = 6;
//****************************
const int layer_41_dw_num_fils = layer_40_pw_num_fils / alpha;
 const int layer_41_dw_depth = layer_41_dw_num_fils;
 const int layer_41_dw_strides = 2;
 const int layer_41_dw_ifm_height = layer_40_pw_ofm_height;
 const int layer_41_dw_ifm_width = layer_40_pw_ofm_width;
 const int layer_41_dw_ofm_height = layer_41_dw_ifm_height / layer_41_dw_strides;
 const int layer_41_dw_ofm_width = layer_41_dw_ifm_width / layer_41_dw_strides;
 const int layer_41_dw_padding_left = 0;
 const int layer_41_dw_padding_right = 1;
 const int layer_41_dw_filter_size = 3;
 const int layer_41_dw_num_of_tiles_in_d = (int)(((float)layer_41_dw_depth / dw_tile_d) + 0.5);
 const int layer_41_dw_num_of_tiles_w = layer_41_dw_ofm_width / dw_tile_w; 
 const int layer_41_dw_num_of_tiles_h = layer_41_dw_ofm_height / dw_tile_h; 
 //****************************
//****************************
 const int layer_42_pw_num_fils = 160 / alpha;
 const int layer_42_pw_depth = layer_41_dw_depth;
 const int layer_42_pw_ifm_height = layer_41_dw_ofm_height;
 const int layer_42_pw_ifm_width = layer_41_dw_ofm_width;
 const int layer_42_pw_ofm_height = layer_42_pw_ifm_height;
 const int layer_42_pw_ofm_width = layer_42_pw_ifm_width;
 const int layer_42_pw_num_of_tiles_in_d = (int)(0.99 + (float)layer_42_pw_depth / pw_tile_d);
 const int layer_42_pw_num_of_tiles_out_d = (int)(0.99 + (float)layer_42_pw_num_fils / pw_conv_parallelism_out);
 const int layer_42_pw_num_of_tiles_w = (int)(0.99 + (float)layer_42_pw_ofm_width / pw_tile_w); 
 const int layer_42_pw_num_of_tiles_h = (int)(0.99 + (float)layer_42_pw_ofm_height / pw_tile_h); 
 const int layer_42_pw_num_of_weight_groups_for_one_pass = layer_42_pw_depth * pw_conv_parallelism_out / weights_group_items; 
 const int layer_42_pw_weights_offset = 547712; 
 const int layer_42_relu = 0;
//****************************
//****************************
 const int layer_43_pw_num_fils = 960 / alpha;
 const int layer_43_pw_depth = layer_42_pw_num_fils;
 const int layer_43_pw_ifm_height = layer_42_pw_ofm_height;
 const int layer_43_pw_ifm_width = layer_42_pw_ofm_width;
 const int layer_43_pw_ofm_height = layer_43_pw_ifm_height;
 const int layer_43_pw_ofm_width = layer_43_pw_ifm_width;
 const int layer_43_pw_num_of_tiles_in_d = (int)(0.99 + (float)layer_43_pw_depth / pw_tile_d);
 const int layer_43_pw_num_of_tiles_out_d = (int)(0.99 + (float)layer_43_pw_num_fils / pw_conv_parallelism_out);
 const int layer_43_pw_num_of_tiles_w = (int)(0.99 + (float)layer_43_pw_ofm_width / pw_tile_w); 
 const int layer_43_pw_num_of_tiles_h = (int)(0.99 + (float)layer_43_pw_ofm_height / pw_tile_h); 
 const int layer_43_pw_num_of_weight_groups_for_one_pass = layer_43_pw_depth * pw_conv_parallelism_out / weights_group_items; 
 const int layer_43_pw_weights_offset = 639872; 
 const int layer_43_relu = 6;
//****************************
const int layer_44_dw_num_fils = layer_43_pw_num_fils / alpha;
 const int layer_44_dw_depth = layer_44_dw_num_fils;
 const int layer_44_dw_strides = 1;
 const int layer_44_dw_ifm_height = layer_43_pw_ofm_height;
 const int layer_44_dw_ifm_width = layer_43_pw_ofm_width;
 const int layer_44_dw_ofm_height = layer_44_dw_ifm_height / layer_44_dw_strides;
 const int layer_44_dw_ofm_width = layer_44_dw_ifm_width / layer_44_dw_strides;
 const int layer_44_dw_padding_left = 1.0;
 const int layer_44_dw_padding_right = 1.0;
 const int layer_44_dw_filter_size = 3;
 const int layer_44_dw_num_of_tiles_in_d = (int)(((float)layer_44_dw_depth / dw_tile_d) + 0.5);
 const int layer_44_dw_num_of_tiles_w = layer_44_dw_ofm_width / dw_tile_w; 
 const int layer_44_dw_num_of_tiles_h = layer_44_dw_ofm_height / dw_tile_h; 
 //****************************
//****************************
 const int layer_45_pw_num_fils = 160 / alpha;
 const int layer_45_pw_depth = layer_44_dw_depth;
 const int layer_45_pw_ifm_height = layer_44_dw_ofm_height;
 const int layer_45_pw_ifm_width = layer_44_dw_ofm_width;
 const int layer_45_pw_ofm_height = layer_45_pw_ifm_height;
 const int layer_45_pw_ofm_width = layer_45_pw_ifm_width;
 const int layer_45_pw_num_of_tiles_in_d = (int)(0.99 + (float)layer_45_pw_depth / pw_tile_d);
 const int layer_45_pw_num_of_tiles_out_d = (int)(0.99 + (float)layer_45_pw_num_fils / pw_conv_parallelism_out);
 const int layer_45_pw_num_of_tiles_w = (int)(0.99 + (float)layer_45_pw_ofm_width / pw_tile_w); 
 const int layer_45_pw_num_of_tiles_h = (int)(0.99 + (float)layer_45_pw_ofm_height / pw_tile_h); 
 const int layer_45_pw_num_of_weight_groups_for_one_pass = layer_45_pw_depth * pw_conv_parallelism_out / weights_group_items; 
 const int layer_45_pw_weights_offset = 793472; 
 const int layer_45_relu = 0;
//****************************
//****************************
 const int layer_46_pw_num_fils = 960 / alpha;
 const int layer_46_pw_depth = layer_45_pw_num_fils;
 const int layer_46_pw_ifm_height = layer_45_pw_ofm_height;
 const int layer_46_pw_ifm_width = layer_45_pw_ofm_width;
 const int layer_46_pw_ofm_height = layer_46_pw_ifm_height;
 const int layer_46_pw_ofm_width = layer_46_pw_ifm_width;
 const int layer_46_pw_num_of_tiles_in_d = (int)(0.99 + (float)layer_46_pw_depth / pw_tile_d);
 const int layer_46_pw_num_of_tiles_out_d = (int)(0.99 + (float)layer_46_pw_num_fils / pw_conv_parallelism_out);
 const int layer_46_pw_num_of_tiles_w = (int)(0.99 + (float)layer_46_pw_ofm_width / pw_tile_w); 
 const int layer_46_pw_num_of_tiles_h = (int)(0.99 + (float)layer_46_pw_ofm_height / pw_tile_h); 
 const int layer_46_pw_num_of_weight_groups_for_one_pass = layer_46_pw_depth * pw_conv_parallelism_out / weights_group_items; 
 const int layer_46_pw_weights_offset = 947072; 
 const int layer_46_relu = 6;
//****************************
const int layer_47_dw_num_fils = layer_46_pw_num_fils / alpha;
 const int layer_47_dw_depth = layer_47_dw_num_fils;
 const int layer_47_dw_strides = 1;
 const int layer_47_dw_ifm_height = layer_46_pw_ofm_height;
 const int layer_47_dw_ifm_width = layer_46_pw_ofm_width;
 const int layer_47_dw_ofm_height = layer_47_dw_ifm_height / layer_47_dw_strides;
 const int layer_47_dw_ofm_width = layer_47_dw_ifm_width / layer_47_dw_strides;
 const int layer_47_dw_padding_left = 1.0;
 const int layer_47_dw_padding_right = 1.0;
 const int layer_47_dw_filter_size = 3;
 const int layer_47_dw_num_of_tiles_in_d = (int)(((float)layer_47_dw_depth / dw_tile_d) + 0.5);
 const int layer_47_dw_num_of_tiles_w = layer_47_dw_ofm_width / dw_tile_w; 
 const int layer_47_dw_num_of_tiles_h = layer_47_dw_ofm_height / dw_tile_h; 
 //****************************
//****************************
 const int layer_48_pw_num_fils = 160 / alpha;
 const int layer_48_pw_depth = layer_47_dw_depth;
 const int layer_48_pw_ifm_height = layer_47_dw_ofm_height;
 const int layer_48_pw_ifm_width = layer_47_dw_ofm_width;
 const int layer_48_pw_ofm_height = layer_48_pw_ifm_height;
 const int layer_48_pw_ofm_width = layer_48_pw_ifm_width;
 const int layer_48_pw_num_of_tiles_in_d = (int)(0.99 + (float)layer_48_pw_depth / pw_tile_d);
 const int layer_48_pw_num_of_tiles_out_d = (int)(0.99 + (float)layer_48_pw_num_fils / pw_conv_parallelism_out);
 const int layer_48_pw_num_of_tiles_w = (int)(0.99 + (float)layer_48_pw_ofm_width / pw_tile_w); 
 const int layer_48_pw_num_of_tiles_h = (int)(0.99 + (float)layer_48_pw_ofm_height / pw_tile_h); 
 const int layer_48_pw_num_of_weight_groups_for_one_pass = layer_48_pw_depth * pw_conv_parallelism_out / weights_group_items; 
 const int layer_48_pw_weights_offset = 1100672; 
 const int layer_48_relu = 0;
//****************************
//****************************
 const int layer_49_pw_num_fils = 960 / alpha;
 const int layer_49_pw_depth = layer_48_pw_num_fils;
 const int layer_49_pw_ifm_height = layer_48_pw_ofm_height;
 const int layer_49_pw_ifm_width = layer_48_pw_ofm_width;
 const int layer_49_pw_ofm_height = layer_49_pw_ifm_height;
 const int layer_49_pw_ofm_width = layer_49_pw_ifm_width;
 const int layer_49_pw_num_of_tiles_in_d = (int)(0.99 + (float)layer_49_pw_depth / pw_tile_d);
 const int layer_49_pw_num_of_tiles_out_d = (int)(0.99 + (float)layer_49_pw_num_fils / pw_conv_parallelism_out);
 const int layer_49_pw_num_of_tiles_w = (int)(0.99 + (float)layer_49_pw_ofm_width / pw_tile_w); 
 const int layer_49_pw_num_of_tiles_h = (int)(0.99 + (float)layer_49_pw_ofm_height / pw_tile_h); 
 const int layer_49_pw_num_of_weight_groups_for_one_pass = layer_49_pw_depth * pw_conv_parallelism_out / weights_group_items; 
 const int layer_49_pw_weights_offset = 1254272; 
 const int layer_49_relu = 6;
//****************************
const int layer_50_dw_num_fils = layer_49_pw_num_fils / alpha;
 const int layer_50_dw_depth = layer_50_dw_num_fils;
 const int layer_50_dw_strides = 1;
 const int layer_50_dw_ifm_height = layer_49_pw_ofm_height;
 const int layer_50_dw_ifm_width = layer_49_pw_ofm_width;
 const int layer_50_dw_ofm_height = layer_50_dw_ifm_height / layer_50_dw_strides;
 const int layer_50_dw_ofm_width = layer_50_dw_ifm_width / layer_50_dw_strides;
 const int layer_50_dw_padding_left = 1.0;
 const int layer_50_dw_padding_right = 1.0;
 const int layer_50_dw_filter_size = 3;
 const int layer_50_dw_num_of_tiles_in_d = (int)(((float)layer_50_dw_depth / dw_tile_d) + 0.5);
 const int layer_50_dw_num_of_tiles_w = layer_50_dw_ofm_width / dw_tile_w; 
 const int layer_50_dw_num_of_tiles_h = layer_50_dw_ofm_height / dw_tile_h; 
 //****************************
//****************************
 const int layer_51_pw_num_fils = 320 / alpha;
 const int layer_51_pw_depth = layer_50_dw_depth;
 const int layer_51_pw_ifm_height = layer_50_dw_ofm_height;
 const int layer_51_pw_ifm_width = layer_50_dw_ofm_width;
 const int layer_51_pw_ofm_height = layer_51_pw_ifm_height;
 const int layer_51_pw_ofm_width = layer_51_pw_ifm_width;
 const int layer_51_pw_num_of_tiles_in_d = (int)(0.99 + (float)layer_51_pw_depth / pw_tile_d);
 const int layer_51_pw_num_of_tiles_out_d = (int)(0.99 + (float)layer_51_pw_num_fils / pw_conv_parallelism_out);
 const int layer_51_pw_num_of_tiles_w = (int)(0.99 + (float)layer_51_pw_ofm_width / pw_tile_w); 
 const int layer_51_pw_num_of_tiles_h = (int)(0.99 + (float)layer_51_pw_ofm_height / pw_tile_h); 
 const int layer_51_pw_num_of_weight_groups_for_one_pass = layer_51_pw_depth * pw_conv_parallelism_out / weights_group_items; 
 const int layer_51_pw_weights_offset = 1407872; 
 const int layer_51_relu = 0;
//****************************
//****************************
 const int layer_52_pw_num_fils = 1280 / alpha;
 const int layer_52_pw_depth = layer_51_pw_num_fils;
 const int layer_52_pw_ifm_height = layer_51_pw_ofm_height;
 const int layer_52_pw_ifm_width = layer_51_pw_ofm_width;
 const int layer_52_pw_ofm_height = layer_52_pw_ifm_height;
 const int layer_52_pw_ofm_width = layer_52_pw_ifm_width;
 const int layer_52_pw_num_of_tiles_in_d = (int)(0.99 + (float)layer_52_pw_depth / pw_tile_d);
 const int layer_52_pw_num_of_tiles_out_d = (int)(0.99 + (float)layer_52_pw_num_fils / pw_conv_parallelism_out);
 const int layer_52_pw_num_of_tiles_w = (int)(0.99 + (float)layer_52_pw_ofm_width / pw_tile_w); 
 const int layer_52_pw_num_of_tiles_h = (int)(0.99 + (float)layer_52_pw_ofm_height / pw_tile_h); 
 const int layer_52_pw_num_of_weight_groups_for_one_pass = layer_52_pw_depth * pw_conv_parallelism_out / weights_group_items; 
 const int layer_52_pw_weights_offset = 1715072; 
 const int layer_52_relu = 6;
//****************************
#endif
