#include "../../basic_defs/basic_defs_glue.h"
#if PIPELINE_LENGTH == 6
#ifndef LAYERS_SPECS
#define LAYERS_SPECS
const Quantization_layer_specs quantize_layer_specs = {
                1.0,//const pooling_fused_scales_dt fused_scale; 
                127,//const biases_dt ifms_zero_point;
                -1//const biases_dt ofms_zero_point;
                };
//****************************
 const int first_conv_layer_num_fils = 32;
const int first_conv_layer_depth = 3;
const int first_conv_layer_filter_dim = 3;
 const int first_conv_layer_strides = 2;
 const int first_conv_layer_padding_left = 0;
 const int first_conv_layer_padding_right = 1;
 const int first_conv_layer_ifm_width = 224;
 //****************************
const layer_specs first_conv_layer_specs = {
                1,//layer_index;
                S_CONV,//conv_layer_type;; 
                32,//layer_num_fils 
                2,//strides;
                3,//filter_size;
                0,//padding_left;
                1,//padding_right;
                0,//padding_top;
                1,//padding_bottom;
                3,//layer_depth;
                224,//layer_ifm_height;
                224,//layer_ifm_width;
                112,//layer_ofm_height;
                112,//layer_ofm_width;
                RELU6,//layer_activation;
                (3 + pw_tile_d - 1) / pw_tile_d,//layer_num_of_tiles_in_d;
                (32 + pw_conv_parallelism_out - 1) / pw_conv_parallelism_out,//layer_num_of_tiles_out_d;
                (224 + pw_tile_h - 1) / pw_tile_h,//layer_num_of_ifm_tiles_h;
                (224 + pw_tile_w - 1) / pw_tile_w,//layer_num_of_ifm_tiles_w;
                (112 + pw_tile_h - 1) / pw_tile_h,//layer_num_of_ofm_tiles_h;
                (112 + pw_tile_w - 1) / pw_tile_w,//layer_num_of_ofm_tiles_w;
                (3 * pw_conv_parallelism_out ) / weights_group_items,//layer_num_of_weight_groups_for_one_pass;
                0,//layer_weights_offset;
                0,//layer_weights_offset_on_chip;
                0,//dw_ifms_cumulative_width_offset;
                1,//bool write_to_result_or_channels;
                0,//bool write_to_tmp;
                0,//bool fused_with_add;
                -1,//fms_dt layer_ifms_zero_point;
                0.0235294122248888,//,fms_dt layer_ofms_scale;
                -128,//fms_dt layer_ofms_zero_point;
                1,//rec_scales_dt add_layer_scale_reciprocal;
                0,//biases_dt add_layer_zero_point;
                1,//scales_dt skip_connection_other_layer_scale;
                0//biases_dt skip_connection_other_layer_zero_point;
                };
//****************************
 const int layer_2_dw_num_fils = 32;
const int layer_2_dw_depth = 32;
const int layer_2_dw_filter_dim = 3;
 const int layer_2_dw_ifm_width = 112;
 //****************************
const layer_specs layer_2_dw_specs = {
                2,//layer_index;
                DW_CONV,//conv_layer_type;; 
                32,//layer_num_fils 
                1,//strides;
                3,//filter_size;
                1,//padding_left;
                1,//padding_right;
                1,//padding_top;
                1,//padding_bottom;
                32,//layer_depth;
                112,//layer_ifm_height;
                112,//layer_ifm_width;
                112,//layer_ofm_height;
                112,//layer_ofm_width;
                RELU6,//layer_activation;
                (32 + pw_tile_d - 1) / pw_tile_d,//layer_num_of_tiles_in_d;
                (32 + pw_conv_parallelism_out - 1) / pw_conv_parallelism_out,//layer_num_of_tiles_out_d;
                (112 + pw_tile_h - 1) / pw_tile_h,//layer_num_of_ifm_tiles_h;
                (112 + pw_tile_w - 1) / pw_tile_w,//layer_num_of_ifm_tiles_w;
                (112 + pw_tile_h - 1) / pw_tile_h,//layer_num_of_ofm_tiles_h;
                (112 + pw_tile_w - 1) / pw_tile_w,//layer_num_of_ofm_tiles_w;
                (32 * pw_conv_parallelism_out ) / weights_group_items,//layer_num_of_weight_groups_for_one_pass;
                0,//layer_weights_offset;
                0,//layer_weights_offset_on_chip;
                0,//dw_ifms_cumulative_width_offset;
                1,//bool write_to_result_or_channels;
                0,//bool write_to_tmp;
                0,//bool fused_with_add;
                -128,//fms_dt layer_ifms_zero_point;
                0.0235294122248888,//,fms_dt layer_ofms_scale;
                -128,//fms_dt layer_ofms_zero_point;
                1,//rec_scales_dt add_layer_scale_reciprocal;
                0,//biases_dt add_layer_zero_point;
                1,//scales_dt skip_connection_other_layer_scale;
                0//biases_dt skip_connection_other_layer_zero_point;
                };
//****************************
 const int layer_3_pw_num_fils = 16;
const int layer_3_pw_depth = 32;
const int layer_3_pw_filter_dim = 1;
 const int layer_3_pw_ifm_width = 112;
 //****************************
const layer_specs layer_3_pw_specs = {
                3,//layer_index;
                PW_CONV,//conv_layer_type;; 
                16,//layer_num_fils 
                1,//strides;
                1,//filter_size;
                0,//padding_left;
                0,//padding_right;
                0,//padding_top;
                0,//padding_bottom;
                32,//layer_depth;
                112,//layer_ifm_height;
                112,//layer_ifm_width;
                112,//layer_ofm_height;
                112,//layer_ofm_width;
                0,//layer_activation;
                (32 + pw_tile_d - 1) / pw_tile_d,//layer_num_of_tiles_in_d;
                (16 + pw_conv_parallelism_out - 1) / pw_conv_parallelism_out,//layer_num_of_tiles_out_d;
                (112 + pw_tile_h - 1) / pw_tile_h,//layer_num_of_ifm_tiles_h;
                (112 + pw_tile_w - 1) / pw_tile_w,//layer_num_of_ifm_tiles_w;
                (112 + pw_tile_h - 1) / pw_tile_h,//layer_num_of_ofm_tiles_h;
                (112 + pw_tile_w - 1) / pw_tile_w,//layer_num_of_ofm_tiles_w;
                (32 * pw_conv_parallelism_out ) / weights_group_items,//layer_num_of_weight_groups_for_one_pass;
                0 / weights_group_items,//layer_weights_offset;
                0,//layer_weights_offset_on_chip;
                0,//dw_ifms_cumulative_width_offset;
                1,//bool write_to_result_or_channels;
                0,//bool write_to_tmp;
                0,//bool fused_with_add;
                -128,//fms_dt layer_ifms_zero_point;
                0.3764464855194092,//,fms_dt layer_ofms_scale;
                4,//fms_dt layer_ofms_zero_point;
                1,//rec_scales_dt add_layer_scale_reciprocal;
                0,//biases_dt add_layer_zero_point;
                1,//scales_dt skip_connection_other_layer_scale;
                0//biases_dt skip_connection_other_layer_zero_point;
                };
//****************************
 const int layer_4_pw_num_fils = 96;
const int layer_4_pw_depth = 16;
const int layer_4_pw_filter_dim = 1;
 const int layer_4_pw_ifm_width = 112;
 //****************************
const layer_specs layer_4_pw_specs = {
                4,//layer_index;
                PW_CONV,//conv_layer_type;; 
                96,//layer_num_fils 
                1,//strides;
                1,//filter_size;
                0,//padding_left;
                0,//padding_right;
                0,//padding_top;
                0,//padding_bottom;
                16,//layer_depth;
                112,//layer_ifm_height;
                112,//layer_ifm_width;
                112,//layer_ofm_height;
                112,//layer_ofm_width;
                RELU6,//layer_activation;
                (16 + pw_tile_d - 1) / pw_tile_d,//layer_num_of_tiles_in_d;
                (96 + pw_conv_parallelism_out - 1) / pw_conv_parallelism_out,//layer_num_of_tiles_out_d;
                (112 + pw_tile_h - 1) / pw_tile_h,//layer_num_of_ifm_tiles_h;
                (112 + pw_tile_w - 1) / pw_tile_w,//layer_num_of_ifm_tiles_w;
                (112 + pw_tile_h - 1) / pw_tile_h,//layer_num_of_ofm_tiles_h;
                (112 + pw_tile_w - 1) / pw_tile_w,//layer_num_of_ofm_tiles_w;
                (16 * pw_conv_parallelism_out ) / weights_group_items,//layer_num_of_weight_groups_for_one_pass;
                0 / weights_group_items,//layer_weights_offset;
                512,//layer_weights_offset_on_chip;
                0,//dw_ifms_cumulative_width_offset;
                1,//bool write_to_result_or_channels;
                0,//bool write_to_tmp;
                0,//bool fused_with_add;
                4,//fms_dt layer_ifms_zero_point;
                0.0235294122248888,//,fms_dt layer_ofms_scale;
                -128,//fms_dt layer_ofms_zero_point;
                1,//rec_scales_dt add_layer_scale_reciprocal;
                0,//biases_dt add_layer_zero_point;
                1,//scales_dt skip_connection_other_layer_scale;
                0//biases_dt skip_connection_other_layer_zero_point;
                };
//****************************
 const int layer_6_dw_num_fils = 96;
const int layer_6_dw_depth = 96;
const int layer_6_dw_filter_dim = 3;
 const int layer_6_dw_ifm_width = 112;
 //****************************
const layer_specs layer_6_dw_specs = {
                6,//layer_index;
                DW_CONV,//conv_layer_type;; 
                96,//layer_num_fils 
                2,//strides;
                3,//filter_size;
                0,//padding_left;
                1,//padding_right;
                0,//padding_top;
                1,//padding_bottom;
                96,//layer_depth;
                112,//layer_ifm_height;
                112,//layer_ifm_width;
                56,//layer_ofm_height;
                56,//layer_ofm_width;
                RELU6,//layer_activation;
                (96 + pw_tile_d - 1) / pw_tile_d,//layer_num_of_tiles_in_d;
                (96 + pw_conv_parallelism_out - 1) / pw_conv_parallelism_out,//layer_num_of_tiles_out_d;
                (112 + pw_tile_h - 1) / pw_tile_h,//layer_num_of_ifm_tiles_h;
                (112 + pw_tile_w - 1) / pw_tile_w,//layer_num_of_ifm_tiles_w;
                (56 + pw_tile_h - 1) / pw_tile_h,//layer_num_of_ofm_tiles_h;
                (56 + pw_tile_w - 1) / pw_tile_w,//layer_num_of_ofm_tiles_w;
                (96 * pw_conv_parallelism_out ) / weights_group_items,//layer_num_of_weight_groups_for_one_pass;
                32,//layer_weights_offset;
                0,//layer_weights_offset_on_chip;
                7168,//dw_ifms_cumulative_width_offset;
                1,//bool write_to_result_or_channels;
                0,//bool write_to_tmp;
                0,//bool fused_with_add;
                -128,//fms_dt layer_ifms_zero_point;
                0.0235294122248888,//,fms_dt layer_ofms_scale;
                -128,//fms_dt layer_ofms_zero_point;
                1,//rec_scales_dt add_layer_scale_reciprocal;
                0,//biases_dt add_layer_zero_point;
                1,//scales_dt skip_connection_other_layer_scale;
                0//biases_dt skip_connection_other_layer_zero_point;
                };
//****************************
 const int layer_7_pw_num_fils = 24;
const int layer_7_pw_depth = 96;
const int layer_7_pw_filter_dim = 1;
 const int layer_7_pw_ifm_width = 56;
 //****************************
const layer_specs layer_7_pw_specs = {
                7,//layer_index;
                PW_CONV,//conv_layer_type;; 
                24,//layer_num_fils 
                1,//strides;
                1,//filter_size;
                0,//padding_left;
                0,//padding_right;
                0,//padding_top;
                0,//padding_bottom;
                96,//layer_depth;
                56,//layer_ifm_height;
                56,//layer_ifm_width;
                56,//layer_ofm_height;
                56,//layer_ofm_width;
                0,//layer_activation;
                (96 + pw_tile_d - 1) / pw_tile_d,//layer_num_of_tiles_in_d;
                (24 + pw_conv_parallelism_out - 1) / pw_conv_parallelism_out,//layer_num_of_tiles_out_d;
                (56 + pw_tile_h - 1) / pw_tile_h,//layer_num_of_ifm_tiles_h;
                (56 + pw_tile_w - 1) / pw_tile_w,//layer_num_of_ifm_tiles_w;
                (56 + pw_tile_h - 1) / pw_tile_h,//layer_num_of_ofm_tiles_h;
                (56 + pw_tile_w - 1) / pw_tile_w,//layer_num_of_ofm_tiles_w;
                (96 * pw_conv_parallelism_out ) / weights_group_items,//layer_num_of_weight_groups_for_one_pass;
                0 / weights_group_items,//layer_weights_offset;
                2048,//layer_weights_offset_on_chip;
                0,//dw_ifms_cumulative_width_offset;
                1,//bool write_to_result_or_channels;
                1,//bool write_to_tmp;
                0,//bool fused_with_add;
                -128,//fms_dt layer_ifms_zero_point;
                0.30003005266189575,//,fms_dt layer_ofms_scale;
                -11,//fms_dt layer_ofms_zero_point;
                1,//rec_scales_dt add_layer_scale_reciprocal;
                0,//biases_dt add_layer_zero_point;
                1,//scales_dt skip_connection_other_layer_scale;
                0//biases_dt skip_connection_other_layer_zero_point;
                };
//****************************
 const int layer_8_pw_num_fils = 144;
const int layer_8_pw_depth = 24;
const int layer_8_pw_filter_dim = 1;
 const int layer_8_pw_ifm_width = 56;
 //****************************
const layer_specs layer_8_pw_specs = {
                8,//layer_index;
                PW_CONV,//conv_layer_type;; 
                144,//layer_num_fils 
                1,//strides;
                1,//filter_size;
                0,//padding_left;
                0,//padding_right;
                0,//padding_top;
                0,//padding_bottom;
                24,//layer_depth;
                56,//layer_ifm_height;
                56,//layer_ifm_width;
                56,//layer_ofm_height;
                56,//layer_ofm_width;
                RELU6,//layer_activation;
                (24 + pw_tile_d - 1) / pw_tile_d,//layer_num_of_tiles_in_d;
                (144 + pw_conv_parallelism_out - 1) / pw_conv_parallelism_out,//layer_num_of_tiles_out_d;
                (56 + pw_tile_h - 1) / pw_tile_h,//layer_num_of_ifm_tiles_h;
                (56 + pw_tile_w - 1) / pw_tile_w,//layer_num_of_ifm_tiles_w;
                (56 + pw_tile_h - 1) / pw_tile_h,//layer_num_of_ofm_tiles_h;
                (56 + pw_tile_w - 1) / pw_tile_w,//layer_num_of_ofm_tiles_w;
                (24 * pw_conv_parallelism_out ) / weights_group_items,//layer_num_of_weight_groups_for_one_pass;
                0 / weights_group_items,//layer_weights_offset;
                4352,//layer_weights_offset_on_chip;
                0,//dw_ifms_cumulative_width_offset;
                1,//bool write_to_result_or_channels;
                0,//bool write_to_tmp;
                0,//bool fused_with_add;
                -11,//fms_dt layer_ifms_zero_point;
                0.0235294122248888,//,fms_dt layer_ofms_scale;
                -128,//fms_dt layer_ofms_zero_point;
                1,//rec_scales_dt add_layer_scale_reciprocal;
                0,//biases_dt add_layer_zero_point;
                1,//scales_dt skip_connection_other_layer_scale;
                0//biases_dt skip_connection_other_layer_zero_point;
                };
//****************************
 const int layer_9_dw_num_fils = 144;
const int layer_9_dw_depth = 144;
const int layer_9_dw_filter_dim = 3;
 const int layer_9_dw_ifm_width = 56;
 //****************************
const layer_specs layer_9_dw_specs = {
                9,//layer_index;
                DW_CONV,//conv_layer_type;; 
                144,//layer_num_fils 
                1,//strides;
                3,//filter_size;
                1,//padding_left;
                1,//padding_right;
                1,//padding_top;
                1,//padding_bottom;
                144,//layer_depth;
                56,//layer_ifm_height;
                56,//layer_ifm_width;
                56,//layer_ofm_height;
                56,//layer_ofm_width;
                RELU6,//layer_activation;
                (144 + pw_tile_d - 1) / pw_tile_d,//layer_num_of_tiles_in_d;
                (144 + pw_conv_parallelism_out - 1) / pw_conv_parallelism_out,//layer_num_of_tiles_out_d;
                (56 + pw_tile_h - 1) / pw_tile_h,//layer_num_of_ifm_tiles_h;
                (56 + pw_tile_w - 1) / pw_tile_w,//layer_num_of_ifm_tiles_w;
                (56 + pw_tile_h - 1) / pw_tile_h,//layer_num_of_ofm_tiles_h;
                (56 + pw_tile_w - 1) / pw_tile_w,//layer_num_of_ofm_tiles_w;
                (144 * pw_conv_parallelism_out ) / weights_group_items,//layer_num_of_weight_groups_for_one_pass;
                128,//layer_weights_offset;
                0,//layer_weights_offset_on_chip;
                17920,//dw_ifms_cumulative_width_offset;
                1,//bool write_to_result_or_channels;
                0,//bool write_to_tmp;
                0,//bool fused_with_add;
                -128,//fms_dt layer_ifms_zero_point;
                0.0235294122248888,//,fms_dt layer_ofms_scale;
                -128,//fms_dt layer_ofms_zero_point;
                1,//rec_scales_dt add_layer_scale_reciprocal;
                0,//biases_dt add_layer_zero_point;
                1,//scales_dt skip_connection_other_layer_scale;
                0//biases_dt skip_connection_other_layer_zero_point;
                };
//****************************
 const int layer_10_pw_num_fils = 24;
const int layer_10_pw_depth = 144;
const int layer_10_pw_filter_dim = 1;
 const int layer_10_pw_ifm_width = 56;
 //****************************
const layer_specs layer_10_pw_specs = {
                10,//layer_index;
                PW_CONV,//conv_layer_type;; 
                24,//layer_num_fils 
                1,//strides;
                1,//filter_size;
                0,//padding_left;
                0,//padding_right;
                0,//padding_top;
                0,//padding_bottom;
                144,//layer_depth;
                56,//layer_ifm_height;
                56,//layer_ifm_width;
                56,//layer_ofm_height;
                56,//layer_ofm_width;
                0,//layer_activation;
                (144 + pw_tile_d - 1) / pw_tile_d,//layer_num_of_tiles_in_d;
                (24 + pw_conv_parallelism_out - 1) / pw_conv_parallelism_out,//layer_num_of_tiles_out_d;
                (56 + pw_tile_h - 1) / pw_tile_h,//layer_num_of_ifm_tiles_h;
                (56 + pw_tile_w - 1) / pw_tile_w,//layer_num_of_ifm_tiles_w;
                (56 + pw_tile_h - 1) / pw_tile_h,//layer_num_of_ofm_tiles_h;
                (56 + pw_tile_w - 1) / pw_tile_w,//layer_num_of_ofm_tiles_w;
                (144 * pw_conv_parallelism_out ) / weights_group_items,//layer_num_of_weight_groups_for_one_pass;
                3456 / weights_group_items,//layer_weights_offset;
                7808,//layer_weights_offset_on_chip;
                0,//dw_ifms_cumulative_width_offset;
                1,//bool write_to_result_or_channels;
                0,//bool write_to_tmp;
                1,//bool fused_with_add;
                -128,//fms_dt layer_ifms_zero_point;
                0.41785937547683716,//,fms_dt layer_ofms_scale;
                6,//fms_dt layer_ofms_zero_point;
                2.04375319960446,//rec_scales_dt add_layer_scale_reciprocal;
                -3,//biases_dt add_layer_zero_point;
                0.30003005266189575,//scales_dt skip_connection_other_layer_scale;
                -11//biases_dt skip_connection_other_layer_zero_point;
                };
//****************************
 const int layer_12_pw_num_fils = 144;
const int layer_12_pw_depth = 24;
const int layer_12_pw_filter_dim = 1;
 const int layer_12_pw_ifm_width = 56;
 //****************************
const layer_specs layer_12_pw_specs = {
                12,//layer_index;
                PW_CONV,//conv_layer_type;; 
                144,//layer_num_fils 
                1,//strides;
                1,//filter_size;
                0,//padding_left;
                0,//padding_right;
                0,//padding_top;
                0,//padding_bottom;
                24,//layer_depth;
                56,//layer_ifm_height;
                56,//layer_ifm_width;
                56,//layer_ofm_height;
                56,//layer_ofm_width;
                RELU6,//layer_activation;
                (24 + pw_tile_d - 1) / pw_tile_d,//layer_num_of_tiles_in_d;
                (144 + pw_conv_parallelism_out - 1) / pw_conv_parallelism_out,//layer_num_of_tiles_out_d;
                (56 + pw_tile_h - 1) / pw_tile_h,//layer_num_of_ifm_tiles_h;
                (56 + pw_tile_w - 1) / pw_tile_w,//layer_num_of_ifm_tiles_w;
                (56 + pw_tile_h - 1) / pw_tile_h,//layer_num_of_ofm_tiles_h;
                (56 + pw_tile_w - 1) / pw_tile_w,//layer_num_of_ofm_tiles_w;
                (24 * pw_conv_parallelism_out ) / weights_group_items,//layer_num_of_weight_groups_for_one_pass;
                6912 / weights_group_items,//layer_weights_offset;
                11264,//layer_weights_offset_on_chip;
                0,//dw_ifms_cumulative_width_offset;
                1,//bool write_to_result_or_channels;
                0,//bool write_to_tmp;
                0,//bool fused_with_add;
                -3,//fms_dt layer_ifms_zero_point;
                0.0235294122248888,//,fms_dt layer_ofms_scale;
                -128,//fms_dt layer_ofms_zero_point;
                1,//rec_scales_dt add_layer_scale_reciprocal;
                0,//biases_dt add_layer_zero_point;
                1,//scales_dt skip_connection_other_layer_scale;
                0//biases_dt skip_connection_other_layer_zero_point;
                };
//****************************
 const int layer_14_dw_num_fils = 144;
const int layer_14_dw_depth = 144;
const int layer_14_dw_filter_dim = 3;
 const int layer_14_dw_ifm_width = 56;
 //****************************
const layer_specs layer_14_dw_specs = {
                14,//layer_index;
                DW_CONV,//conv_layer_type;; 
                144,//layer_num_fils 
                2,//strides;
                3,//filter_size;
                0,//padding_left;
                1,//padding_right;
                0,//padding_top;
                1,//padding_bottom;
                144,//layer_depth;
                56,//layer_ifm_height;
                56,//layer_ifm_width;
                28,//layer_ofm_height;
                28,//layer_ofm_width;
                RELU6,//layer_activation;
                (144 + pw_tile_d - 1) / pw_tile_d,//layer_num_of_tiles_in_d;
                (144 + pw_conv_parallelism_out - 1) / pw_conv_parallelism_out,//layer_num_of_tiles_out_d;
                (56 + pw_tile_h - 1) / pw_tile_h,//layer_num_of_ifm_tiles_h;
                (56 + pw_tile_w - 1) / pw_tile_w,//layer_num_of_ifm_tiles_w;
                (28 + pw_tile_h - 1) / pw_tile_h,//layer_num_of_ofm_tiles_h;
                (28 + pw_tile_w - 1) / pw_tile_w,//layer_num_of_ofm_tiles_w;
                (144 * pw_conv_parallelism_out ) / weights_group_items,//layer_num_of_weight_groups_for_one_pass;
                272,//layer_weights_offset;
                0,//layer_weights_offset_on_chip;
                34048,//dw_ifms_cumulative_width_offset;
                1,//bool write_to_result_or_channels;
                0,//bool write_to_tmp;
                0,//bool fused_with_add;
                -128,//fms_dt layer_ifms_zero_point;
                0.0235294122248888,//,fms_dt layer_ofms_scale;
                -128,//fms_dt layer_ofms_zero_point;
                1,//rec_scales_dt add_layer_scale_reciprocal;
                0,//biases_dt add_layer_zero_point;
                1,//scales_dt skip_connection_other_layer_scale;
                0//biases_dt skip_connection_other_layer_zero_point;
                };
//****************************
 const int layer_15_pw_num_fils = 32;
const int layer_15_pw_depth = 144;
const int layer_15_pw_filter_dim = 1;
 const int layer_15_pw_ifm_width = 28;
 //****************************
const layer_specs layer_15_pw_specs = {
                15,//layer_index;
                PW_CONV,//conv_layer_type;; 
                32,//layer_num_fils 
                1,//strides;
                1,//filter_size;
                0,//padding_left;
                0,//padding_right;
                0,//padding_top;
                0,//padding_bottom;
                144,//layer_depth;
                28,//layer_ifm_height;
                28,//layer_ifm_width;
                28,//layer_ofm_height;
                28,//layer_ofm_width;
                0,//layer_activation;
                (144 + pw_tile_d - 1) / pw_tile_d,//layer_num_of_tiles_in_d;
                (32 + pw_conv_parallelism_out - 1) / pw_conv_parallelism_out,//layer_num_of_tiles_out_d;
                (28 + pw_tile_h - 1) / pw_tile_h,//layer_num_of_ifm_tiles_h;
                (28 + pw_tile_w - 1) / pw_tile_w,//layer_num_of_ifm_tiles_w;
                (28 + pw_tile_h - 1) / pw_tile_h,//layer_num_of_ofm_tiles_h;
                (28 + pw_tile_w - 1) / pw_tile_w,//layer_num_of_ofm_tiles_w;
                (144 * pw_conv_parallelism_out ) / weights_group_items,//layer_num_of_weight_groups_for_one_pass;
                10368 / weights_group_items,//layer_weights_offset;
                14720,//layer_weights_offset_on_chip;
                0,//dw_ifms_cumulative_width_offset;
                1,//bool write_to_result_or_channels;
                1,//bool write_to_tmp;
                0,//bool fused_with_add;
                -128,//fms_dt layer_ifms_zero_point;
                0.21887806057929993,//,fms_dt layer_ofms_scale;
                -6,//fms_dt layer_ofms_zero_point;
                1,//rec_scales_dt add_layer_scale_reciprocal;
                0,//biases_dt add_layer_zero_point;
                1,//scales_dt skip_connection_other_layer_scale;
                0//biases_dt skip_connection_other_layer_zero_point;
                };
//****************************
 const int layer_16_pw_num_fils = 192;
const int layer_16_pw_depth = 32;
const int layer_16_pw_filter_dim = 1;
 const int layer_16_pw_ifm_width = 28;
 //****************************
const layer_specs layer_16_pw_specs = {
                16,//layer_index;
                PW_CONV,//conv_layer_type;; 
                192,//layer_num_fils 
                1,//strides;
                1,//filter_size;
                0,//padding_left;
                0,//padding_right;
                0,//padding_top;
                0,//padding_bottom;
                32,//layer_depth;
                28,//layer_ifm_height;
                28,//layer_ifm_width;
                28,//layer_ofm_height;
                28,//layer_ofm_width;
                RELU6,//layer_activation;
                (32 + pw_tile_d - 1) / pw_tile_d,//layer_num_of_tiles_in_d;
                (192 + pw_conv_parallelism_out - 1) / pw_conv_parallelism_out,//layer_num_of_tiles_out_d;
                (28 + pw_tile_h - 1) / pw_tile_h,//layer_num_of_ifm_tiles_h;
                (28 + pw_tile_w - 1) / pw_tile_w,//layer_num_of_ifm_tiles_w;
                (28 + pw_tile_h - 1) / pw_tile_h,//layer_num_of_ofm_tiles_h;
                (28 + pw_tile_w - 1) / pw_tile_w,//layer_num_of_ofm_tiles_w;
                (32 * pw_conv_parallelism_out ) / weights_group_items,//layer_num_of_weight_groups_for_one_pass;
                14976 / weights_group_items,//layer_weights_offset;
                19328,//layer_weights_offset_on_chip;
                0,//dw_ifms_cumulative_width_offset;
                1,//bool write_to_result_or_channels;
                0,//bool write_to_tmp;
                0,//bool fused_with_add;
                -6,//fms_dt layer_ifms_zero_point;
                0.0235294122248888,//,fms_dt layer_ofms_scale;
                -128,//fms_dt layer_ofms_zero_point;
                1,//rec_scales_dt add_layer_scale_reciprocal;
                0,//biases_dt add_layer_zero_point;
                1,//scales_dt skip_connection_other_layer_scale;
                0//biases_dt skip_connection_other_layer_zero_point;
                };
//****************************
 const int layer_17_dw_num_fils = 192;
const int layer_17_dw_depth = 192;
const int layer_17_dw_filter_dim = 3;
 const int layer_17_dw_ifm_width = 28;
 //****************************
const layer_specs layer_17_dw_specs = {
                17,//layer_index;
                DW_CONV,//conv_layer_type;; 
                192,//layer_num_fils 
                1,//strides;
                3,//filter_size;
                1,//padding_left;
                1,//padding_right;
                1,//padding_top;
                1,//padding_bottom;
                192,//layer_depth;
                28,//layer_ifm_height;
                28,//layer_ifm_width;
                28,//layer_ofm_height;
                28,//layer_ofm_width;
                RELU6,//layer_activation;
                (192 + pw_tile_d - 1) / pw_tile_d,//layer_num_of_tiles_in_d;
                (192 + pw_conv_parallelism_out - 1) / pw_conv_parallelism_out,//layer_num_of_tiles_out_d;
                (28 + pw_tile_h - 1) / pw_tile_h,//layer_num_of_ifm_tiles_h;
                (28 + pw_tile_w - 1) / pw_tile_w,//layer_num_of_ifm_tiles_w;
                (28 + pw_tile_h - 1) / pw_tile_h,//layer_num_of_ofm_tiles_h;
                (28 + pw_tile_w - 1) / pw_tile_w,//layer_num_of_ofm_tiles_w;
                (192 * pw_conv_parallelism_out ) / weights_group_items,//layer_num_of_weight_groups_for_one_pass;
                416,//layer_weights_offset;
                0,//layer_weights_offset_on_chip;
                42112,//dw_ifms_cumulative_width_offset;
                1,//bool write_to_result_or_channels;
                0,//bool write_to_tmp;
                0,//bool fused_with_add;
                -128,//fms_dt layer_ifms_zero_point;
                0.0235294122248888,//,fms_dt layer_ofms_scale;
                -128,//fms_dt layer_ofms_zero_point;
                1,//rec_scales_dt add_layer_scale_reciprocal;
                0,//biases_dt add_layer_zero_point;
                1,//scales_dt skip_connection_other_layer_scale;
                0//biases_dt skip_connection_other_layer_zero_point;
                };
//****************************
 const int layer_18_pw_num_fils = 32;
const int layer_18_pw_depth = 192;
const int layer_18_pw_filter_dim = 1;
 const int layer_18_pw_ifm_width = 28;
 //****************************
const layer_specs layer_18_pw_specs = {
                18,//layer_index;
                PW_CONV,//conv_layer_type;; 
                32,//layer_num_fils 
                1,//strides;
                1,//filter_size;
                0,//padding_left;
                0,//padding_right;
                0,//padding_top;
                0,//padding_bottom;
                192,//layer_depth;
                28,//layer_ifm_height;
                28,//layer_ifm_width;
                28,//layer_ofm_height;
                28,//layer_ofm_width;
                0,//layer_activation;
                (192 + pw_tile_d - 1) / pw_tile_d,//layer_num_of_tiles_in_d;
                (32 + pw_conv_parallelism_out - 1) / pw_conv_parallelism_out,//layer_num_of_tiles_out_d;
                (28 + pw_tile_h - 1) / pw_tile_h,//layer_num_of_ifm_tiles_h;
                (28 + pw_tile_w - 1) / pw_tile_w,//layer_num_of_ifm_tiles_w;
                (28 + pw_tile_h - 1) / pw_tile_h,//layer_num_of_ofm_tiles_h;
                (28 + pw_tile_w - 1) / pw_tile_w,//layer_num_of_ofm_tiles_w;
                (192 * pw_conv_parallelism_out ) / weights_group_items,//layer_num_of_weight_groups_for_one_pass;
                21120 / weights_group_items,//layer_weights_offset;
                25472,//layer_weights_offset_on_chip;
                0,//dw_ifms_cumulative_width_offset;
                1,//bool write_to_result_or_channels;
                1,//bool write_to_tmp;
                1,//bool fused_with_add;
                -128,//fms_dt layer_ifms_zero_point;
                0.22161531448364258,//,fms_dt layer_ofms_scale;
                -8,//fms_dt layer_ofms_zero_point;
                3.7931430840069655,//rec_scales_dt add_layer_scale_reciprocal;
                7,//biases_dt add_layer_zero_point;
                0.21887806057929993,//scales_dt skip_connection_other_layer_scale;
                -6//biases_dt skip_connection_other_layer_zero_point;
                };
//****************************
 const int layer_20_pw_num_fils = 192;
const int layer_20_pw_depth = 32;
const int layer_20_pw_filter_dim = 1;
 const int layer_20_pw_ifm_width = 28;
 //****************************
const layer_specs layer_20_pw_specs = {
                20,//layer_index;
                PW_CONV,//conv_layer_type;; 
                192,//layer_num_fils 
                1,//strides;
                1,//filter_size;
                0,//padding_left;
                0,//padding_right;
                0,//padding_top;
                0,//padding_bottom;
                32,//layer_depth;
                28,//layer_ifm_height;
                28,//layer_ifm_width;
                28,//layer_ofm_height;
                28,//layer_ofm_width;
                RELU6,//layer_activation;
                (32 + pw_tile_d - 1) / pw_tile_d,//layer_num_of_tiles_in_d;
                (192 + pw_conv_parallelism_out - 1) / pw_conv_parallelism_out,//layer_num_of_tiles_out_d;
                (28 + pw_tile_h - 1) / pw_tile_h,//layer_num_of_ifm_tiles_h;
                (28 + pw_tile_w - 1) / pw_tile_w,//layer_num_of_ifm_tiles_w;
                (28 + pw_tile_h - 1) / pw_tile_h,//layer_num_of_ofm_tiles_h;
                (28 + pw_tile_w - 1) / pw_tile_w,//layer_num_of_ofm_tiles_w;
                (32 * pw_conv_parallelism_out ) / weights_group_items,//layer_num_of_weight_groups_for_one_pass;
                27264 / weights_group_items,//layer_weights_offset;
                31616,//layer_weights_offset_on_chip;
                0,//dw_ifms_cumulative_width_offset;
                1,//bool write_to_result_or_channels;
                0,//bool write_to_tmp;
                0,//bool fused_with_add;
                7,//fms_dt layer_ifms_zero_point;
                0.0235294122248888,//,fms_dt layer_ofms_scale;
                -128,//fms_dt layer_ofms_zero_point;
                1,//rec_scales_dt add_layer_scale_reciprocal;
                0,//biases_dt add_layer_zero_point;
                1,//scales_dt skip_connection_other_layer_scale;
                0//biases_dt skip_connection_other_layer_zero_point;
                };
//****************************
 const int layer_21_dw_num_fils = 192;
const int layer_21_dw_depth = 192;
const int layer_21_dw_filter_dim = 3;
 const int layer_21_dw_ifm_width = 28;
 //****************************
const layer_specs layer_21_dw_specs = {
                21,//layer_index;
                DW_CONV,//conv_layer_type;; 
                192,//layer_num_fils 
                1,//strides;
                3,//filter_size;
                1,//padding_left;
                1,//padding_right;
                1,//padding_top;
                1,//padding_bottom;
                192,//layer_depth;
                28,//layer_ifm_height;
                28,//layer_ifm_width;
                28,//layer_ofm_height;
                28,//layer_ofm_width;
                RELU6,//layer_activation;
                (192 + pw_tile_d - 1) / pw_tile_d,//layer_num_of_tiles_in_d;
                (192 + pw_conv_parallelism_out - 1) / pw_conv_parallelism_out,//layer_num_of_tiles_out_d;
                (28 + pw_tile_h - 1) / pw_tile_h,//layer_num_of_ifm_tiles_h;
                (28 + pw_tile_w - 1) / pw_tile_w,//layer_num_of_ifm_tiles_w;
                (28 + pw_tile_h - 1) / pw_tile_h,//layer_num_of_ofm_tiles_h;
                (28 + pw_tile_w - 1) / pw_tile_w,//layer_num_of_ofm_tiles_w;
                (192 * pw_conv_parallelism_out ) / weights_group_items,//layer_num_of_weight_groups_for_one_pass;
                608,//layer_weights_offset;
                0,//layer_weights_offset_on_chip;
                52864,//dw_ifms_cumulative_width_offset;
                1,//bool write_to_result_or_channels;
                0,//bool write_to_tmp;
                0,//bool fused_with_add;
                -128,//fms_dt layer_ifms_zero_point;
                0.0235294122248888,//,fms_dt layer_ofms_scale;
                -128,//fms_dt layer_ofms_zero_point;
                1,//rec_scales_dt add_layer_scale_reciprocal;
                0,//biases_dt add_layer_zero_point;
                1,//scales_dt skip_connection_other_layer_scale;
                0//biases_dt skip_connection_other_layer_zero_point;
                };
//****************************
 const int layer_22_pw_num_fils = 32;
const int layer_22_pw_depth = 192;
const int layer_22_pw_filter_dim = 1;
 const int layer_22_pw_ifm_width = 28;
 //****************************
const layer_specs layer_22_pw_specs = {
                22,//layer_index;
                PW_CONV,//conv_layer_type;; 
                32,//layer_num_fils 
                1,//strides;
                1,//filter_size;
                0,//padding_left;
                0,//padding_right;
                0,//padding_top;
                0,//padding_bottom;
                192,//layer_depth;
                28,//layer_ifm_height;
                28,//layer_ifm_width;
                28,//layer_ofm_height;
                28,//layer_ofm_width;
                0,//layer_activation;
                (192 + pw_tile_d - 1) / pw_tile_d,//layer_num_of_tiles_in_d;
                (32 + pw_conv_parallelism_out - 1) / pw_conv_parallelism_out,//layer_num_of_tiles_out_d;
                (28 + pw_tile_h - 1) / pw_tile_h,//layer_num_of_ifm_tiles_h;
                (28 + pw_tile_w - 1) / pw_tile_w,//layer_num_of_ifm_tiles_w;
                (28 + pw_tile_h - 1) / pw_tile_h,//layer_num_of_ofm_tiles_h;
                (28 + pw_tile_w - 1) / pw_tile_w,//layer_num_of_ofm_tiles_w;
                (192 * pw_conv_parallelism_out ) / weights_group_items,//layer_num_of_weight_groups_for_one_pass;
                33408 / weights_group_items,//layer_weights_offset;
                37760,//layer_weights_offset_on_chip;
                0,//dw_ifms_cumulative_width_offset;
                1,//bool write_to_result_or_channels;
                0,//bool write_to_tmp;
                1,//bool fused_with_add;
                -128,//fms_dt layer_ifms_zero_point;
                0.2900388240814209,//,fms_dt layer_ofms_scale;
                -4,//fms_dt layer_ofms_zero_point;
                2.6695711158574493,//rec_scales_dt add_layer_scale_reciprocal;
                -17,//biases_dt add_layer_zero_point;
                0.2636336088180542,//scales_dt skip_connection_other_layer_scale;
                7//biases_dt skip_connection_other_layer_zero_point;
                };
//****************************
 const int layer_24_pw_num_fils = 192;
const int layer_24_pw_depth = 32;
const int layer_24_pw_filter_dim = 1;
 const int layer_24_pw_ifm_width = 28;
 //****************************
const layer_specs layer_24_pw_specs = {
                24,//layer_index;
                PW_CONV,//conv_layer_type;; 
                192,//layer_num_fils 
                1,//strides;
                1,//filter_size;
                0,//padding_left;
                0,//padding_right;
                0,//padding_top;
                0,//padding_bottom;
                32,//layer_depth;
                28,//layer_ifm_height;
                28,//layer_ifm_width;
                28,//layer_ofm_height;
                28,//layer_ofm_width;
                RELU6,//layer_activation;
                (32 + pw_tile_d - 1) / pw_tile_d,//layer_num_of_tiles_in_d;
                (192 + pw_conv_parallelism_out - 1) / pw_conv_parallelism_out,//layer_num_of_tiles_out_d;
                (28 + pw_tile_h - 1) / pw_tile_h,//layer_num_of_ifm_tiles_h;
                (28 + pw_tile_w - 1) / pw_tile_w,//layer_num_of_ifm_tiles_w;
                (28 + pw_tile_h - 1) / pw_tile_h,//layer_num_of_ofm_tiles_h;
                (28 + pw_tile_w - 1) / pw_tile_w,//layer_num_of_ofm_tiles_w;
                (32 * pw_conv_parallelism_out ) / weights_group_items,//layer_num_of_weight_groups_for_one_pass;
                39552 / weights_group_items,//layer_weights_offset;
                43904,//layer_weights_offset_on_chip;
                0,//dw_ifms_cumulative_width_offset;
                1,//bool write_to_result_or_channels;
                0,//bool write_to_tmp;
                0,//bool fused_with_add;
                -17,//fms_dt layer_ifms_zero_point;
                0.0235294122248888,//,fms_dt layer_ofms_scale;
                -128,//fms_dt layer_ofms_zero_point;
                1,//rec_scales_dt add_layer_scale_reciprocal;
                0,//biases_dt add_layer_zero_point;
                1,//scales_dt skip_connection_other_layer_scale;
                0//biases_dt skip_connection_other_layer_zero_point;
                };
//****************************
 const int layer_26_dw_num_fils = 192;
const int layer_26_dw_depth = 192;
const int layer_26_dw_filter_dim = 3;
 const int layer_26_dw_ifm_width = 28;
 //****************************
const layer_specs layer_26_dw_specs = {
                26,//layer_index;
                DW_CONV,//conv_layer_type;; 
                192,//layer_num_fils 
                2,//strides;
                3,//filter_size;
                0,//padding_left;
                1,//padding_right;
                0,//padding_top;
                1,//padding_bottom;
                192,//layer_depth;
                28,//layer_ifm_height;
                28,//layer_ifm_width;
                14,//layer_ofm_height;
                14,//layer_ofm_width;
                RELU6,//layer_activation;
                (192 + pw_tile_d - 1) / pw_tile_d,//layer_num_of_tiles_in_d;
                (192 + pw_conv_parallelism_out - 1) / pw_conv_parallelism_out,//layer_num_of_tiles_out_d;
                (28 + pw_tile_h - 1) / pw_tile_h,//layer_num_of_ifm_tiles_h;
                (28 + pw_tile_w - 1) / pw_tile_w,//layer_num_of_ifm_tiles_w;
                (14 + pw_tile_h - 1) / pw_tile_h,//layer_num_of_ofm_tiles_h;
                (14 + pw_tile_w - 1) / pw_tile_w,//layer_num_of_ofm_tiles_w;
                (192 * pw_conv_parallelism_out ) / weights_group_items,//layer_num_of_weight_groups_for_one_pass;
                800,//layer_weights_offset;
                0,//layer_weights_offset_on_chip;
                63616,//dw_ifms_cumulative_width_offset;
                1,//bool write_to_result_or_channels;
                0,//bool write_to_tmp;
                0,//bool fused_with_add;
                -128,//fms_dt layer_ifms_zero_point;
                0.0235294122248888,//,fms_dt layer_ofms_scale;
                -128,//fms_dt layer_ofms_zero_point;
                1,//rec_scales_dt add_layer_scale_reciprocal;
                0,//biases_dt add_layer_zero_point;
                1,//scales_dt skip_connection_other_layer_scale;
                0//biases_dt skip_connection_other_layer_zero_point;
                };
//****************************
 const int layer_27_pw_num_fils = 64;
const int layer_27_pw_depth = 192;
const int layer_27_pw_filter_dim = 1;
 const int layer_27_pw_ifm_width = 14;
 //****************************
const layer_specs layer_27_pw_specs = {
                27,//layer_index;
                PW_CONV,//conv_layer_type;; 
                64,//layer_num_fils 
                1,//strides;
                1,//filter_size;
                0,//padding_left;
                0,//padding_right;
                0,//padding_top;
                0,//padding_bottom;
                192,//layer_depth;
                14,//layer_ifm_height;
                14,//layer_ifm_width;
                14,//layer_ofm_height;
                14,//layer_ofm_width;
                0,//layer_activation;
                (192 + pw_tile_d - 1) / pw_tile_d,//layer_num_of_tiles_in_d;
                (64 + pw_conv_parallelism_out - 1) / pw_conv_parallelism_out,//layer_num_of_tiles_out_d;
                (14 + pw_tile_h - 1) / pw_tile_h,//layer_num_of_ifm_tiles_h;
                (14 + pw_tile_w - 1) / pw_tile_w,//layer_num_of_ifm_tiles_w;
                (14 + pw_tile_h - 1) / pw_tile_h,//layer_num_of_ofm_tiles_h;
                (14 + pw_tile_w - 1) / pw_tile_w,//layer_num_of_ofm_tiles_w;
                (192 * pw_conv_parallelism_out ) / weights_group_items,//layer_num_of_weight_groups_for_one_pass;
                45696 / weights_group_items,//layer_weights_offset;
                50048,//layer_weights_offset_on_chip;
                0,//dw_ifms_cumulative_width_offset;
                1,//bool write_to_result_or_channels;
                1,//bool write_to_tmp;
                0,//bool fused_with_add;
                -128,//fms_dt layer_ifms_zero_point;
                0.20964163541793823,//,fms_dt layer_ofms_scale;
                -6,//fms_dt layer_ofms_zero_point;
                1,//rec_scales_dt add_layer_scale_reciprocal;
                0,//biases_dt add_layer_zero_point;
                1,//scales_dt skip_connection_other_layer_scale;
                0//biases_dt skip_connection_other_layer_zero_point;
                };
//****************************
 const int layer_28_pw_num_fils = 384;
const int layer_28_pw_depth = 64;
const int layer_28_pw_filter_dim = 1;
 const int layer_28_pw_ifm_width = 14;
 //****************************
const layer_specs layer_28_pw_specs = {
                28,//layer_index;
                PW_CONV,//conv_layer_type;; 
                384,//layer_num_fils 
                1,//strides;
                1,//filter_size;
                0,//padding_left;
                0,//padding_right;
                0,//padding_top;
                0,//padding_bottom;
                64,//layer_depth;
                14,//layer_ifm_height;
                14,//layer_ifm_width;
                14,//layer_ofm_height;
                14,//layer_ofm_width;
                RELU6,//layer_activation;
                (64 + pw_tile_d - 1) / pw_tile_d,//layer_num_of_tiles_in_d;
                (384 + pw_conv_parallelism_out - 1) / pw_conv_parallelism_out,//layer_num_of_tiles_out_d;
                (14 + pw_tile_h - 1) / pw_tile_h,//layer_num_of_ifm_tiles_h;
                (14 + pw_tile_w - 1) / pw_tile_w,//layer_num_of_ifm_tiles_w;
                (14 + pw_tile_h - 1) / pw_tile_h,//layer_num_of_ofm_tiles_h;
                (14 + pw_tile_w - 1) / pw_tile_w,//layer_num_of_ofm_tiles_w;
                (64 * pw_conv_parallelism_out ) / weights_group_items,//layer_num_of_weight_groups_for_one_pass;
                57984 / weights_group_items,//layer_weights_offset;
                62336,//layer_weights_offset_on_chip;
                0,//dw_ifms_cumulative_width_offset;
                1,//bool write_to_result_or_channels;
                0,//bool write_to_tmp;
                0,//bool fused_with_add;
                -6,//fms_dt layer_ifms_zero_point;
                0.0235294122248888,//,fms_dt layer_ofms_scale;
                -128,//fms_dt layer_ofms_zero_point;
                1,//rec_scales_dt add_layer_scale_reciprocal;
                0,//biases_dt add_layer_zero_point;
                1,//scales_dt skip_connection_other_layer_scale;
                0//biases_dt skip_connection_other_layer_zero_point;
                };
//****************************
 const int layer_29_dw_num_fils = 384;
const int layer_29_dw_depth = 384;
const int layer_29_dw_filter_dim = 3;
 const int layer_29_dw_ifm_width = 14;
 //****************************
const layer_specs layer_29_dw_specs = {
                29,//layer_index;
                DW_CONV,//conv_layer_type;; 
                384,//layer_num_fils 
                1,//strides;
                3,//filter_size;
                1,//padding_left;
                1,//padding_right;
                1,//padding_top;
                1,//padding_bottom;
                384,//layer_depth;
                14,//layer_ifm_height;
                14,//layer_ifm_width;
                14,//layer_ofm_height;
                14,//layer_ofm_width;
                RELU6,//layer_activation;
                (384 + pw_tile_d - 1) / pw_tile_d,//layer_num_of_tiles_in_d;
                (384 + pw_conv_parallelism_out - 1) / pw_conv_parallelism_out,//layer_num_of_tiles_out_d;
                (14 + pw_tile_h - 1) / pw_tile_h,//layer_num_of_ifm_tiles_h;
                (14 + pw_tile_w - 1) / pw_tile_w,//layer_num_of_ifm_tiles_w;
                (14 + pw_tile_h - 1) / pw_tile_h,//layer_num_of_ofm_tiles_h;
                (14 + pw_tile_w - 1) / pw_tile_w,//layer_num_of_ofm_tiles_w;
                (384 * pw_conv_parallelism_out ) / weights_group_items,//layer_num_of_weight_groups_for_one_pass;
                992,//layer_weights_offset;
                0,//layer_weights_offset_on_chip;
                68992,//dw_ifms_cumulative_width_offset;
                1,//bool write_to_result_or_channels;
                0,//bool write_to_tmp;
                0,//bool fused_with_add;
                -128,//fms_dt layer_ifms_zero_point;
                0.0235294122248888,//,fms_dt layer_ofms_scale;
                -128,//fms_dt layer_ofms_zero_point;
                1,//rec_scales_dt add_layer_scale_reciprocal;
                0,//biases_dt add_layer_zero_point;
                1,//scales_dt skip_connection_other_layer_scale;
                0//biases_dt skip_connection_other_layer_zero_point;
                };
//****************************
 const int layer_30_pw_num_fils = 64;
const int layer_30_pw_depth = 384;
const int layer_30_pw_filter_dim = 1;
 const int layer_30_pw_ifm_width = 14;
 //****************************
const layer_specs layer_30_pw_specs = {
                30,//layer_index;
                PW_CONV,//conv_layer_type;; 
                64,//layer_num_fils 
                1,//strides;
                1,//filter_size;
                0,//padding_left;
                0,//padding_right;
                0,//padding_top;
                0,//padding_bottom;
                384,//layer_depth;
                14,//layer_ifm_height;
                14,//layer_ifm_width;
                14,//layer_ofm_height;
                14,//layer_ofm_width;
                0,//layer_activation;
                (384 + pw_tile_d - 1) / pw_tile_d,//layer_num_of_tiles_in_d;
                (64 + pw_conv_parallelism_out - 1) / pw_conv_parallelism_out,//layer_num_of_tiles_out_d;
                (14 + pw_tile_h - 1) / pw_tile_h,//layer_num_of_ifm_tiles_h;
                (14 + pw_tile_w - 1) / pw_tile_w,//layer_num_of_ifm_tiles_w;
                (14 + pw_tile_h - 1) / pw_tile_h,//layer_num_of_ofm_tiles_h;
                (14 + pw_tile_w - 1) / pw_tile_w,//layer_num_of_ofm_tiles_w;
                (384 * pw_conv_parallelism_out ) / weights_group_items,//layer_num_of_weight_groups_for_one_pass;
                82560 / weights_group_items,//layer_weights_offset;
                86912,//layer_weights_offset_on_chip;
                0,//dw_ifms_cumulative_width_offset;
                1,//bool write_to_result_or_channels;
                1,//bool write_to_tmp;
                1,//bool fused_with_add;
                -128,//fms_dt layer_ifms_zero_point;
                0.1939420849084854,//,fms_dt layer_ofms_scale;
                -15,//fms_dt layer_ofms_zero_point;
                4.909893694722064,//rec_scales_dt add_layer_scale_reciprocal;
                -8,//biases_dt add_layer_zero_point;
                0.20964163541793823,//scales_dt skip_connection_other_layer_scale;
                -6//biases_dt skip_connection_other_layer_zero_point;
                };
//****************************
 const int layer_32_pw_num_fils = 384;
const int layer_32_pw_depth = 64;
const int layer_32_pw_filter_dim = 1;
 const int layer_32_pw_ifm_width = 14;
 //****************************
const layer_specs layer_32_pw_specs = {
                32,//layer_index;
                PW_CONV,//conv_layer_type;; 
                384,//layer_num_fils 
                1,//strides;
                1,//filter_size;
                0,//padding_left;
                0,//padding_right;
                0,//padding_top;
                0,//padding_bottom;
                64,//layer_depth;
                14,//layer_ifm_height;
                14,//layer_ifm_width;
                14,//layer_ofm_height;
                14,//layer_ofm_width;
                RELU6,//layer_activation;
                (64 + pw_tile_d - 1) / pw_tile_d,//layer_num_of_tiles_in_d;
                (384 + pw_conv_parallelism_out - 1) / pw_conv_parallelism_out,//layer_num_of_tiles_out_d;
                (14 + pw_tile_h - 1) / pw_tile_h,//layer_num_of_ifm_tiles_h;
                (14 + pw_tile_w - 1) / pw_tile_w,//layer_num_of_ifm_tiles_w;
                (14 + pw_tile_h - 1) / pw_tile_h,//layer_num_of_ofm_tiles_h;
                (14 + pw_tile_w - 1) / pw_tile_w,//layer_num_of_ofm_tiles_w;
                (64 * pw_conv_parallelism_out ) / weights_group_items,//layer_num_of_weight_groups_for_one_pass;
                107136 / weights_group_items,//layer_weights_offset;
                111488,//layer_weights_offset_on_chip;
                0,//dw_ifms_cumulative_width_offset;
                1,//bool write_to_result_or_channels;
                0,//bool write_to_tmp;
                0,//bool fused_with_add;
                -8,//fms_dt layer_ifms_zero_point;
                0.0235294122248888,//,fms_dt layer_ofms_scale;
                -128,//fms_dt layer_ofms_zero_point;
                1,//rec_scales_dt add_layer_scale_reciprocal;
                0,//biases_dt add_layer_zero_point;
                1,//scales_dt skip_connection_other_layer_scale;
                0//biases_dt skip_connection_other_layer_zero_point;
                };
//****************************
 const int layer_33_dw_num_fils = 384;
const int layer_33_dw_depth = 384;
const int layer_33_dw_filter_dim = 3;
 const int layer_33_dw_ifm_width = 14;
 //****************************
const layer_specs layer_33_dw_specs = {
                33,//layer_index;
                DW_CONV,//conv_layer_type;; 
                384,//layer_num_fils 
                1,//strides;
                3,//filter_size;
                1,//padding_left;
                1,//padding_right;
                1,//padding_top;
                1,//padding_bottom;
                384,//layer_depth;
                14,//layer_ifm_height;
                14,//layer_ifm_width;
                14,//layer_ofm_height;
                14,//layer_ofm_width;
                RELU6,//layer_activation;
                (384 + pw_tile_d - 1) / pw_tile_d,//layer_num_of_tiles_in_d;
                (384 + pw_conv_parallelism_out - 1) / pw_conv_parallelism_out,//layer_num_of_tiles_out_d;
                (14 + pw_tile_h - 1) / pw_tile_h,//layer_num_of_ifm_tiles_h;
                (14 + pw_tile_w - 1) / pw_tile_w,//layer_num_of_ifm_tiles_w;
                (14 + pw_tile_h - 1) / pw_tile_h,//layer_num_of_ofm_tiles_h;
                (14 + pw_tile_w - 1) / pw_tile_w,//layer_num_of_ofm_tiles_w;
                (384 * pw_conv_parallelism_out ) / weights_group_items,//layer_num_of_weight_groups_for_one_pass;
                1376,//layer_weights_offset;
                0,//layer_weights_offset_on_chip;
                79744,//dw_ifms_cumulative_width_offset;
                1,//bool write_to_result_or_channels;
                0,//bool write_to_tmp;
                0,//bool fused_with_add;
                -128,//fms_dt layer_ifms_zero_point;
                0.0235294122248888,//,fms_dt layer_ofms_scale;
                -128,//fms_dt layer_ofms_zero_point;
                1,//rec_scales_dt add_layer_scale_reciprocal;
                0,//biases_dt add_layer_zero_point;
                1,//scales_dt skip_connection_other_layer_scale;
                0//biases_dt skip_connection_other_layer_zero_point;
                };
//****************************
 const int layer_34_pw_num_fils = 64;
const int layer_34_pw_depth = 384;
const int layer_34_pw_filter_dim = 1;
 const int layer_34_pw_ifm_width = 14;
 //****************************
const layer_specs layer_34_pw_specs = {
                34,//layer_index;
                PW_CONV,//conv_layer_type;; 
                64,//layer_num_fils 
                1,//strides;
                1,//filter_size;
                0,//padding_left;
                0,//padding_right;
                0,//padding_top;
                0,//padding_bottom;
                384,//layer_depth;
                14,//layer_ifm_height;
                14,//layer_ifm_width;
                14,//layer_ofm_height;
                14,//layer_ofm_width;
                0,//layer_activation;
                (384 + pw_tile_d - 1) / pw_tile_d,//layer_num_of_tiles_in_d;
                (64 + pw_conv_parallelism_out - 1) / pw_conv_parallelism_out,//layer_num_of_tiles_out_d;
                (14 + pw_tile_h - 1) / pw_tile_h,//layer_num_of_ifm_tiles_h;
                (14 + pw_tile_w - 1) / pw_tile_w,//layer_num_of_ifm_tiles_w;
                (14 + pw_tile_h - 1) / pw_tile_h,//layer_num_of_ofm_tiles_h;
                (14 + pw_tile_w - 1) / pw_tile_w,//layer_num_of_ofm_tiles_w;
                (384 * pw_conv_parallelism_out ) / weights_group_items,//layer_num_of_weight_groups_for_one_pass;
                131712 / weights_group_items,//layer_weights_offset;
                136064,//layer_weights_offset_on_chip;
                0,//dw_ifms_cumulative_width_offset;
                1,//bool write_to_result_or_channels;
                1,//bool write_to_tmp;
                1,//bool fused_with_add;
                -128,//fms_dt layer_ifms_zero_point;
                0.15376076102256775,//,fms_dt layer_ofms_scale;
                -16,//fms_dt layer_ofms_zero_point;
                4.591107699591028,//rec_scales_dt add_layer_scale_reciprocal;
                -14,//biases_dt add_layer_zero_point;
                0.20367039740085602,//scales_dt skip_connection_other_layer_scale;
                -8//biases_dt skip_connection_other_layer_zero_point;
                };
//****************************
 const int layer_36_pw_num_fils = 384;
const int layer_36_pw_depth = 64;
const int layer_36_pw_filter_dim = 1;
 const int layer_36_pw_ifm_width = 14;
 //****************************
const layer_specs layer_36_pw_specs = {
                36,//layer_index;
                PW_CONV,//conv_layer_type;; 
                384,//layer_num_fils 
                1,//strides;
                1,//filter_size;
                0,//padding_left;
                0,//padding_right;
                0,//padding_top;
                0,//padding_bottom;
                64,//layer_depth;
                14,//layer_ifm_height;
                14,//layer_ifm_width;
                14,//layer_ofm_height;
                14,//layer_ofm_width;
                RELU6,//layer_activation;
                (64 + pw_tile_d - 1) / pw_tile_d,//layer_num_of_tiles_in_d;
                (384 + pw_conv_parallelism_out - 1) / pw_conv_parallelism_out,//layer_num_of_tiles_out_d;
                (14 + pw_tile_h - 1) / pw_tile_h,//layer_num_of_ifm_tiles_h;
                (14 + pw_tile_w - 1) / pw_tile_w,//layer_num_of_ifm_tiles_w;
                (14 + pw_tile_h - 1) / pw_tile_h,//layer_num_of_ofm_tiles_h;
                (14 + pw_tile_w - 1) / pw_tile_w,//layer_num_of_ofm_tiles_w;
                (64 * pw_conv_parallelism_out ) / weights_group_items,//layer_num_of_weight_groups_for_one_pass;
                156288 / weights_group_items,//layer_weights_offset;
                160640,//layer_weights_offset_on_chip;
                0,//dw_ifms_cumulative_width_offset;
                1,//bool write_to_result_or_channels;
                0,//bool write_to_tmp;
                0,//bool fused_with_add;
                -14,//fms_dt layer_ifms_zero_point;
                0.0235294122248888,//,fms_dt layer_ofms_scale;
                -128,//fms_dt layer_ofms_zero_point;
                1,//rec_scales_dt add_layer_scale_reciprocal;
                0,//biases_dt add_layer_zero_point;
                1,//scales_dt skip_connection_other_layer_scale;
                0//biases_dt skip_connection_other_layer_zero_point;
                };
//****************************
 const int layer_37_dw_num_fils = 384;
const int layer_37_dw_depth = 384;
const int layer_37_dw_filter_dim = 3;
 const int layer_37_dw_ifm_width = 14;
 //****************************
const layer_specs layer_37_dw_specs = {
                37,//layer_index;
                DW_CONV,//conv_layer_type;; 
                384,//layer_num_fils 
                1,//strides;
                3,//filter_size;
                1,//padding_left;
                1,//padding_right;
                1,//padding_top;
                1,//padding_bottom;
                384,//layer_depth;
                14,//layer_ifm_height;
                14,//layer_ifm_width;
                14,//layer_ofm_height;
                14,//layer_ofm_width;
                RELU6,//layer_activation;
                (384 + pw_tile_d - 1) / pw_tile_d,//layer_num_of_tiles_in_d;
                (384 + pw_conv_parallelism_out - 1) / pw_conv_parallelism_out,//layer_num_of_tiles_out_d;
                (14 + pw_tile_h - 1) / pw_tile_h,//layer_num_of_ifm_tiles_h;
                (14 + pw_tile_w - 1) / pw_tile_w,//layer_num_of_ifm_tiles_w;
                (14 + pw_tile_h - 1) / pw_tile_h,//layer_num_of_ofm_tiles_h;
                (14 + pw_tile_w - 1) / pw_tile_w,//layer_num_of_ofm_tiles_w;
                (384 * pw_conv_parallelism_out ) / weights_group_items,//layer_num_of_weight_groups_for_one_pass;
                1760,//layer_weights_offset;
                0,//layer_weights_offset_on_chip;
                90496,//dw_ifms_cumulative_width_offset;
                1,//bool write_to_result_or_channels;
                0,//bool write_to_tmp;
                0,//bool fused_with_add;
                -128,//fms_dt layer_ifms_zero_point;
                0.0235294122248888,//,fms_dt layer_ofms_scale;
                -128,//fms_dt layer_ofms_zero_point;
                1,//rec_scales_dt add_layer_scale_reciprocal;
                0,//biases_dt add_layer_zero_point;
                1,//scales_dt skip_connection_other_layer_scale;
                0//biases_dt skip_connection_other_layer_zero_point;
                };
//****************************
 const int layer_38_pw_num_fils = 64;
const int layer_38_pw_depth = 384;
const int layer_38_pw_filter_dim = 1;
 const int layer_38_pw_ifm_width = 14;
 //****************************
const layer_specs layer_38_pw_specs = {
                38,//layer_index;
                PW_CONV,//conv_layer_type;; 
                64,//layer_num_fils 
                1,//strides;
                1,//filter_size;
                0,//padding_left;
                0,//padding_right;
                0,//padding_top;
                0,//padding_bottom;
                384,//layer_depth;
                14,//layer_ifm_height;
                14,//layer_ifm_width;
                14,//layer_ofm_height;
                14,//layer_ofm_width;
                0,//layer_activation;
                (384 + pw_tile_d - 1) / pw_tile_d,//layer_num_of_tiles_in_d;
                (64 + pw_conv_parallelism_out - 1) / pw_conv_parallelism_out,//layer_num_of_tiles_out_d;
                (14 + pw_tile_h - 1) / pw_tile_h,//layer_num_of_ifm_tiles_h;
                (14 + pw_tile_w - 1) / pw_tile_w,//layer_num_of_ifm_tiles_w;
                (14 + pw_tile_h - 1) / pw_tile_h,//layer_num_of_ofm_tiles_h;
                (14 + pw_tile_w - 1) / pw_tile_w,//layer_num_of_ofm_tiles_w;
                (384 * pw_conv_parallelism_out ) / weights_group_items,//layer_num_of_weight_groups_for_one_pass;
                180864 / weights_group_items,//layer_weights_offset;
                185216,//layer_weights_offset_on_chip;
                0,//dw_ifms_cumulative_width_offset;
                1,//bool write_to_result_or_channels;
                0,//bool write_to_tmp;
                1,//bool fused_with_add;
                -128,//fms_dt layer_ifms_zero_point;
                0.17933277785778046,//,fms_dt layer_ofms_scale;
                -14,//fms_dt layer_ofms_zero_point;
                4.23036255280951,//rec_scales_dt add_layer_scale_reciprocal;
                -7,//biases_dt add_layer_zero_point;
                0.21781235933303833,//scales_dt skip_connection_other_layer_scale;
                -14//biases_dt skip_connection_other_layer_zero_point;
                };
//****************************
 const int layer_40_pw_num_fils = 384;
const int layer_40_pw_depth = 64;
const int layer_40_pw_filter_dim = 1;
 const int layer_40_pw_ifm_width = 14;
 //****************************
const layer_specs layer_40_pw_specs = {
                40,//layer_index;
                PW_CONV,//conv_layer_type;; 
                384,//layer_num_fils 
                1,//strides;
                1,//filter_size;
                0,//padding_left;
                0,//padding_right;
                0,//padding_top;
                0,//padding_bottom;
                64,//layer_depth;
                14,//layer_ifm_height;
                14,//layer_ifm_width;
                14,//layer_ofm_height;
                14,//layer_ofm_width;
                RELU6,//layer_activation;
                (64 + pw_tile_d - 1) / pw_tile_d,//layer_num_of_tiles_in_d;
                (384 + pw_conv_parallelism_out - 1) / pw_conv_parallelism_out,//layer_num_of_tiles_out_d;
                (14 + pw_tile_h - 1) / pw_tile_h,//layer_num_of_ifm_tiles_h;
                (14 + pw_tile_w - 1) / pw_tile_w,//layer_num_of_ifm_tiles_w;
                (14 + pw_tile_h - 1) / pw_tile_h,//layer_num_of_ofm_tiles_h;
                (14 + pw_tile_w - 1) / pw_tile_w,//layer_num_of_ofm_tiles_w;
                (64 * pw_conv_parallelism_out ) / weights_group_items,//layer_num_of_weight_groups_for_one_pass;
                205440 / weights_group_items,//layer_weights_offset;
                209792,//layer_weights_offset_on_chip;
                0,//dw_ifms_cumulative_width_offset;
                1,//bool write_to_result_or_channels;
                0,//bool write_to_tmp;
                0,//bool fused_with_add;
                -7,//fms_dt layer_ifms_zero_point;
                0.0235294122248888,//,fms_dt layer_ofms_scale;
                -128,//fms_dt layer_ofms_zero_point;
                1,//rec_scales_dt add_layer_scale_reciprocal;
                0,//biases_dt add_layer_zero_point;
                1,//scales_dt skip_connection_other_layer_scale;
                0//biases_dt skip_connection_other_layer_zero_point;
                };
//****************************
 const int layer_41_dw_num_fils = 384;
const int layer_41_dw_depth = 384;
const int layer_41_dw_filter_dim = 3;
 const int layer_41_dw_ifm_width = 14;
 //****************************
const layer_specs layer_41_dw_specs = {
                41,//layer_index;
                DW_CONV,//conv_layer_type;; 
                384,//layer_num_fils 
                1,//strides;
                3,//filter_size;
                1,//padding_left;
                1,//padding_right;
                1,//padding_top;
                1,//padding_bottom;
                384,//layer_depth;
                14,//layer_ifm_height;
                14,//layer_ifm_width;
                14,//layer_ofm_height;
                14,//layer_ofm_width;
                RELU6,//layer_activation;
                (384 + pw_tile_d - 1) / pw_tile_d,//layer_num_of_tiles_in_d;
                (384 + pw_conv_parallelism_out - 1) / pw_conv_parallelism_out,//layer_num_of_tiles_out_d;
                (14 + pw_tile_h - 1) / pw_tile_h,//layer_num_of_ifm_tiles_h;
                (14 + pw_tile_w - 1) / pw_tile_w,//layer_num_of_ifm_tiles_w;
                (14 + pw_tile_h - 1) / pw_tile_h,//layer_num_of_ofm_tiles_h;
                (14 + pw_tile_w - 1) / pw_tile_w,//layer_num_of_ofm_tiles_w;
                (384 * pw_conv_parallelism_out ) / weights_group_items,//layer_num_of_weight_groups_for_one_pass;
                2144,//layer_weights_offset;
                0,//layer_weights_offset_on_chip;
                101248,//dw_ifms_cumulative_width_offset;
                1,//bool write_to_result_or_channels;
                0,//bool write_to_tmp;
                0,//bool fused_with_add;
                -128,//fms_dt layer_ifms_zero_point;
                0.0235294122248888,//,fms_dt layer_ofms_scale;
                -128,//fms_dt layer_ofms_zero_point;
                1,//rec_scales_dt add_layer_scale_reciprocal;
                0,//biases_dt add_layer_zero_point;
                1,//scales_dt skip_connection_other_layer_scale;
                0//biases_dt skip_connection_other_layer_zero_point;
                };
//****************************
 const int layer_42_pw_num_fils = 96;
const int layer_42_pw_depth = 384;
const int layer_42_pw_filter_dim = 1;
 const int layer_42_pw_ifm_width = 14;
 //****************************
const layer_specs layer_42_pw_specs = {
                42,//layer_index;
                PW_CONV,//conv_layer_type;; 
                96,//layer_num_fils 
                1,//strides;
                1,//filter_size;
                0,//padding_left;
                0,//padding_right;
                0,//padding_top;
                0,//padding_bottom;
                384,//layer_depth;
                14,//layer_ifm_height;
                14,//layer_ifm_width;
                14,//layer_ofm_height;
                14,//layer_ofm_width;
                0,//layer_activation;
                (384 + pw_tile_d - 1) / pw_tile_d,//layer_num_of_tiles_in_d;
                (96 + pw_conv_parallelism_out - 1) / pw_conv_parallelism_out,//layer_num_of_tiles_out_d;
                (14 + pw_tile_h - 1) / pw_tile_h,//layer_num_of_ifm_tiles_h;
                (14 + pw_tile_w - 1) / pw_tile_w,//layer_num_of_ifm_tiles_w;
                (14 + pw_tile_h - 1) / pw_tile_h,//layer_num_of_ofm_tiles_h;
                (14 + pw_tile_w - 1) / pw_tile_w,//layer_num_of_ofm_tiles_w;
                (384 * pw_conv_parallelism_out ) / weights_group_items,//layer_num_of_weight_groups_for_one_pass;
                230016 / weights_group_items,//layer_weights_offset;
                234368,//layer_weights_offset_on_chip;
                0,//dw_ifms_cumulative_width_offset;
                1,//bool write_to_result_or_channels;
                1,//bool write_to_tmp;
                0,//bool fused_with_add;
                -128,//fms_dt layer_ifms_zero_point;
                0.1666376143693924,//,fms_dt layer_ofms_scale;
                13,//fms_dt layer_ofms_zero_point;
                1,//rec_scales_dt add_layer_scale_reciprocal;
                0,//biases_dt add_layer_zero_point;
                1,//scales_dt skip_connection_other_layer_scale;
                0//biases_dt skip_connection_other_layer_zero_point;
                };
//****************************
 const int layer_43_pw_num_fils = 576;
const int layer_43_pw_depth = 96;
const int layer_43_pw_filter_dim = 1;
 const int layer_43_pw_ifm_width = 14;
 //****************************
const layer_specs layer_43_pw_specs = {
                43,//layer_index;
                PW_CONV,//conv_layer_type;; 
                576,//layer_num_fils 
                1,//strides;
                1,//filter_size;
                0,//padding_left;
                0,//padding_right;
                0,//padding_top;
                0,//padding_bottom;
                96,//layer_depth;
                14,//layer_ifm_height;
                14,//layer_ifm_width;
                14,//layer_ofm_height;
                14,//layer_ofm_width;
                RELU6,//layer_activation;
                (96 + pw_tile_d - 1) / pw_tile_d,//layer_num_of_tiles_in_d;
                (576 + pw_conv_parallelism_out - 1) / pw_conv_parallelism_out,//layer_num_of_tiles_out_d;
                (14 + pw_tile_h - 1) / pw_tile_h,//layer_num_of_ifm_tiles_h;
                (14 + pw_tile_w - 1) / pw_tile_w,//layer_num_of_ifm_tiles_w;
                (14 + pw_tile_h - 1) / pw_tile_h,//layer_num_of_ofm_tiles_h;
                (14 + pw_tile_w - 1) / pw_tile_w,//layer_num_of_ofm_tiles_w;
                (96 * pw_conv_parallelism_out ) / weights_group_items,//layer_num_of_weight_groups_for_one_pass;
                266880 / weights_group_items,//layer_weights_offset;
                271232,//layer_weights_offset_on_chip;
                0,//dw_ifms_cumulative_width_offset;
                1,//bool write_to_result_or_channels;
                0,//bool write_to_tmp;
                0,//bool fused_with_add;
                13,//fms_dt layer_ifms_zero_point;
                0.0235294122248888,//,fms_dt layer_ofms_scale;
                -128,//fms_dt layer_ofms_zero_point;
                1,//rec_scales_dt add_layer_scale_reciprocal;
                0,//biases_dt add_layer_zero_point;
                1,//scales_dt skip_connection_other_layer_scale;
                0//biases_dt skip_connection_other_layer_zero_point;
                };
//****************************
 const int layer_44_dw_num_fils = 576;
const int layer_44_dw_depth = 576;
const int layer_44_dw_filter_dim = 3;
 const int layer_44_dw_ifm_width = 14;
 //****************************
const layer_specs layer_44_dw_specs = {
                44,//layer_index;
                DW_CONV,//conv_layer_type;; 
                576,//layer_num_fils 
                1,//strides;
                3,//filter_size;
                1,//padding_left;
                1,//padding_right;
                1,//padding_top;
                1,//padding_bottom;
                576,//layer_depth;
                14,//layer_ifm_height;
                14,//layer_ifm_width;
                14,//layer_ofm_height;
                14,//layer_ofm_width;
                RELU6,//layer_activation;
                (576 + pw_tile_d - 1) / pw_tile_d,//layer_num_of_tiles_in_d;
                (576 + pw_conv_parallelism_out - 1) / pw_conv_parallelism_out,//layer_num_of_tiles_out_d;
                (14 + pw_tile_h - 1) / pw_tile_h,//layer_num_of_ifm_tiles_h;
                (14 + pw_tile_w - 1) / pw_tile_w,//layer_num_of_ifm_tiles_w;
                (14 + pw_tile_h - 1) / pw_tile_h,//layer_num_of_ofm_tiles_h;
                (14 + pw_tile_w - 1) / pw_tile_w,//layer_num_of_ofm_tiles_w;
                (576 * pw_conv_parallelism_out ) / weights_group_items,//layer_num_of_weight_groups_for_one_pass;
                2528,//layer_weights_offset;
                0,//layer_weights_offset_on_chip;
                112000,//dw_ifms_cumulative_width_offset;
                1,//bool write_to_result_or_channels;
                0,//bool write_to_tmp;
                0,//bool fused_with_add;
                -128,//fms_dt layer_ifms_zero_point;
                0.0235294122248888,//,fms_dt layer_ofms_scale;
                -128,//fms_dt layer_ofms_zero_point;
                1,//rec_scales_dt add_layer_scale_reciprocal;
                0,//biases_dt add_layer_zero_point;
                1,//scales_dt skip_connection_other_layer_scale;
                0//biases_dt skip_connection_other_layer_zero_point;
                };
//****************************
 const int layer_45_pw_num_fils = 96;
const int layer_45_pw_depth = 576;
const int layer_45_pw_filter_dim = 1;
 const int layer_45_pw_ifm_width = 14;
 //****************************
const layer_specs layer_45_pw_specs = {
                45,//layer_index;
                PW_CONV,//conv_layer_type;; 
                96,//layer_num_fils 
                1,//strides;
                1,//filter_size;
                0,//padding_left;
                0,//padding_right;
                0,//padding_top;
                0,//padding_bottom;
                576,//layer_depth;
                14,//layer_ifm_height;
                14,//layer_ifm_width;
                14,//layer_ofm_height;
                14,//layer_ofm_width;
                0,//layer_activation;
                (576 + pw_tile_d - 1) / pw_tile_d,//layer_num_of_tiles_in_d;
                (96 + pw_conv_parallelism_out - 1) / pw_conv_parallelism_out,//layer_num_of_tiles_out_d;
                (14 + pw_tile_h - 1) / pw_tile_h,//layer_num_of_ifm_tiles_h;
                (14 + pw_tile_w - 1) / pw_tile_w,//layer_num_of_ifm_tiles_w;
                (14 + pw_tile_h - 1) / pw_tile_h,//layer_num_of_ofm_tiles_h;
                (14 + pw_tile_w - 1) / pw_tile_w,//layer_num_of_ofm_tiles_w;
                (576 * pw_conv_parallelism_out ) / weights_group_items,//layer_num_of_weight_groups_for_one_pass;
                322176 / weights_group_items,//layer_weights_offset;
                326528,//layer_weights_offset_on_chip;
                0,//dw_ifms_cumulative_width_offset;
                1,//bool write_to_result_or_channels;
                1,//bool write_to_tmp;
                1,//bool fused_with_add;
                -128,//fms_dt layer_ifms_zero_point;
                0.142830491065979,//,fms_dt layer_ofms_scale;
                -11,//fms_dt layer_ofms_zero_point;
                5.718978723607331,//rec_scales_dt add_layer_scale_reciprocal;
                5,//biases_dt add_layer_zero_point;
                0.1666376143693924,//scales_dt skip_connection_other_layer_scale;
                13//biases_dt skip_connection_other_layer_zero_point;
                };
//****************************
 const int layer_47_pw_num_fils = 576;
const int layer_47_pw_depth = 96;
const int layer_47_pw_filter_dim = 1;
 const int layer_47_pw_ifm_width = 14;
 //****************************
const layer_specs layer_47_pw_specs = {
                47,//layer_index;
                PW_CONV,//conv_layer_type;; 
                576,//layer_num_fils 
                1,//strides;
                1,//filter_size;
                0,//padding_left;
                0,//padding_right;
                0,//padding_top;
                0,//padding_bottom;
                96,//layer_depth;
                14,//layer_ifm_height;
                14,//layer_ifm_width;
                14,//layer_ofm_height;
                14,//layer_ofm_width;
                RELU6,//layer_activation;
                (96 + pw_tile_d - 1) / pw_tile_d,//layer_num_of_tiles_in_d;
                (576 + pw_conv_parallelism_out - 1) / pw_conv_parallelism_out,//layer_num_of_tiles_out_d;
                (14 + pw_tile_h - 1) / pw_tile_h,//layer_num_of_ifm_tiles_h;
                (14 + pw_tile_w - 1) / pw_tile_w,//layer_num_of_ifm_tiles_w;
                (14 + pw_tile_h - 1) / pw_tile_h,//layer_num_of_ofm_tiles_h;
                (14 + pw_tile_w - 1) / pw_tile_w,//layer_num_of_ofm_tiles_w;
                (96 * pw_conv_parallelism_out ) / weights_group_items,//layer_num_of_weight_groups_for_one_pass;
                377472 / weights_group_items,//layer_weights_offset;
                381824,//layer_weights_offset_on_chip;
                0,//dw_ifms_cumulative_width_offset;
                1,//bool write_to_result_or_channels;
                0,//bool write_to_tmp;
                0,//bool fused_with_add;
                5,//fms_dt layer_ifms_zero_point;
                0.0235294122248888,//,fms_dt layer_ofms_scale;
                -128,//fms_dt layer_ofms_zero_point;
                1,//rec_scales_dt add_layer_scale_reciprocal;
                0,//biases_dt add_layer_zero_point;
                1,//scales_dt skip_connection_other_layer_scale;
                0//biases_dt skip_connection_other_layer_zero_point;
                };
//****************************
 const int layer_48_dw_num_fils = 576;
const int layer_48_dw_depth = 576;
const int layer_48_dw_filter_dim = 3;
 const int layer_48_dw_ifm_width = 14;
 //****************************
const layer_specs layer_48_dw_specs = {
                48,//layer_index;
                DW_CONV,//conv_layer_type;; 
                576,//layer_num_fils 
                1,//strides;
                3,//filter_size;
                1,//padding_left;
                1,//padding_right;
                1,//padding_top;
                1,//padding_bottom;
                576,//layer_depth;
                14,//layer_ifm_height;
                14,//layer_ifm_width;
                14,//layer_ofm_height;
                14,//layer_ofm_width;
                RELU6,//layer_activation;
                (576 + pw_tile_d - 1) / pw_tile_d,//layer_num_of_tiles_in_d;
                (576 + pw_conv_parallelism_out - 1) / pw_conv_parallelism_out,//layer_num_of_tiles_out_d;
                (14 + pw_tile_h - 1) / pw_tile_h,//layer_num_of_ifm_tiles_h;
                (14 + pw_tile_w - 1) / pw_tile_w,//layer_num_of_ifm_tiles_w;
                (14 + pw_tile_h - 1) / pw_tile_h,//layer_num_of_ofm_tiles_h;
                (14 + pw_tile_w - 1) / pw_tile_w,//layer_num_of_ofm_tiles_w;
                (576 * pw_conv_parallelism_out ) / weights_group_items,//layer_num_of_weight_groups_for_one_pass;
                3104,//layer_weights_offset;
                0,//layer_weights_offset_on_chip;
                128128,//dw_ifms_cumulative_width_offset;
                1,//bool write_to_result_or_channels;
                0,//bool write_to_tmp;
                0,//bool fused_with_add;
                -128,//fms_dt layer_ifms_zero_point;
                0.0235294122248888,//,fms_dt layer_ofms_scale;
                -128,//fms_dt layer_ofms_zero_point;
                1,//rec_scales_dt add_layer_scale_reciprocal;
                0,//biases_dt add_layer_zero_point;
                1,//scales_dt skip_connection_other_layer_scale;
                0//biases_dt skip_connection_other_layer_zero_point;
                };
//****************************
 const int layer_49_pw_num_fils = 96;
const int layer_49_pw_depth = 576;
const int layer_49_pw_filter_dim = 1;
 const int layer_49_pw_ifm_width = 14;
 //****************************
const layer_specs layer_49_pw_specs = {
                49,//layer_index;
                PW_CONV,//conv_layer_type;; 
                96,//layer_num_fils 
                1,//strides;
                1,//filter_size;
                0,//padding_left;
                0,//padding_right;
                0,//padding_top;
                0,//padding_bottom;
                576,//layer_depth;
                14,//layer_ifm_height;
                14,//layer_ifm_width;
                14,//layer_ofm_height;
                14,//layer_ofm_width;
                0,//layer_activation;
                (576 + pw_tile_d - 1) / pw_tile_d,//layer_num_of_tiles_in_d;
                (96 + pw_conv_parallelism_out - 1) / pw_conv_parallelism_out,//layer_num_of_tiles_out_d;
                (14 + pw_tile_h - 1) / pw_tile_h,//layer_num_of_ifm_tiles_h;
                (14 + pw_tile_w - 1) / pw_tile_w,//layer_num_of_ifm_tiles_w;
                (14 + pw_tile_h - 1) / pw_tile_h,//layer_num_of_ofm_tiles_h;
                (14 + pw_tile_w - 1) / pw_tile_w,//layer_num_of_ofm_tiles_w;
                (576 * pw_conv_parallelism_out ) / weights_group_items,//layer_num_of_weight_groups_for_one_pass;
                432768 / weights_group_items,//layer_weights_offset;
                437120,//layer_weights_offset_on_chip;
                0,//dw_ifms_cumulative_width_offset;
                1,//bool write_to_result_or_channels;
                0,//bool write_to_tmp;
                1,//bool fused_with_add;
                -128,//fms_dt layer_ifms_zero_point;
                0.18632979691028595,//,fms_dt layer_ofms_scale;
                8,//fms_dt layer_ofms_zero_point;
                3.619118024720866,//rec_scales_dt add_layer_scale_reciprocal;
                -2,//biases_dt add_layer_zero_point;
                0.17485639452934265,//scales_dt skip_connection_other_layer_scale;
                5//biases_dt skip_connection_other_layer_zero_point;
                };
//****************************
 const int layer_51_pw_num_fils = 576;
const int layer_51_pw_depth = 96;
const int layer_51_pw_filter_dim = 1;
 const int layer_51_pw_ifm_width = 14;
 //****************************
const layer_specs layer_51_pw_specs = {
                51,//layer_index;
                PW_CONV,//conv_layer_type;; 
                576,//layer_num_fils 
                1,//strides;
                1,//filter_size;
                0,//padding_left;
                0,//padding_right;
                0,//padding_top;
                0,//padding_bottom;
                96,//layer_depth;
                14,//layer_ifm_height;
                14,//layer_ifm_width;
                14,//layer_ofm_height;
                14,//layer_ofm_width;
                RELU6,//layer_activation;
                (96 + pw_tile_d - 1) / pw_tile_d,//layer_num_of_tiles_in_d;
                (576 + pw_conv_parallelism_out - 1) / pw_conv_parallelism_out,//layer_num_of_tiles_out_d;
                (14 + pw_tile_h - 1) / pw_tile_h,//layer_num_of_ifm_tiles_h;
                (14 + pw_tile_w - 1) / pw_tile_w,//layer_num_of_ifm_tiles_w;
                (14 + pw_tile_h - 1) / pw_tile_h,//layer_num_of_ofm_tiles_h;
                (14 + pw_tile_w - 1) / pw_tile_w,//layer_num_of_ofm_tiles_w;
                (96 * pw_conv_parallelism_out ) / weights_group_items,//layer_num_of_weight_groups_for_one_pass;
                488064 / weights_group_items,//layer_weights_offset;
                492416,//layer_weights_offset_on_chip;
                0,//dw_ifms_cumulative_width_offset;
                1,//bool write_to_result_or_channels;
                0,//bool write_to_tmp;
                0,//bool fused_with_add;
                -2,//fms_dt layer_ifms_zero_point;
                0.0235294122248888,//,fms_dt layer_ofms_scale;
                -128,//fms_dt layer_ofms_zero_point;
                1,//rec_scales_dt add_layer_scale_reciprocal;
                0,//biases_dt add_layer_zero_point;
                1,//scales_dt skip_connection_other_layer_scale;
                0//biases_dt skip_connection_other_layer_zero_point;
                };
//****************************
 const int layer_53_dw_num_fils = 576;
const int layer_53_dw_depth = 576;
const int layer_53_dw_filter_dim = 3;
 const int layer_53_dw_ifm_width = 14;
 //****************************
const layer_specs layer_53_dw_specs = {
                53,//layer_index;
                DW_CONV,//conv_layer_type;; 
                576,//layer_num_fils 
                2,//strides;
                3,//filter_size;
                0,//padding_left;
                1,//padding_right;
                0,//padding_top;
                1,//padding_bottom;
                576,//layer_depth;
                14,//layer_ifm_height;
                14,//layer_ifm_width;
                7,//layer_ofm_height;
                7,//layer_ofm_width;
                RELU6,//layer_activation;
                (576 + pw_tile_d - 1) / pw_tile_d,//layer_num_of_tiles_in_d;
                (576 + pw_conv_parallelism_out - 1) / pw_conv_parallelism_out,//layer_num_of_tiles_out_d;
                (14 + pw_tile_h - 1) / pw_tile_h,//layer_num_of_ifm_tiles_h;
                (14 + pw_tile_w - 1) / pw_tile_w,//layer_num_of_ifm_tiles_w;
                (7 + pw_tile_h - 1) / pw_tile_h,//layer_num_of_ofm_tiles_h;
                (7 + pw_tile_w - 1) / pw_tile_w,//layer_num_of_ofm_tiles_w;
                (576 * pw_conv_parallelism_out ) / weights_group_items,//layer_num_of_weight_groups_for_one_pass;
                3680,//layer_weights_offset;
                0,//layer_weights_offset_on_chip;
                144256,//dw_ifms_cumulative_width_offset;
                1,//bool write_to_result_or_channels;
                0,//bool write_to_tmp;
                0,//bool fused_with_add;
                -128,//fms_dt layer_ifms_zero_point;
                0.0235294122248888,//,fms_dt layer_ofms_scale;
                -128,//fms_dt layer_ofms_zero_point;
                1,//rec_scales_dt add_layer_scale_reciprocal;
                0,//biases_dt add_layer_zero_point;
                1,//scales_dt skip_connection_other_layer_scale;
                0//biases_dt skip_connection_other_layer_zero_point;
                };
//****************************
 const int layer_54_pw_num_fils = 160;
const int layer_54_pw_depth = 576;
const int layer_54_pw_filter_dim = 1;
 const int layer_54_pw_ifm_width = 7;
 //****************************
const layer_specs layer_54_pw_specs = {
                54,//layer_index;
                PW_CONV,//conv_layer_type;; 
                160,//layer_num_fils 
                1,//strides;
                1,//filter_size;
                0,//padding_left;
                0,//padding_right;
                0,//padding_top;
                0,//padding_bottom;
                576,//layer_depth;
                7,//layer_ifm_height;
                7,//layer_ifm_width;
                7,//layer_ofm_height;
                7,//layer_ofm_width;
                0,//layer_activation;
                (576 + pw_tile_d - 1) / pw_tile_d,//layer_num_of_tiles_in_d;
                (160 + pw_conv_parallelism_out - 1) / pw_conv_parallelism_out,//layer_num_of_tiles_out_d;
                (7 + pw_tile_h - 1) / pw_tile_h,//layer_num_of_ifm_tiles_h;
                (7 + pw_tile_w - 1) / pw_tile_w,//layer_num_of_ifm_tiles_w;
                (7 + pw_tile_h - 1) / pw_tile_h,//layer_num_of_ofm_tiles_h;
                (7 + pw_tile_w - 1) / pw_tile_w,//layer_num_of_ofm_tiles_w;
                (576 * pw_conv_parallelism_out ) / weights_group_items,//layer_num_of_weight_groups_for_one_pass;
                543360 / weights_group_items,//layer_weights_offset;
                547712,//layer_weights_offset_on_chip;
                0,//dw_ifms_cumulative_width_offset;
                1,//bool write_to_result_or_channels;
                1,//bool write_to_tmp;
                0,//bool fused_with_add;
                -128,//fms_dt layer_ifms_zero_point;
                0.13677388429641724,//,fms_dt layer_ofms_scale;
                11,//fms_dt layer_ofms_zero_point;
                1,//rec_scales_dt add_layer_scale_reciprocal;
                0,//biases_dt add_layer_zero_point;
                1,//scales_dt skip_connection_other_layer_scale;
                0//biases_dt skip_connection_other_layer_zero_point;
                };
//****************************
 const int layer_55_pw_num_fils = 960;
const int layer_55_pw_depth = 160;
const int layer_55_pw_filter_dim = 1;
 const int layer_55_pw_ifm_width = 7;
 //****************************
const layer_specs layer_55_pw_specs = {
                55,//layer_index;
                PW_CONV,//conv_layer_type;; 
                960,//layer_num_fils 
                1,//strides;
                1,//filter_size;
                0,//padding_left;
                0,//padding_right;
                0,//padding_top;
                0,//padding_bottom;
                160,//layer_depth;
                7,//layer_ifm_height;
                7,//layer_ifm_width;
                7,//layer_ofm_height;
                7,//layer_ofm_width;
                RELU6,//layer_activation;
                (160 + pw_tile_d - 1) / pw_tile_d,//layer_num_of_tiles_in_d;
                (960 + pw_conv_parallelism_out - 1) / pw_conv_parallelism_out,//layer_num_of_tiles_out_d;
                (7 + pw_tile_h - 1) / pw_tile_h,//layer_num_of_ifm_tiles_h;
                (7 + pw_tile_w - 1) / pw_tile_w,//layer_num_of_ifm_tiles_w;
                (7 + pw_tile_h - 1) / pw_tile_h,//layer_num_of_ofm_tiles_h;
                (7 + pw_tile_w - 1) / pw_tile_w,//layer_num_of_ofm_tiles_w;
                (160 * pw_conv_parallelism_out ) / weights_group_items,//layer_num_of_weight_groups_for_one_pass;
                635520 / weights_group_items,//layer_weights_offset;
                639872,//layer_weights_offset_on_chip;
                0,//dw_ifms_cumulative_width_offset;
                1,//bool write_to_result_or_channels;
                0,//bool write_to_tmp;
                0,//bool fused_with_add;
                11,//fms_dt layer_ifms_zero_point;
                0.0235294122248888,//,fms_dt layer_ofms_scale;
                -128,//fms_dt layer_ofms_zero_point;
                1,//rec_scales_dt add_layer_scale_reciprocal;
                0,//biases_dt add_layer_zero_point;
                1,//scales_dt skip_connection_other_layer_scale;
                0//biases_dt skip_connection_other_layer_zero_point;
                };
//****************************
 const int layer_56_dw_num_fils = 960;
const int layer_56_dw_depth = 960;
const int layer_56_dw_filter_dim = 3;
 const int layer_56_dw_ifm_width = 7;
 //****************************
const layer_specs layer_56_dw_specs = {
                56,//layer_index;
                DW_CONV,//conv_layer_type;; 
                960,//layer_num_fils 
                1,//strides;
                3,//filter_size;
                1,//padding_left;
                1,//padding_right;
                1,//padding_top;
                1,//padding_bottom;
                960,//layer_depth;
                7,//layer_ifm_height;
                7,//layer_ifm_width;
                7,//layer_ofm_height;
                7,//layer_ofm_width;
                RELU6,//layer_activation;
                (960 + pw_tile_d - 1) / pw_tile_d,//layer_num_of_tiles_in_d;
                (960 + pw_conv_parallelism_out - 1) / pw_conv_parallelism_out,//layer_num_of_tiles_out_d;
                (7 + pw_tile_h - 1) / pw_tile_h,//layer_num_of_ifm_tiles_h;
                (7 + pw_tile_w - 1) / pw_tile_w,//layer_num_of_ifm_tiles_w;
                (7 + pw_tile_h - 1) / pw_tile_h,//layer_num_of_ofm_tiles_h;
                (7 + pw_tile_w - 1) / pw_tile_w,//layer_num_of_ofm_tiles_w;
                (960 * pw_conv_parallelism_out ) / weights_group_items,//layer_num_of_weight_groups_for_one_pass;
                4256,//layer_weights_offset;
                0,//layer_weights_offset_on_chip;
                152320,//dw_ifms_cumulative_width_offset;
                1,//bool write_to_result_or_channels;
                0,//bool write_to_tmp;
                0,//bool fused_with_add;
                -128,//fms_dt layer_ifms_zero_point;
                0.0235294122248888,//,fms_dt layer_ofms_scale;
                -128,//fms_dt layer_ofms_zero_point;
                1,//rec_scales_dt add_layer_scale_reciprocal;
                0,//biases_dt add_layer_zero_point;
                1,//scales_dt skip_connection_other_layer_scale;
                0//biases_dt skip_connection_other_layer_zero_point;
                };
//****************************
 const int layer_57_pw_num_fils = 160;
const int layer_57_pw_depth = 960;
const int layer_57_pw_filter_dim = 1;
 const int layer_57_pw_ifm_width = 7;
 //****************************
const layer_specs layer_57_pw_specs = {
                57,//layer_index;
                PW_CONV,//conv_layer_type;; 
                160,//layer_num_fils 
                1,//strides;
                1,//filter_size;
                0,//padding_left;
                0,//padding_right;
                0,//padding_top;
                0,//padding_bottom;
                960,//layer_depth;
                7,//layer_ifm_height;
                7,//layer_ifm_width;
                7,//layer_ofm_height;
                7,//layer_ofm_width;
                0,//layer_activation;
                (960 + pw_tile_d - 1) / pw_tile_d,//layer_num_of_tiles_in_d;
                (160 + pw_conv_parallelism_out - 1) / pw_conv_parallelism_out,//layer_num_of_tiles_out_d;
                (7 + pw_tile_h - 1) / pw_tile_h,//layer_num_of_ifm_tiles_h;
                (7 + pw_tile_w - 1) / pw_tile_w,//layer_num_of_ifm_tiles_w;
                (7 + pw_tile_h - 1) / pw_tile_h,//layer_num_of_ofm_tiles_h;
                (7 + pw_tile_w - 1) / pw_tile_w,//layer_num_of_ofm_tiles_w;
                (960 * pw_conv_parallelism_out ) / weights_group_items,//layer_num_of_weight_groups_for_one_pass;
                789120 / weights_group_items,//layer_weights_offset;
                793472,//layer_weights_offset_on_chip;
                0,//dw_ifms_cumulative_width_offset;
                1,//bool write_to_result_or_channels;
                1,//bool write_to_tmp;
                1,//bool fused_with_add;
                -128,//fms_dt layer_ifms_zero_point;
                0.10249070078134537,//,fms_dt layer_ofms_scale;
                3,//fms_dt layer_ofms_zero_point;
                5.994435647514072,//rec_scales_dt add_layer_scale_reciprocal;
                -5,//biases_dt add_layer_zero_point;
                0.13677388429641724,//scales_dt skip_connection_other_layer_scale;
                11//biases_dt skip_connection_other_layer_zero_point;
                };
//****************************
 const int layer_59_pw_num_fils = 960;
const int layer_59_pw_depth = 160;
const int layer_59_pw_filter_dim = 1;
 const int layer_59_pw_ifm_width = 7;
 //****************************
const layer_specs layer_59_pw_specs = {
                59,//layer_index;
                PW_CONV,//conv_layer_type;; 
                960,//layer_num_fils 
                1,//strides;
                1,//filter_size;
                0,//padding_left;
                0,//padding_right;
                0,//padding_top;
                0,//padding_bottom;
                160,//layer_depth;
                7,//layer_ifm_height;
                7,//layer_ifm_width;
                7,//layer_ofm_height;
                7,//layer_ofm_width;
                RELU6,//layer_activation;
                (160 + pw_tile_d - 1) / pw_tile_d,//layer_num_of_tiles_in_d;
                (960 + pw_conv_parallelism_out - 1) / pw_conv_parallelism_out,//layer_num_of_tiles_out_d;
                (7 + pw_tile_h - 1) / pw_tile_h,//layer_num_of_ifm_tiles_h;
                (7 + pw_tile_w - 1) / pw_tile_w,//layer_num_of_ifm_tiles_w;
                (7 + pw_tile_h - 1) / pw_tile_h,//layer_num_of_ofm_tiles_h;
                (7 + pw_tile_w - 1) / pw_tile_w,//layer_num_of_ofm_tiles_w;
                (160 * pw_conv_parallelism_out ) / weights_group_items,//layer_num_of_weight_groups_for_one_pass;
                942720 / weights_group_items,//layer_weights_offset;
                947072,//layer_weights_offset_on_chip;
                0,//dw_ifms_cumulative_width_offset;
                1,//bool write_to_result_or_channels;
                0,//bool write_to_tmp;
                0,//bool fused_with_add;
                -5,//fms_dt layer_ifms_zero_point;
                0.0235294122248888,//,fms_dt layer_ofms_scale;
                -128,//fms_dt layer_ofms_zero_point;
                1,//rec_scales_dt add_layer_scale_reciprocal;
                0,//biases_dt add_layer_zero_point;
                1,//scales_dt skip_connection_other_layer_scale;
                0//biases_dt skip_connection_other_layer_zero_point;
                };
//****************************
 const int layer_60_dw_num_fils = 960;
const int layer_60_dw_depth = 960;
const int layer_60_dw_filter_dim = 3;
 const int layer_60_dw_ifm_width = 7;
 //****************************
const layer_specs layer_60_dw_specs = {
                60,//layer_index;
                DW_CONV,//conv_layer_type;; 
                960,//layer_num_fils 
                1,//strides;
                3,//filter_size;
                1,//padding_left;
                1,//padding_right;
                1,//padding_top;
                1,//padding_bottom;
                960,//layer_depth;
                7,//layer_ifm_height;
                7,//layer_ifm_width;
                7,//layer_ofm_height;
                7,//layer_ofm_width;
                RELU6,//layer_activation;
                (960 + pw_tile_d - 1) / pw_tile_d,//layer_num_of_tiles_in_d;
                (960 + pw_conv_parallelism_out - 1) / pw_conv_parallelism_out,//layer_num_of_tiles_out_d;
                (7 + pw_tile_h - 1) / pw_tile_h,//layer_num_of_ifm_tiles_h;
                (7 + pw_tile_w - 1) / pw_tile_w,//layer_num_of_ifm_tiles_w;
                (7 + pw_tile_h - 1) / pw_tile_h,//layer_num_of_ofm_tiles_h;
                (7 + pw_tile_w - 1) / pw_tile_w,//layer_num_of_ofm_tiles_w;
                (960 * pw_conv_parallelism_out ) / weights_group_items,//layer_num_of_weight_groups_for_one_pass;
                5216,//layer_weights_offset;
                0,//layer_weights_offset_on_chip;
                165760,//dw_ifms_cumulative_width_offset;
                1,//bool write_to_result_or_channels;
                0,//bool write_to_tmp;
                0,//bool fused_with_add;
                -128,//fms_dt layer_ifms_zero_point;
                0.0235294122248888,//,fms_dt layer_ofms_scale;
                -128,//fms_dt layer_ofms_zero_point;
                1,//rec_scales_dt add_layer_scale_reciprocal;
                0,//biases_dt add_layer_zero_point;
                1,//scales_dt skip_connection_other_layer_scale;
                0//biases_dt skip_connection_other_layer_zero_point;
                };
//****************************
 const int layer_61_pw_num_fils = 160;
const int layer_61_pw_depth = 960;
const int layer_61_pw_filter_dim = 1;
 const int layer_61_pw_ifm_width = 7;
 //****************************
const layer_specs layer_61_pw_specs = {
                61,//layer_index;
                PW_CONV,//conv_layer_type;; 
                160,//layer_num_fils 
                1,//strides;
                1,//filter_size;
                0,//padding_left;
                0,//padding_right;
                0,//padding_top;
                0,//padding_bottom;
                960,//layer_depth;
                7,//layer_ifm_height;
                7,//layer_ifm_width;
                7,//layer_ofm_height;
                7,//layer_ofm_width;
                0,//layer_activation;
                (960 + pw_tile_d - 1) / pw_tile_d,//layer_num_of_tiles_in_d;
                (160 + pw_conv_parallelism_out - 1) / pw_conv_parallelism_out,//layer_num_of_tiles_out_d;
                (7 + pw_tile_h - 1) / pw_tile_h,//layer_num_of_ifm_tiles_h;
                (7 + pw_tile_w - 1) / pw_tile_w,//layer_num_of_ifm_tiles_w;
                (7 + pw_tile_h - 1) / pw_tile_h,//layer_num_of_ofm_tiles_h;
                (7 + pw_tile_w - 1) / pw_tile_w,//layer_num_of_ofm_tiles_w;
                (960 * pw_conv_parallelism_out ) / weights_group_items,//layer_num_of_weight_groups_for_one_pass;
                1096320 / weights_group_items,//layer_weights_offset;
                1100672,//layer_weights_offset_on_chip;
                0,//dw_ifms_cumulative_width_offset;
                1,//bool write_to_result_or_channels;
                0,//bool write_to_tmp;
                1,//bool fused_with_add;
                -128,//fms_dt layer_ifms_zero_point;
                0.22082959115505219,//,fms_dt layer_ofms_scale;
                0,//fms_dt layer_ofms_zero_point;
                3.3522585543733454,//rec_scales_dt add_layer_scale_reciprocal;
                4,//biases_dt add_layer_zero_point;
                0.16682137548923492,//scales_dt skip_connection_other_layer_scale;
                -5//biases_dt skip_connection_other_layer_zero_point;
                };
//****************************
 const int layer_63_pw_num_fils = 960;
const int layer_63_pw_depth = 160;
const int layer_63_pw_filter_dim = 1;
 const int layer_63_pw_ifm_width = 7;
 //****************************
const layer_specs layer_63_pw_specs = {
                63,//layer_index;
                PW_CONV,//conv_layer_type;; 
                960,//layer_num_fils 
                1,//strides;
                1,//filter_size;
                0,//padding_left;
                0,//padding_right;
                0,//padding_top;
                0,//padding_bottom;
                160,//layer_depth;
                7,//layer_ifm_height;
                7,//layer_ifm_width;
                7,//layer_ofm_height;
                7,//layer_ofm_width;
                RELU6,//layer_activation;
                (160 + pw_tile_d - 1) / pw_tile_d,//layer_num_of_tiles_in_d;
                (960 + pw_conv_parallelism_out - 1) / pw_conv_parallelism_out,//layer_num_of_tiles_out_d;
                (7 + pw_tile_h - 1) / pw_tile_h,//layer_num_of_ifm_tiles_h;
                (7 + pw_tile_w - 1) / pw_tile_w,//layer_num_of_ifm_tiles_w;
                (7 + pw_tile_h - 1) / pw_tile_h,//layer_num_of_ofm_tiles_h;
                (7 + pw_tile_w - 1) / pw_tile_w,//layer_num_of_ofm_tiles_w;
                (160 * pw_conv_parallelism_out ) / weights_group_items,//layer_num_of_weight_groups_for_one_pass;
                1249920 / weights_group_items,//layer_weights_offset;
                1254272,//layer_weights_offset_on_chip;
                0,//dw_ifms_cumulative_width_offset;
                1,//bool write_to_result_or_channels;
                0,//bool write_to_tmp;
                0,//bool fused_with_add;
                4,//fms_dt layer_ifms_zero_point;
                0.0235294122248888,//,fms_dt layer_ofms_scale;
                -128,//fms_dt layer_ofms_zero_point;
                1,//rec_scales_dt add_layer_scale_reciprocal;
                0,//biases_dt add_layer_zero_point;
                1,//scales_dt skip_connection_other_layer_scale;
                0//biases_dt skip_connection_other_layer_zero_point;
                };
//****************************
 const int layer_64_dw_num_fils = 960;
const int layer_64_dw_depth = 960;
const int layer_64_dw_filter_dim = 3;
 const int layer_64_dw_ifm_width = 7;
 //****************************
const layer_specs layer_64_dw_specs = {
                64,//layer_index;
                DW_CONV,//conv_layer_type;; 
                960,//layer_num_fils 
                1,//strides;
                3,//filter_size;
                1,//padding_left;
                1,//padding_right;
                1,//padding_top;
                1,//padding_bottom;
                960,//layer_depth;
                7,//layer_ifm_height;
                7,//layer_ifm_width;
                7,//layer_ofm_height;
                7,//layer_ofm_width;
                RELU6,//layer_activation;
                (960 + pw_tile_d - 1) / pw_tile_d,//layer_num_of_tiles_in_d;
                (960 + pw_conv_parallelism_out - 1) / pw_conv_parallelism_out,//layer_num_of_tiles_out_d;
                (7 + pw_tile_h - 1) / pw_tile_h,//layer_num_of_ifm_tiles_h;
                (7 + pw_tile_w - 1) / pw_tile_w,//layer_num_of_ifm_tiles_w;
                (7 + pw_tile_h - 1) / pw_tile_h,//layer_num_of_ofm_tiles_h;
                (7 + pw_tile_w - 1) / pw_tile_w,//layer_num_of_ofm_tiles_w;
                (960 * pw_conv_parallelism_out ) / weights_group_items,//layer_num_of_weight_groups_for_one_pass;
                6176,//layer_weights_offset;
                0,//layer_weights_offset_on_chip;
                179200,//dw_ifms_cumulative_width_offset;
                1,//bool write_to_result_or_channels;
                0,//bool write_to_tmp;
                0,//bool fused_with_add;
                -128,//fms_dt layer_ifms_zero_point;
                0.0235294122248888,//,fms_dt layer_ofms_scale;
                -128,//fms_dt layer_ofms_zero_point;
                1,//rec_scales_dt add_layer_scale_reciprocal;
                0,//biases_dt add_layer_zero_point;
                1,//scales_dt skip_connection_other_layer_scale;
                0//biases_dt skip_connection_other_layer_zero_point;
                };
//****************************
 const int layer_65_pw_num_fils = 320;
const int layer_65_pw_depth = 960;
const int layer_65_pw_filter_dim = 1;
 const int layer_65_pw_ifm_width = 7;
 //****************************
const layer_specs layer_65_pw_specs = {
                65,//layer_index;
                PW_CONV,//conv_layer_type;; 
                320,//layer_num_fils 
                1,//strides;
                1,//filter_size;
                0,//padding_left;
                0,//padding_right;
                0,//padding_top;
                0,//padding_bottom;
                960,//layer_depth;
                7,//layer_ifm_height;
                7,//layer_ifm_width;
                7,//layer_ofm_height;
                7,//layer_ofm_width;
                0,//layer_activation;
                (960 + pw_tile_d - 1) / pw_tile_d,//layer_num_of_tiles_in_d;
                (320 + pw_conv_parallelism_out - 1) / pw_conv_parallelism_out,//layer_num_of_tiles_out_d;
                (7 + pw_tile_h - 1) / pw_tile_h,//layer_num_of_ifm_tiles_h;
                (7 + pw_tile_w - 1) / pw_tile_w,//layer_num_of_ifm_tiles_w;
                (7 + pw_tile_h - 1) / pw_tile_h,//layer_num_of_ofm_tiles_h;
                (7 + pw_tile_w - 1) / pw_tile_w,//layer_num_of_ofm_tiles_w;
                (960 * pw_conv_parallelism_out ) / weights_group_items,//layer_num_of_weight_groups_for_one_pass;
                1403520 / weights_group_items,//layer_weights_offset;
                1407872,//layer_weights_offset_on_chip;
                0,//dw_ifms_cumulative_width_offset;
                1,//bool write_to_result_or_channels;
                0,//bool write_to_tmp;
                0,//bool fused_with_add;
                -128,//fms_dt layer_ifms_zero_point;
                0.14633579552173615,//,fms_dt layer_ofms_scale;
                40,//fms_dt layer_ofms_zero_point;
                1,//rec_scales_dt add_layer_scale_reciprocal;
                0,//biases_dt add_layer_zero_point;
                1,//scales_dt skip_connection_other_layer_scale;
                0//biases_dt skip_connection_other_layer_zero_point;
                };
//****************************
 const int layer_66_pw_num_fils = 1280;
const int layer_66_pw_depth = 320;
const int layer_66_pw_filter_dim = 1;
 const int layer_66_pw_ifm_width = 7;
 //****************************
const layer_specs layer_66_pw_specs = {
                66,//layer_index;
                PW_CONV,//conv_layer_type;; 
                1280,//layer_num_fils 
                1,//strides;
                1,//filter_size;
                0,//padding_left;
                0,//padding_right;
                0,//padding_top;
                0,//padding_bottom;
                320,//layer_depth;
                7,//layer_ifm_height;
                7,//layer_ifm_width;
                7,//layer_ofm_height;
                7,//layer_ofm_width;
                RELU6,//layer_activation;
                (320 + pw_tile_d - 1) / pw_tile_d,//layer_num_of_tiles_in_d;
                (1280 + pw_conv_parallelism_out - 1) / pw_conv_parallelism_out,//layer_num_of_tiles_out_d;
                (7 + pw_tile_h - 1) / pw_tile_h,//layer_num_of_ifm_tiles_h;
                (7 + pw_tile_w - 1) / pw_tile_w,//layer_num_of_ifm_tiles_w;
                (7 + pw_tile_h - 1) / pw_tile_h,//layer_num_of_ofm_tiles_h;
                (7 + pw_tile_w - 1) / pw_tile_w,//layer_num_of_ofm_tiles_w;
                (320 * pw_conv_parallelism_out ) / weights_group_items,//layer_num_of_weight_groups_for_one_pass;
                1710720 / weights_group_items,//layer_weights_offset;
                1715072,//layer_weights_offset_on_chip;
                0,//dw_ifms_cumulative_width_offset;
                1,//bool write_to_result_or_channels;
                0,//bool write_to_tmp;
                0,//bool fused_with_add;
                40,//fms_dt layer_ifms_zero_point;
                0.0235294122248888,//,fms_dt layer_ofms_scale;
                -128,//fms_dt layer_ofms_zero_point;
                1,//rec_scales_dt add_layer_scale_reciprocal;
                0,//biases_dt add_layer_zero_point;
                1,//scales_dt skip_connection_other_layer_scale;
                0//biases_dt skip_connection_other_layer_zero_point;
                };
const pooling_layer_specs layer_67_avgpool_specs = {
                1280,//ifm_depth
                7,//ifm_height
                7,//ifm_width
                1.0620487151969,//const pooling_fused_scales_dt fused_scale; 
                -128,//const biases_dt ifms_zero_point;
                -128,//const biases_dt ofms_zero_point;
                };
const fc_layer_specs layer_68_fc_specs = {
                -128,//const fms_dt ifm_zero_point
                };
const layer_specs layer_1_s_specs = first_conv_layer_specs;


 static void get_layer_specs_from_index(int &layer_index, layer_specs &l_specs){
 switch(layer_index) {
case 1:
l_specs = first_conv_layer_specs;
break;
case 2:
l_specs = layer_2_dw_specs;
break;
case 3:
l_specs = layer_3_pw_specs;
break;
case 4:
l_specs = layer_4_pw_specs;
break;
case 6:
l_specs = layer_6_dw_specs;
break;
case 7:
l_specs = layer_7_pw_specs;
break;
case 8:
l_specs = layer_8_pw_specs;
break;
case 9:
l_specs = layer_9_dw_specs;
break;
case 10:
l_specs = layer_10_pw_specs;
break;
case 12:
l_specs = layer_12_pw_specs;
break;
case 14:
l_specs = layer_14_dw_specs;
break;
case 15:
l_specs = layer_15_pw_specs;
break;
case 16:
l_specs = layer_16_pw_specs;
break;
case 17:
l_specs = layer_17_dw_specs;
break;
case 18:
l_specs = layer_18_pw_specs;
break;
case 20:
l_specs = layer_20_pw_specs;
break;
case 21:
l_specs = layer_21_dw_specs;
break;
case 22:
l_specs = layer_22_pw_specs;
break;
case 24:
l_specs = layer_24_pw_specs;
break;
case 26:
l_specs = layer_26_dw_specs;
break;
case 27:
l_specs = layer_27_pw_specs;
break;
case 28:
l_specs = layer_28_pw_specs;
break;
case 29:
l_specs = layer_29_dw_specs;
break;
case 30:
l_specs = layer_30_pw_specs;
break;
case 32:
l_specs = layer_32_pw_specs;
break;
case 33:
l_specs = layer_33_dw_specs;
break;
case 34:
l_specs = layer_34_pw_specs;
break;
case 36:
l_specs = layer_36_pw_specs;
break;
case 37:
l_specs = layer_37_dw_specs;
break;
case 38:
l_specs = layer_38_pw_specs;
break;
case 40:
l_specs = layer_40_pw_specs;
break;
case 41:
l_specs = layer_41_dw_specs;
break;
case 42:
l_specs = layer_42_pw_specs;
break;
case 43:
l_specs = layer_43_pw_specs;
break;
case 44:
l_specs = layer_44_dw_specs;
break;
case 45:
l_specs = layer_45_pw_specs;
break;
case 47:
l_specs = layer_47_pw_specs;
break;
case 48:
l_specs = layer_48_dw_specs;
break;
case 49:
l_specs = layer_49_pw_specs;
break;
case 51:
l_specs = layer_51_pw_specs;
break;
case 53:
l_specs = layer_53_dw_specs;
break;
case 54:
l_specs = layer_54_pw_specs;
break;
case 55:
l_specs = layer_55_pw_specs;
break;
case 56:
l_specs = layer_56_dw_specs;
break;
case 57:
l_specs = layer_57_pw_specs;
break;
case 59:
l_specs = layer_59_pw_specs;
break;
case 60:
l_specs = layer_60_dw_specs;
break;
case 61:
l_specs = layer_61_pw_specs;
break;
case 63:
l_specs = layer_63_pw_specs;
break;
case 64:
l_specs = layer_64_dw_specs;
break;
case 65:
l_specs = layer_65_pw_specs;
break;
case 66:
l_specs = layer_66_pw_specs;
break;
default:
 layer_index = -1; 
}

}

#define LAYER_LIMIT 66

#endif
#endif
