#include "../../basic_defs/basic_defs_glue.h"
#ifndef LAYERS_SPECS
#define LAYERS_SPECS
//****************************
 const int layer_1_s_num_fils = 32 / alpha;
const int layer_1_s_depth = 3;
const int layer_1_s_filter_dim = 3;
 const int layer_1_s_ifm_width = 224;
 //****************************
const layer_specs layer_1_s_specs = {
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
                6,//layer_activation;
                (3 + pw_tile_d) / pw_tile_d,//layer_num_of_tiles_in_d;
                (32 + pw_conv_parallelism_out) / pw_conv_parallelism_out,//layer_num_of_tiles_out_d;
                (224 + pw_tile_h) / pw_tile_h,//layer_num_of_ifm_tiles_h;
                (224 + pw_tile_w) / pw_tile_w,//layer_num_of_ifm_tiles_w;
                (112 + pw_tile_h) / pw_tile_h,//layer_num_of_ofm_tiles_h;
                (112 + pw_tile_w) / pw_tile_w,//layer_num_of_ofm_tiles_w;
                3 * pw_conv_parallelism_out / weights_group_items,//layer_num_of_weight_groups_for_one_pass;
                0,//layer_weights_offset;
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
 const int layer_2_dw_num_fils = 32 / alpha;
const int layer_2_dw_depth = 32;
const int layer_2_dw_filter_dim = 3;
 const int layer_2_dw_ifm_width = 112;
 //****************************
const layer_specs layer_2_dw_specs = {
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
                6,//layer_activation;
                (32 + pw_tile_d) / pw_tile_d,//layer_num_of_tiles_in_d;
                (32 + pw_conv_parallelism_out) / pw_conv_parallelism_out,//layer_num_of_tiles_out_d;
                (112 + pw_tile_h) / pw_tile_h,//layer_num_of_ifm_tiles_h;
                (112 + pw_tile_w) / pw_tile_w,//layer_num_of_ifm_tiles_w;
                (112 + pw_tile_h) / pw_tile_h,//layer_num_of_ofm_tiles_h;
                (112 + pw_tile_w) / pw_tile_w,//layer_num_of_ofm_tiles_w;
                32 * pw_conv_parallelism_out / weights_group_items,//layer_num_of_weight_groups_for_one_pass;
                0,//layer_weights_offset;
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
 const int layer_3_pw_num_fils = 16 / alpha;
const int layer_3_pw_depth = 32;
const int layer_3_pw_filter_dim = 1;
 const int layer_3_pw_ifm_width = 112;
 //****************************
const layer_specs layer_3_pw_specs = {
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
                (32 + pw_tile_d) / pw_tile_d,//layer_num_of_tiles_in_d;
                (16 + pw_conv_parallelism_out) / pw_conv_parallelism_out,//layer_num_of_tiles_out_d;
                (112 + pw_tile_h) / pw_tile_h,//layer_num_of_ifm_tiles_h;
                (112 + pw_tile_w) / pw_tile_w,//layer_num_of_ifm_tiles_w;
                (112 + pw_tile_h) / pw_tile_h,//layer_num_of_ofm_tiles_h;
                (112 + pw_tile_w) / pw_tile_w,//layer_num_of_ofm_tiles_w;
                32 * pw_conv_parallelism_out / weights_group_items,//layer_num_of_weight_groups_for_one_pass;
                13,//layer_weights_offset;
                0,//bool write_to_tmp;
                0,//bool fused_with_add;
                -128,//fms_dt layer_ifms_zero_point;
                0.3762853741645813,//,fms_dt layer_ofms_scale;
                4,//fms_dt layer_ofms_zero_point;
                1,//rec_scales_dt add_layer_scale_reciprocal;
                0,//biases_dt add_layer_zero_point;
                1,//scales_dt skip_connection_other_layer_scale;
                0//biases_dt skip_connection_other_layer_zero_point;
                };
//****************************
 const int layer_4_pw_num_fils = 96 / alpha;
const int layer_4_pw_depth = 16;
const int layer_4_pw_filter_dim = 1;
 const int layer_4_pw_ifm_width = 112;
 //****************************
const layer_specs layer_4_pw_specs = {
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
                6,//layer_activation;
                (16 + pw_tile_d) / pw_tile_d,//layer_num_of_tiles_in_d;
                (96 + pw_conv_parallelism_out) / pw_conv_parallelism_out,//layer_num_of_tiles_out_d;
                (112 + pw_tile_h) / pw_tile_h,//layer_num_of_ifm_tiles_h;
                (112 + pw_tile_w) / pw_tile_w,//layer_num_of_ifm_tiles_w;
                (112 + pw_tile_h) / pw_tile_h,//layer_num_of_ofm_tiles_h;
                (112 + pw_tile_w) / pw_tile_w,//layer_num_of_ofm_tiles_w;
                16 * pw_conv_parallelism_out / weights_group_items,//layer_num_of_weight_groups_for_one_pass;
                21,//layer_weights_offset;
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
 const int layer_6_dw_num_fils = 96 / alpha;
const int layer_6_dw_depth = 96;
const int layer_6_dw_filter_dim = 3;
 const int layer_6_dw_ifm_width = 113;
 //****************************
const layer_specs layer_6_dw_specs = {
                DW_CONV,//conv_layer_type;; 
                96,//layer_num_fils 
                2,//strides;
                3,//filter_size;
                0,//padding_left;
                1,//padding_right;
                0,//padding_top;
                1,//padding_bottom;
                96,//layer_depth;
                113,//layer_ifm_height;
                113,//layer_ifm_width;
                56,//layer_ofm_height;
                56,//layer_ofm_width;
                6,//layer_activation;
                (96 + pw_tile_d) / pw_tile_d,//layer_num_of_tiles_in_d;
                (96 + pw_conv_parallelism_out) / pw_conv_parallelism_out,//layer_num_of_tiles_out_d;
                (113 + pw_tile_h) / pw_tile_h,//layer_num_of_ifm_tiles_h;
                (113 + pw_tile_w) / pw_tile_w,//layer_num_of_ifm_tiles_w;
                (56 + pw_tile_h) / pw_tile_h,//layer_num_of_ofm_tiles_h;
                (56 + pw_tile_w) / pw_tile_w,//layer_num_of_ofm_tiles_w;
                96 * pw_conv_parallelism_out / weights_group_items,//layer_num_of_weight_groups_for_one_pass;
                0,//layer_weights_offset;
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
 const int layer_7_pw_num_fils = 24 / alpha;
const int layer_7_pw_depth = 96;
const int layer_7_pw_filter_dim = 1;
 const int layer_7_pw_ifm_width = 56;
 //****************************
const layer_specs layer_7_pw_specs = {
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
                (96 + pw_tile_d) / pw_tile_d,//layer_num_of_tiles_in_d;
                (24 + pw_conv_parallelism_out) / pw_conv_parallelism_out,//layer_num_of_tiles_out_d;
                (56 + pw_tile_h) / pw_tile_h,//layer_num_of_ifm_tiles_h;
                (56 + pw_tile_w) / pw_tile_w,//layer_num_of_ifm_tiles_w;
                (56 + pw_tile_h) / pw_tile_h,//layer_num_of_ofm_tiles_h;
                (56 + pw_tile_w) / pw_tile_w,//layer_num_of_ofm_tiles_w;
                96 * pw_conv_parallelism_out / weights_group_items,//layer_num_of_weight_groups_for_one_pass;
                45,//layer_weights_offset;
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
 const int layer_8_pw_num_fils = 144 / alpha;
const int layer_8_pw_depth = 24;
const int layer_8_pw_filter_dim = 1;
 const int layer_8_pw_ifm_width = 56;
 //****************************
const layer_specs layer_8_pw_specs = {
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
                6,//layer_activation;
                (24 + pw_tile_d) / pw_tile_d,//layer_num_of_tiles_in_d;
                (144 + pw_conv_parallelism_out) / pw_conv_parallelism_out,//layer_num_of_tiles_out_d;
                (56 + pw_tile_h) / pw_tile_h,//layer_num_of_ifm_tiles_h;
                (56 + pw_tile_w) / pw_tile_w,//layer_num_of_ifm_tiles_w;
                (56 + pw_tile_h) / pw_tile_h,//layer_num_of_ofm_tiles_h;
                (56 + pw_tile_w) / pw_tile_w,//layer_num_of_ofm_tiles_w;
                24 * pw_conv_parallelism_out / weights_group_items,//layer_num_of_weight_groups_for_one_pass;
                81,//layer_weights_offset;
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
 const int layer_9_dw_num_fils = 144 / alpha;
const int layer_9_dw_depth = 144;
const int layer_9_dw_filter_dim = 3;
 const int layer_9_dw_ifm_width = 56;
 //****************************
const layer_specs layer_9_dw_specs = {
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
                6,//layer_activation;
                (144 + pw_tile_d) / pw_tile_d,//layer_num_of_tiles_in_d;
                (144 + pw_conv_parallelism_out) / pw_conv_parallelism_out,//layer_num_of_tiles_out_d;
                (56 + pw_tile_h) / pw_tile_h,//layer_num_of_ifm_tiles_h;
                (56 + pw_tile_w) / pw_tile_w,//layer_num_of_ifm_tiles_w;
                (56 + pw_tile_h) / pw_tile_h,//layer_num_of_ofm_tiles_h;
                (56 + pw_tile_w) / pw_tile_w,//layer_num_of_ofm_tiles_w;
                144 * pw_conv_parallelism_out / weights_group_items,//layer_num_of_weight_groups_for_one_pass;
                0,//layer_weights_offset;
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
 const int layer_10_pw_num_fils = 24 / alpha;
const int layer_10_pw_depth = 144;
const int layer_10_pw_filter_dim = 1;
 const int layer_10_pw_ifm_width = 56;
 //****************************
const layer_specs layer_10_pw_specs = {
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
                (144 + pw_tile_d) / pw_tile_d,//layer_num_of_tiles_in_d;
                (24 + pw_conv_parallelism_out) / pw_conv_parallelism_out,//layer_num_of_tiles_out_d;
                (56 + pw_tile_h) / pw_tile_h,//layer_num_of_ifm_tiles_h;
                (56 + pw_tile_w) / pw_tile_w,//layer_num_of_ifm_tiles_w;
                (56 + pw_tile_h) / pw_tile_h,//layer_num_of_ofm_tiles_h;
                (56 + pw_tile_w) / pw_tile_w,//layer_num_of_ofm_tiles_w;
                144 * pw_conv_parallelism_out / weights_group_items,//layer_num_of_weight_groups_for_one_pass;
                135,//layer_weights_offset;
                0,//bool write_to_tmp;
                1,//bool fused_with_add;
                -128,//fms_dt layer_ifms_zero_point;
                0.38619235157966614,//,fms_dt layer_ofms_scale;
                17,//fms_dt layer_ofms_zero_point;
                2.196542097829769,//rec_scales_dt add_layer_scale_reciprocal;
                6,//biases_dt add_layer_zero_point;
                0.30003005266189575,//scales_dt skip_connection_other_layer_scale;
                -11//biases_dt skip_connection_other_layer_zero_point;
                };
//****************************
 const int layer_12_pw_num_fils = 144 / alpha;
const int layer_12_pw_depth = 24;
const int layer_12_pw_filter_dim = 1;
 const int layer_12_pw_ifm_width = 56;
 //****************************
const layer_specs layer_12_pw_specs = {
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
                6,//layer_activation;
                (24 + pw_tile_d) / pw_tile_d,//layer_num_of_tiles_in_d;
                (144 + pw_conv_parallelism_out) / pw_conv_parallelism_out,//layer_num_of_tiles_out_d;
                (56 + pw_tile_h) / pw_tile_h,//layer_num_of_ifm_tiles_h;
                (56 + pw_tile_w) / pw_tile_w,//layer_num_of_ifm_tiles_w;
                (56 + pw_tile_h) / pw_tile_h,//layer_num_of_ofm_tiles_h;
                (56 + pw_tile_w) / pw_tile_w,//layer_num_of_ofm_tiles_w;
                24 * pw_conv_parallelism_out / weights_group_items,//layer_num_of_weight_groups_for_one_pass;
                189,//layer_weights_offset;
                0,//bool write_to_tmp;
                0,//bool fused_with_add;
                6,//fms_dt layer_ifms_zero_point;
                0.0235294122248888,//,fms_dt layer_ofms_scale;
                -128,//fms_dt layer_ofms_zero_point;
                1,//rec_scales_dt add_layer_scale_reciprocal;
                0,//biases_dt add_layer_zero_point;
                1,//scales_dt skip_connection_other_layer_scale;
                0//biases_dt skip_connection_other_layer_zero_point;
                };
//****************************
 const int layer_14_dw_num_fils = 144 / alpha;
const int layer_14_dw_depth = 144;
const int layer_14_dw_filter_dim = 3;
 const int layer_14_dw_ifm_width = 57;
 //****************************
const layer_specs layer_14_dw_specs = {
                DW_CONV,//conv_layer_type;; 
                144,//layer_num_fils 
                2,//strides;
                3,//filter_size;
                0,//padding_left;
                1,//padding_right;
                0,//padding_top;
                1,//padding_bottom;
                144,//layer_depth;
                57,//layer_ifm_height;
                57,//layer_ifm_width;
                28,//layer_ofm_height;
                28,//layer_ofm_width;
                6,//layer_activation;
                (144 + pw_tile_d) / pw_tile_d,//layer_num_of_tiles_in_d;
                (144 + pw_conv_parallelism_out) / pw_conv_parallelism_out,//layer_num_of_tiles_out_d;
                (57 + pw_tile_h) / pw_tile_h,//layer_num_of_ifm_tiles_h;
                (57 + pw_tile_w) / pw_tile_w,//layer_num_of_ifm_tiles_w;
                (28 + pw_tile_h) / pw_tile_h,//layer_num_of_ofm_tiles_h;
                (28 + pw_tile_w) / pw_tile_w,//layer_num_of_ofm_tiles_w;
                144 * pw_conv_parallelism_out / weights_group_items,//layer_num_of_weight_groups_for_one_pass;
                0,//layer_weights_offset;
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
 const int layer_15_pw_num_fils = 32 / alpha;
const int layer_15_pw_depth = 144;
const int layer_15_pw_filter_dim = 1;
 const int layer_15_pw_ifm_width = 28;
 //****************************
const layer_specs layer_15_pw_specs = {
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
                (144 + pw_tile_d) / pw_tile_d,//layer_num_of_tiles_in_d;
                (32 + pw_conv_parallelism_out) / pw_conv_parallelism_out,//layer_num_of_tiles_out_d;
                (28 + pw_tile_h) / pw_tile_h,//layer_num_of_ifm_tiles_h;
                (28 + pw_tile_w) / pw_tile_w,//layer_num_of_ifm_tiles_w;
                (28 + pw_tile_h) / pw_tile_h,//layer_num_of_ofm_tiles_h;
                (28 + pw_tile_w) / pw_tile_w,//layer_num_of_ofm_tiles_w;
                144 * pw_conv_parallelism_out / weights_group_items,//layer_num_of_weight_groups_for_one_pass;
                243,//layer_weights_offset;
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
 const int layer_16_pw_num_fils = 192 / alpha;
const int layer_16_pw_depth = 32;
const int layer_16_pw_filter_dim = 1;
 const int layer_16_pw_ifm_width = 28;
 //****************************
const layer_specs layer_16_pw_specs = {
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
                6,//layer_activation;
                (32 + pw_tile_d) / pw_tile_d,//layer_num_of_tiles_in_d;
                (192 + pw_conv_parallelism_out) / pw_conv_parallelism_out,//layer_num_of_tiles_out_d;
                (28 + pw_tile_h) / pw_tile_h,//layer_num_of_ifm_tiles_h;
                (28 + pw_tile_w) / pw_tile_w,//layer_num_of_ifm_tiles_w;
                (28 + pw_tile_h) / pw_tile_h,//layer_num_of_ofm_tiles_h;
                (28 + pw_tile_w) / pw_tile_w,//layer_num_of_ofm_tiles_w;
                32 * pw_conv_parallelism_out / weights_group_items,//layer_num_of_weight_groups_for_one_pass;
                315,//layer_weights_offset;
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
 const int layer_17_dw_num_fils = 192 / alpha;
const int layer_17_dw_depth = 192;
const int layer_17_dw_filter_dim = 3;
 const int layer_17_dw_ifm_width = 28;
 //****************************
const layer_specs layer_17_dw_specs = {
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
                6,//layer_activation;
                (192 + pw_tile_d) / pw_tile_d,//layer_num_of_tiles_in_d;
                (192 + pw_conv_parallelism_out) / pw_conv_parallelism_out,//layer_num_of_tiles_out_d;
                (28 + pw_tile_h) / pw_tile_h,//layer_num_of_ifm_tiles_h;
                (28 + pw_tile_w) / pw_tile_w,//layer_num_of_ifm_tiles_w;
                (28 + pw_tile_h) / pw_tile_h,//layer_num_of_ofm_tiles_h;
                (28 + pw_tile_w) / pw_tile_w,//layer_num_of_ofm_tiles_w;
                192 * pw_conv_parallelism_out / weights_group_items,//layer_num_of_weight_groups_for_one_pass;
                0,//layer_weights_offset;
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
 const int layer_18_pw_num_fils = 32 / alpha;
const int layer_18_pw_depth = 192;
const int layer_18_pw_filter_dim = 1;
 const int layer_18_pw_ifm_width = 28;
 //****************************
const layer_specs layer_18_pw_specs = {
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
                (192 + pw_tile_d) / pw_tile_d,//layer_num_of_tiles_in_d;
                (32 + pw_conv_parallelism_out) / pw_conv_parallelism_out,//layer_num_of_tiles_out_d;
                (28 + pw_tile_h) / pw_tile_h,//layer_num_of_ifm_tiles_h;
                (28 + pw_tile_w) / pw_tile_w,//layer_num_of_ifm_tiles_w;
                (28 + pw_tile_h) / pw_tile_h,//layer_num_of_ofm_tiles_h;
                (28 + pw_tile_w) / pw_tile_w,//layer_num_of_ofm_tiles_w;
                192 * pw_conv_parallelism_out / weights_group_items,//layer_num_of_weight_groups_for_one_pass;
                411,//layer_weights_offset;
                1,//bool write_to_tmp;
                1,//bool fused_with_add;
                -128,//fms_dt layer_ifms_zero_point;
                0.22161531448364258,//,fms_dt layer_ofms_scale;
                -8,//fms_dt layer_ofms_zero_point;
                3.937667042700383,//rec_scales_dt add_layer_scale_reciprocal;
                3,//biases_dt add_layer_zero_point;
                0.21887806057929993,//scales_dt skip_connection_other_layer_scale;
                -6//biases_dt skip_connection_other_layer_zero_point;
                };
//****************************
 const int layer_20_pw_num_fils = 192 / alpha;
const int layer_20_pw_depth = 32;
const int layer_20_pw_filter_dim = 1;
 const int layer_20_pw_ifm_width = 28;
 //****************************
const layer_specs layer_20_pw_specs = {
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
                6,//layer_activation;
                (32 + pw_tile_d) / pw_tile_d,//layer_num_of_tiles_in_d;
                (192 + pw_conv_parallelism_out) / pw_conv_parallelism_out,//layer_num_of_tiles_out_d;
                (28 + pw_tile_h) / pw_tile_h,//layer_num_of_ifm_tiles_h;
                (28 + pw_tile_w) / pw_tile_w,//layer_num_of_ifm_tiles_w;
                (28 + pw_tile_h) / pw_tile_h,//layer_num_of_ofm_tiles_h;
                (28 + pw_tile_w) / pw_tile_w,//layer_num_of_ofm_tiles_w;
                32 * pw_conv_parallelism_out / weights_group_items,//layer_num_of_weight_groups_for_one_pass;
                507,//layer_weights_offset;
                0,//bool write_to_tmp;
                0,//bool fused_with_add;
                3,//fms_dt layer_ifms_zero_point;
                0.0235294122248888,//,fms_dt layer_ofms_scale;
                -128,//fms_dt layer_ofms_zero_point;
                1,//rec_scales_dt add_layer_scale_reciprocal;
                0,//biases_dt add_layer_zero_point;
                1,//scales_dt skip_connection_other_layer_scale;
                0//biases_dt skip_connection_other_layer_zero_point;
                };
//****************************
 const int layer_21_dw_num_fils = 192 / alpha;
const int layer_21_dw_depth = 192;
const int layer_21_dw_filter_dim = 3;
 const int layer_21_dw_ifm_width = 28;
 //****************************
const layer_specs layer_21_dw_specs = {
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
                6,//layer_activation;
                (192 + pw_tile_d) / pw_tile_d,//layer_num_of_tiles_in_d;
                (192 + pw_conv_parallelism_out) / pw_conv_parallelism_out,//layer_num_of_tiles_out_d;
                (28 + pw_tile_h) / pw_tile_h,//layer_num_of_ifm_tiles_h;
                (28 + pw_tile_w) / pw_tile_w,//layer_num_of_ifm_tiles_w;
                (28 + pw_tile_h) / pw_tile_h,//layer_num_of_ofm_tiles_h;
                (28 + pw_tile_w) / pw_tile_w,//layer_num_of_ofm_tiles_w;
                192 * pw_conv_parallelism_out / weights_group_items,//layer_num_of_weight_groups_for_one_pass;
                0,//layer_weights_offset;
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
 const int layer_22_pw_num_fils = 32 / alpha;
const int layer_22_pw_depth = 192;
const int layer_22_pw_filter_dim = 1;
 const int layer_22_pw_ifm_width = 28;
 //****************************
const layer_specs layer_22_pw_specs = {
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
                (192 + pw_tile_d) / pw_tile_d,//layer_num_of_tiles_in_d;
                (32 + pw_conv_parallelism_out) / pw_conv_parallelism_out,//layer_num_of_tiles_out_d;
                (28 + pw_tile_h) / pw_tile_h,//layer_num_of_ifm_tiles_h;
                (28 + pw_tile_w) / pw_tile_w,//layer_num_of_ifm_tiles_w;
                (28 + pw_tile_h) / pw_tile_h,//layer_num_of_ofm_tiles_h;
                (28 + pw_tile_w) / pw_tile_w,//layer_num_of_ofm_tiles_w;
                192 * pw_conv_parallelism_out / weights_group_items,//layer_num_of_weight_groups_for_one_pass;
                603,//layer_weights_offset;
                0,//bool write_to_tmp;
                1,//bool fused_with_add;
                -128,//fms_dt layer_ifms_zero_point;
                0.26030266284942627,//,fms_dt layer_ofms_scale;
                -19,//fms_dt layer_ofms_zero_point;
                2.6695711158574493,//rec_scales_dt add_layer_scale_reciprocal;
                -17,//biases_dt add_layer_zero_point;
                0.25395748019218445,//scales_dt skip_connection_other_layer_scale;
                3//biases_dt skip_connection_other_layer_zero_point;
                };
//****************************
 const int layer_24_pw_num_fils = 192 / alpha;
const int layer_24_pw_depth = 32;
const int layer_24_pw_filter_dim = 1;
 const int layer_24_pw_ifm_width = 28;
 //****************************
const layer_specs layer_24_pw_specs = {
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
                6,//layer_activation;
                (32 + pw_tile_d) / pw_tile_d,//layer_num_of_tiles_in_d;
                (192 + pw_conv_parallelism_out) / pw_conv_parallelism_out,//layer_num_of_tiles_out_d;
                (28 + pw_tile_h) / pw_tile_h,//layer_num_of_ifm_tiles_h;
                (28 + pw_tile_w) / pw_tile_w,//layer_num_of_ifm_tiles_w;
                (28 + pw_tile_h) / pw_tile_h,//layer_num_of_ofm_tiles_h;
                (28 + pw_tile_w) / pw_tile_w,//layer_num_of_ofm_tiles_w;
                32 * pw_conv_parallelism_out / weights_group_items,//layer_num_of_weight_groups_for_one_pass;
                699,//layer_weights_offset;
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
 const int layer_26_dw_num_fils = 192 / alpha;
const int layer_26_dw_depth = 192;
const int layer_26_dw_filter_dim = 3;
 const int layer_26_dw_ifm_width = 29;
 //****************************
const layer_specs layer_26_dw_specs = {
                DW_CONV,//conv_layer_type;; 
                192,//layer_num_fils 
                2,//strides;
                3,//filter_size;
                0,//padding_left;
                1,//padding_right;
                0,//padding_top;
                1,//padding_bottom;
                192,//layer_depth;
                29,//layer_ifm_height;
                29,//layer_ifm_width;
                14,//layer_ofm_height;
                14,//layer_ofm_width;
                6,//layer_activation;
                (192 + pw_tile_d) / pw_tile_d,//layer_num_of_tiles_in_d;
                (192 + pw_conv_parallelism_out) / pw_conv_parallelism_out,//layer_num_of_tiles_out_d;
                (29 + pw_tile_h) / pw_tile_h,//layer_num_of_ifm_tiles_h;
                (29 + pw_tile_w) / pw_tile_w,//layer_num_of_ifm_tiles_w;
                (14 + pw_tile_h) / pw_tile_h,//layer_num_of_ofm_tiles_h;
                (14 + pw_tile_w) / pw_tile_w,//layer_num_of_ofm_tiles_w;
                192 * pw_conv_parallelism_out / weights_group_items,//layer_num_of_weight_groups_for_one_pass;
                0,//layer_weights_offset;
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
 const int layer_27_pw_num_fils = 64 / alpha;
const int layer_27_pw_depth = 192;
const int layer_27_pw_filter_dim = 1;
 const int layer_27_pw_ifm_width = 14;
 //****************************
const layer_specs layer_27_pw_specs = {
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
                (192 + pw_tile_d) / pw_tile_d,//layer_num_of_tiles_in_d;
                (64 + pw_conv_parallelism_out) / pw_conv_parallelism_out,//layer_num_of_tiles_out_d;
                (14 + pw_tile_h) / pw_tile_h,//layer_num_of_ifm_tiles_h;
                (14 + pw_tile_w) / pw_tile_w,//layer_num_of_ifm_tiles_w;
                (14 + pw_tile_h) / pw_tile_h,//layer_num_of_ofm_tiles_h;
                (14 + pw_tile_w) / pw_tile_w,//layer_num_of_ofm_tiles_w;
                192 * pw_conv_parallelism_out / weights_group_items,//layer_num_of_weight_groups_for_one_pass;
                795,//layer_weights_offset;
                1,//bool write_to_tmp;
                0,//bool fused_with_add;
                -128,//fms_dt layer_ifms_zero_point;
                0.17392461001873016,//,fms_dt layer_ofms_scale;
                -2,//fms_dt layer_ofms_zero_point;
                1,//rec_scales_dt add_layer_scale_reciprocal;
                0,//biases_dt add_layer_zero_point;
                1,//scales_dt skip_connection_other_layer_scale;
                0//biases_dt skip_connection_other_layer_zero_point;
                };
//****************************
 const int layer_28_pw_num_fils = 384 / alpha;
const int layer_28_pw_depth = 64;
const int layer_28_pw_filter_dim = 1;
 const int layer_28_pw_ifm_width = 14;
 //****************************
const layer_specs layer_28_pw_specs = {
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
                6,//layer_activation;
                (64 + pw_tile_d) / pw_tile_d,//layer_num_of_tiles_in_d;
                (384 + pw_conv_parallelism_out) / pw_conv_parallelism_out,//layer_num_of_tiles_out_d;
                (14 + pw_tile_h) / pw_tile_h,//layer_num_of_ifm_tiles_h;
                (14 + pw_tile_w) / pw_tile_w,//layer_num_of_ifm_tiles_w;
                (14 + pw_tile_h) / pw_tile_h,//layer_num_of_ofm_tiles_h;
                (14 + pw_tile_w) / pw_tile_w,//layer_num_of_ofm_tiles_w;
                64 * pw_conv_parallelism_out / weights_group_items,//layer_num_of_weight_groups_for_one_pass;
                987,//layer_weights_offset;
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
 const int layer_29_dw_num_fils = 384 / alpha;
const int layer_29_dw_depth = 384;
const int layer_29_dw_filter_dim = 3;
 const int layer_29_dw_ifm_width = 14;
 //****************************
const layer_specs layer_29_dw_specs = {
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
                6,//layer_activation;
                (384 + pw_tile_d) / pw_tile_d,//layer_num_of_tiles_in_d;
                (384 + pw_conv_parallelism_out) / pw_conv_parallelism_out,//layer_num_of_tiles_out_d;
                (14 + pw_tile_h) / pw_tile_h,//layer_num_of_ifm_tiles_h;
                (14 + pw_tile_w) / pw_tile_w,//layer_num_of_ifm_tiles_w;
                (14 + pw_tile_h) / pw_tile_h,//layer_num_of_ofm_tiles_h;
                (14 + pw_tile_w) / pw_tile_w,//layer_num_of_ofm_tiles_w;
                384 * pw_conv_parallelism_out / weights_group_items,//layer_num_of_weight_groups_for_one_pass;
                0,//layer_weights_offset;
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
 const int layer_30_pw_num_fils = 64 / alpha;
const int layer_30_pw_depth = 384;
const int layer_30_pw_filter_dim = 1;
 const int layer_30_pw_ifm_width = 14;
 //****************************
const layer_specs layer_30_pw_specs = {
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
                (384 + pw_tile_d) / pw_tile_d,//layer_num_of_tiles_in_d;
                (64 + pw_conv_parallelism_out) / pw_conv_parallelism_out,//layer_num_of_tiles_out_d;
                (14 + pw_tile_h) / pw_tile_h,//layer_num_of_ifm_tiles_h;
                (14 + pw_tile_w) / pw_tile_w,//layer_num_of_ifm_tiles_w;
                (14 + pw_tile_h) / pw_tile_h,//layer_num_of_ofm_tiles_h;
                (14 + pw_tile_w) / pw_tile_w,//layer_num_of_ofm_tiles_w;
                384 * pw_conv_parallelism_out / weights_group_items,//layer_num_of_weight_groups_for_one_pass;
                1371,//layer_weights_offset;
                1,//bool write_to_tmp;
                1,//bool fused_with_add;
                -128,//fms_dt layer_ifms_zero_point;
                0.1939420849084854,//,fms_dt layer_ofms_scale;
                -15,//fms_dt layer_ofms_zero_point;
                5.126175408149313,//rec_scales_dt add_layer_scale_reciprocal;
                -14,//biases_dt add_layer_zero_point;
                0.17392461001873016,//scales_dt skip_connection_other_layer_scale;
                -2//biases_dt skip_connection_other_layer_zero_point;
                };
//****************************
 const int layer_32_pw_num_fils = 384 / alpha;
const int layer_32_pw_depth = 64;
const int layer_32_pw_filter_dim = 1;
 const int layer_32_pw_ifm_width = 14;
 //****************************
const layer_specs layer_32_pw_specs = {
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
                6,//layer_activation;
                (64 + pw_tile_d) / pw_tile_d,//layer_num_of_tiles_in_d;
                (384 + pw_conv_parallelism_out) / pw_conv_parallelism_out,//layer_num_of_tiles_out_d;
                (14 + pw_tile_h) / pw_tile_h,//layer_num_of_ifm_tiles_h;
                (14 + pw_tile_w) / pw_tile_w,//layer_num_of_ifm_tiles_w;
                (14 + pw_tile_h) / pw_tile_h,//layer_num_of_ofm_tiles_h;
                (14 + pw_tile_w) / pw_tile_w,//layer_num_of_ofm_tiles_w;
                64 * pw_conv_parallelism_out / weights_group_items,//layer_num_of_weight_groups_for_one_pass;
                1755,//layer_weights_offset;
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
 const int layer_33_dw_num_fils = 384 / alpha;
const int layer_33_dw_depth = 384;
const int layer_33_dw_filter_dim = 3;
 const int layer_33_dw_ifm_width = 14;
 //****************************
const layer_specs layer_33_dw_specs = {
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
                6,//layer_activation;
                (384 + pw_tile_d) / pw_tile_d,//layer_num_of_tiles_in_d;
                (384 + pw_conv_parallelism_out) / pw_conv_parallelism_out,//layer_num_of_tiles_out_d;
                (14 + pw_tile_h) / pw_tile_h,//layer_num_of_ifm_tiles_h;
                (14 + pw_tile_w) / pw_tile_w,//layer_num_of_ifm_tiles_w;
                (14 + pw_tile_h) / pw_tile_h,//layer_num_of_ofm_tiles_h;
                (14 + pw_tile_w) / pw_tile_w,//layer_num_of_ofm_tiles_w;
                384 * pw_conv_parallelism_out / weights_group_items,//layer_num_of_weight_groups_for_one_pass;
                0,//layer_weights_offset;
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
 const int layer_34_pw_num_fils = 64 / alpha;
const int layer_34_pw_depth = 384;
const int layer_34_pw_filter_dim = 1;
 const int layer_34_pw_ifm_width = 14;
 //****************************
const layer_specs layer_34_pw_specs = {
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
                (384 + pw_tile_d) / pw_tile_d,//layer_num_of_tiles_in_d;
                (64 + pw_conv_parallelism_out) / pw_conv_parallelism_out,//layer_num_of_tiles_out_d;
                (14 + pw_tile_h) / pw_tile_h,//layer_num_of_ifm_tiles_h;
                (14 + pw_tile_w) / pw_tile_w,//layer_num_of_ifm_tiles_w;
                (14 + pw_tile_h) / pw_tile_h,//layer_num_of_ofm_tiles_h;
                (14 + pw_tile_w) / pw_tile_w,//layer_num_of_ofm_tiles_w;
                384 * pw_conv_parallelism_out / weights_group_items,//layer_num_of_weight_groups_for_one_pass;
                2139,//layer_weights_offset;
                1,//bool write_to_tmp;
                1,//bool fused_with_add;
                -128,//fms_dt layer_ifms_zero_point;
                0.15376076102256775,//,fms_dt layer_ofms_scale;
                -16,//fms_dt layer_ofms_zero_point;
                4.722017491673354,//rec_scales_dt add_layer_scale_reciprocal;
                -10,//biases_dt add_layer_zero_point;
                0.19507721066474915,//scales_dt skip_connection_other_layer_scale;
                -14//biases_dt skip_connection_other_layer_zero_point;
                };
//****************************
 const int layer_36_pw_num_fils = 384 / alpha;
const int layer_36_pw_depth = 64;
const int layer_36_pw_filter_dim = 1;
 const int layer_36_pw_ifm_width = 14;
 //****************************
const layer_specs layer_36_pw_specs = {
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
                6,//layer_activation;
                (64 + pw_tile_d) / pw_tile_d,//layer_num_of_tiles_in_d;
                (384 + pw_conv_parallelism_out) / pw_conv_parallelism_out,//layer_num_of_tiles_out_d;
                (14 + pw_tile_h) / pw_tile_h,//layer_num_of_ifm_tiles_h;
                (14 + pw_tile_w) / pw_tile_w,//layer_num_of_ifm_tiles_w;
                (14 + pw_tile_h) / pw_tile_h,//layer_num_of_ofm_tiles_h;
                (14 + pw_tile_w) / pw_tile_w,//layer_num_of_ofm_tiles_w;
                64 * pw_conv_parallelism_out / weights_group_items,//layer_num_of_weight_groups_for_one_pass;
                2523,//layer_weights_offset;
                0,//bool write_to_tmp;
                0,//bool fused_with_add;
                -10,//fms_dt layer_ifms_zero_point;
                0.0235294122248888,//,fms_dt layer_ofms_scale;
                -128,//fms_dt layer_ofms_zero_point;
                1,//rec_scales_dt add_layer_scale_reciprocal;
                0,//biases_dt add_layer_zero_point;
                1,//scales_dt skip_connection_other_layer_scale;
                0//biases_dt skip_connection_other_layer_zero_point;
                };
//****************************
 const int layer_37_dw_num_fils = 384 / alpha;
const int layer_37_dw_depth = 384;
const int layer_37_dw_filter_dim = 3;
 const int layer_37_dw_ifm_width = 14;
 //****************************
const layer_specs layer_37_dw_specs = {
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
                6,//layer_activation;
                (384 + pw_tile_d) / pw_tile_d,//layer_num_of_tiles_in_d;
                (384 + pw_conv_parallelism_out) / pw_conv_parallelism_out,//layer_num_of_tiles_out_d;
                (14 + pw_tile_h) / pw_tile_h,//layer_num_of_ifm_tiles_h;
                (14 + pw_tile_w) / pw_tile_w,//layer_num_of_ifm_tiles_w;
                (14 + pw_tile_h) / pw_tile_h,//layer_num_of_ofm_tiles_h;
                (14 + pw_tile_w) / pw_tile_w,//layer_num_of_ofm_tiles_w;
                384 * pw_conv_parallelism_out / weights_group_items,//layer_num_of_weight_groups_for_one_pass;
                0,//layer_weights_offset;
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
 const int layer_38_pw_num_fils = 64 / alpha;
const int layer_38_pw_depth = 384;
const int layer_38_pw_filter_dim = 1;
 const int layer_38_pw_ifm_width = 14;
 //****************************
const layer_specs layer_38_pw_specs = {
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
                (384 + pw_tile_d) / pw_tile_d,//layer_num_of_tiles_in_d;
                (64 + pw_conv_parallelism_out) / pw_conv_parallelism_out,//layer_num_of_tiles_out_d;
                (14 + pw_tile_h) / pw_tile_h,//layer_num_of_ifm_tiles_h;
                (14 + pw_tile_w) / pw_tile_w,//layer_num_of_ifm_tiles_w;
                (14 + pw_tile_h) / pw_tile_h,//layer_num_of_ofm_tiles_h;
                (14 + pw_tile_w) / pw_tile_w,//layer_num_of_ofm_tiles_w;
                384 * pw_conv_parallelism_out / weights_group_items,//layer_num_of_weight_groups_for_one_pass;
                2907,//layer_weights_offset;
                0,//bool write_to_tmp;
                1,//bool fused_with_add;
                -128,//fms_dt layer_ifms_zero_point;
                0.16587591171264648,//,fms_dt layer_ofms_scale;
                -12,//fms_dt layer_ofms_zero_point;
                4.23036255280951,//rec_scales_dt add_layer_scale_reciprocal;
                -7,//biases_dt add_layer_zero_point;
                0.21177388727664948,//scales_dt skip_connection_other_layer_scale;
                -10//biases_dt skip_connection_other_layer_zero_point;
                };
//****************************
 const int layer_40_pw_num_fils = 384 / alpha;
const int layer_40_pw_depth = 64;
const int layer_40_pw_filter_dim = 1;
 const int layer_40_pw_ifm_width = 14;
 //****************************
const layer_specs layer_40_pw_specs = {
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
                6,//layer_activation;
                (64 + pw_tile_d) / pw_tile_d,//layer_num_of_tiles_in_d;
                (384 + pw_conv_parallelism_out) / pw_conv_parallelism_out,//layer_num_of_tiles_out_d;
                (14 + pw_tile_h) / pw_tile_h,//layer_num_of_ifm_tiles_h;
                (14 + pw_tile_w) / pw_tile_w,//layer_num_of_ifm_tiles_w;
                (14 + pw_tile_h) / pw_tile_h,//layer_num_of_ofm_tiles_h;
                (14 + pw_tile_w) / pw_tile_w,//layer_num_of_ofm_tiles_w;
                64 * pw_conv_parallelism_out / weights_group_items,//layer_num_of_weight_groups_for_one_pass;
                3291,//layer_weights_offset;
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
 const int layer_41_dw_num_fils = 384 / alpha;
const int layer_41_dw_depth = 384;
const int layer_41_dw_filter_dim = 3;
 const int layer_41_dw_ifm_width = 14;
 //****************************
const layer_specs layer_41_dw_specs = {
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
                6,//layer_activation;
                (384 + pw_tile_d) / pw_tile_d,//layer_num_of_tiles_in_d;
                (384 + pw_conv_parallelism_out) / pw_conv_parallelism_out,//layer_num_of_tiles_out_d;
                (14 + pw_tile_h) / pw_tile_h,//layer_num_of_ifm_tiles_h;
                (14 + pw_tile_w) / pw_tile_w,//layer_num_of_ifm_tiles_w;
                (14 + pw_tile_h) / pw_tile_h,//layer_num_of_ofm_tiles_h;
                (14 + pw_tile_w) / pw_tile_w,//layer_num_of_ofm_tiles_w;
                384 * pw_conv_parallelism_out / weights_group_items,//layer_num_of_weight_groups_for_one_pass;
                0,//layer_weights_offset;
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
 const int layer_42_pw_num_fils = 96 / alpha;
const int layer_42_pw_depth = 384;
const int layer_42_pw_filter_dim = 1;
 const int layer_42_pw_ifm_width = 14;
 //****************************
const layer_specs layer_42_pw_specs = {
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
                (384 + pw_tile_d) / pw_tile_d,//layer_num_of_tiles_in_d;
                (96 + pw_conv_parallelism_out) / pw_conv_parallelism_out,//layer_num_of_tiles_out_d;
                (14 + pw_tile_h) / pw_tile_h,//layer_num_of_ifm_tiles_h;
                (14 + pw_tile_w) / pw_tile_w,//layer_num_of_ifm_tiles_w;
                (14 + pw_tile_h) / pw_tile_h,//layer_num_of_ofm_tiles_h;
                (14 + pw_tile_w) / pw_tile_w,//layer_num_of_ofm_tiles_w;
                384 * pw_conv_parallelism_out / weights_group_items,//layer_num_of_weight_groups_for_one_pass;
                3675,//layer_weights_offset;
                1,//bool write_to_tmp;
                0,//bool fused_with_add;
                -128,//fms_dt layer_ifms_zero_point;
                0.1662880778312683,//,fms_dt layer_ofms_scale;
                12,//fms_dt layer_ofms_zero_point;
                1,//rec_scales_dt add_layer_scale_reciprocal;
                0,//biases_dt add_layer_zero_point;
                1,//scales_dt skip_connection_other_layer_scale;
                0//biases_dt skip_connection_other_layer_zero_point;
                };
//****************************
 const int layer_43_pw_num_fils = 576 / alpha;
const int layer_43_pw_depth = 96;
const int layer_43_pw_filter_dim = 1;
 const int layer_43_pw_ifm_width = 14;
 //****************************
const layer_specs layer_43_pw_specs = {
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
                6,//layer_activation;
                (96 + pw_tile_d) / pw_tile_d,//layer_num_of_tiles_in_d;
                (576 + pw_conv_parallelism_out) / pw_conv_parallelism_out,//layer_num_of_tiles_out_d;
                (14 + pw_tile_h) / pw_tile_h,//layer_num_of_ifm_tiles_h;
                (14 + pw_tile_w) / pw_tile_w,//layer_num_of_ifm_tiles_w;
                (14 + pw_tile_h) / pw_tile_h,//layer_num_of_ofm_tiles_h;
                (14 + pw_tile_w) / pw_tile_w,//layer_num_of_ofm_tiles_w;
                96 * pw_conv_parallelism_out / weights_group_items,//layer_num_of_weight_groups_for_one_pass;
                4251,//layer_weights_offset;
                0,//bool write_to_tmp;
                0,//bool fused_with_add;
                12,//fms_dt layer_ifms_zero_point;
                0.0235294122248888,//,fms_dt layer_ofms_scale;
                -128,//fms_dt layer_ofms_zero_point;
                1,//rec_scales_dt add_layer_scale_reciprocal;
                0,//biases_dt add_layer_zero_point;
                1,//scales_dt skip_connection_other_layer_scale;
                0//biases_dt skip_connection_other_layer_zero_point;
                };
//****************************
 const int layer_44_dw_num_fils = 576 / alpha;
const int layer_44_dw_depth = 576;
const int layer_44_dw_filter_dim = 3;
 const int layer_44_dw_ifm_width = 14;
 //****************************
const layer_specs layer_44_dw_specs = {
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
                6,//layer_activation;
                (576 + pw_tile_d) / pw_tile_d,//layer_num_of_tiles_in_d;
                (576 + pw_conv_parallelism_out) / pw_conv_parallelism_out,//layer_num_of_tiles_out_d;
                (14 + pw_tile_h) / pw_tile_h,//layer_num_of_ifm_tiles_h;
                (14 + pw_tile_w) / pw_tile_w,//layer_num_of_ifm_tiles_w;
                (14 + pw_tile_h) / pw_tile_h,//layer_num_of_ofm_tiles_h;
                (14 + pw_tile_w) / pw_tile_w,//layer_num_of_ofm_tiles_w;
                576 * pw_conv_parallelism_out / weights_group_items,//layer_num_of_weight_groups_for_one_pass;
                0,//layer_weights_offset;
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
 const int layer_45_pw_num_fils = 96 / alpha;
const int layer_45_pw_depth = 576;
const int layer_45_pw_filter_dim = 1;
 const int layer_45_pw_ifm_width = 14;
 //****************************
const layer_specs layer_45_pw_specs = {
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
                (576 + pw_tile_d) / pw_tile_d,//layer_num_of_tiles_in_d;
                (96 + pw_conv_parallelism_out) / pw_conv_parallelism_out,//layer_num_of_tiles_out_d;
                (14 + pw_tile_h) / pw_tile_h,//layer_num_of_ifm_tiles_h;
                (14 + pw_tile_w) / pw_tile_w,//layer_num_of_ifm_tiles_w;
                (14 + pw_tile_h) / pw_tile_h,//layer_num_of_ofm_tiles_h;
                (14 + pw_tile_w) / pw_tile_w,//layer_num_of_ofm_tiles_w;
                576 * pw_conv_parallelism_out / weights_group_items,//layer_num_of_weight_groups_for_one_pass;
                5115,//layer_weights_offset;
                1,//bool write_to_tmp;
                1,//bool fused_with_add;
                -128,//fms_dt layer_ifms_zero_point;
                0.12118788063526154,//,fms_dt layer_ofms_scale;
                10,//fms_dt layer_ofms_zero_point;
                5.718978723607331,//rec_scales_dt add_layer_scale_reciprocal;
                5,//biases_dt add_layer_zero_point;
                0.1662880778312683,//scales_dt skip_connection_other_layer_scale;
                12//biases_dt skip_connection_other_layer_zero_point;
                };
//****************************
 const int layer_47_pw_num_fils = 576 / alpha;
const int layer_47_pw_depth = 96;
const int layer_47_pw_filter_dim = 1;
 const int layer_47_pw_ifm_width = 14;
 //****************************
const layer_specs layer_47_pw_specs = {
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
                6,//layer_activation;
                (96 + pw_tile_d) / pw_tile_d,//layer_num_of_tiles_in_d;
                (576 + pw_conv_parallelism_out) / pw_conv_parallelism_out,//layer_num_of_tiles_out_d;
                (14 + pw_tile_h) / pw_tile_h,//layer_num_of_ifm_tiles_h;
                (14 + pw_tile_w) / pw_tile_w,//layer_num_of_ifm_tiles_w;
                (14 + pw_tile_h) / pw_tile_h,//layer_num_of_ofm_tiles_h;
                (14 + pw_tile_w) / pw_tile_w,//layer_num_of_ofm_tiles_w;
                96 * pw_conv_parallelism_out / weights_group_items,//layer_num_of_weight_groups_for_one_pass;
                5979,//layer_weights_offset;
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
 const int layer_48_dw_num_fils = 576 / alpha;
const int layer_48_dw_depth = 576;
const int layer_48_dw_filter_dim = 3;
 const int layer_48_dw_ifm_width = 14;
 //****************************
const layer_specs layer_48_dw_specs = {
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
                6,//layer_activation;
                (576 + pw_tile_d) / pw_tile_d,//layer_num_of_tiles_in_d;
                (576 + pw_conv_parallelism_out) / pw_conv_parallelism_out,//layer_num_of_tiles_out_d;
                (14 + pw_tile_h) / pw_tile_h,//layer_num_of_ifm_tiles_h;
                (14 + pw_tile_w) / pw_tile_w,//layer_num_of_ifm_tiles_w;
                (14 + pw_tile_h) / pw_tile_h,//layer_num_of_ofm_tiles_h;
                (14 + pw_tile_w) / pw_tile_w,//layer_num_of_ofm_tiles_w;
                576 * pw_conv_parallelism_out / weights_group_items,//layer_num_of_weight_groups_for_one_pass;
                0,//layer_weights_offset;
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
 const int layer_49_pw_num_fils = 96 / alpha;
const int layer_49_pw_depth = 576;
const int layer_49_pw_filter_dim = 1;
 const int layer_49_pw_ifm_width = 14;
 //****************************
const layer_specs layer_49_pw_specs = {
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
                (576 + pw_tile_d) / pw_tile_d,//layer_num_of_tiles_in_d;
                (96 + pw_conv_parallelism_out) / pw_conv_parallelism_out,//layer_num_of_tiles_out_d;
                (14 + pw_tile_h) / pw_tile_h,//layer_num_of_ifm_tiles_h;
                (14 + pw_tile_w) / pw_tile_w,//layer_num_of_ifm_tiles_w;
                (14 + pw_tile_h) / pw_tile_h,//layer_num_of_ofm_tiles_h;
                (14 + pw_tile_w) / pw_tile_w,//layer_num_of_ofm_tiles_w;
                576 * pw_conv_parallelism_out / weights_group_items,//layer_num_of_weight_groups_for_one_pass;
                6843,//layer_weights_offset;
                0,//bool write_to_tmp;
                1,//bool fused_with_add;
                -128,//fms_dt layer_ifms_zero_point;
                0.18612954020500183,//,fms_dt layer_ofms_scale;
                9,//fms_dt layer_ofms_zero_point;
                3.619118024720866,//rec_scales_dt add_layer_scale_reciprocal;
                -2,//biases_dt add_layer_zero_point;
                0.17485639452934265,//scales_dt skip_connection_other_layer_scale;
                5//biases_dt skip_connection_other_layer_zero_point;
                };
//****************************
 const int layer_51_pw_num_fils = 576 / alpha;
const int layer_51_pw_depth = 96;
const int layer_51_pw_filter_dim = 1;
 const int layer_51_pw_ifm_width = 14;
 //****************************
const layer_specs layer_51_pw_specs = {
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
                6,//layer_activation;
                (96 + pw_tile_d) / pw_tile_d,//layer_num_of_tiles_in_d;
                (576 + pw_conv_parallelism_out) / pw_conv_parallelism_out,//layer_num_of_tiles_out_d;
                (14 + pw_tile_h) / pw_tile_h,//layer_num_of_ifm_tiles_h;
                (14 + pw_tile_w) / pw_tile_w,//layer_num_of_ifm_tiles_w;
                (14 + pw_tile_h) / pw_tile_h,//layer_num_of_ofm_tiles_h;
                (14 + pw_tile_w) / pw_tile_w,//layer_num_of_ofm_tiles_w;
                96 * pw_conv_parallelism_out / weights_group_items,//layer_num_of_weight_groups_for_one_pass;
                7707,//layer_weights_offset;
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
 const int layer_53_dw_num_fils = 576 / alpha;
const int layer_53_dw_depth = 576;
const int layer_53_dw_filter_dim = 3;
 const int layer_53_dw_ifm_width = 15;
 //****************************
const layer_specs layer_53_dw_specs = {
                DW_CONV,//conv_layer_type;; 
                576,//layer_num_fils 
                2,//strides;
                3,//filter_size;
                0,//padding_left;
                1,//padding_right;
                0,//padding_top;
                1,//padding_bottom;
                576,//layer_depth;
                15,//layer_ifm_height;
                15,//layer_ifm_width;
                7,//layer_ofm_height;
                7,//layer_ofm_width;
                6,//layer_activation;
                (576 + pw_tile_d) / pw_tile_d,//layer_num_of_tiles_in_d;
                (576 + pw_conv_parallelism_out) / pw_conv_parallelism_out,//layer_num_of_tiles_out_d;
                (15 + pw_tile_h) / pw_tile_h,//layer_num_of_ifm_tiles_h;
                (15 + pw_tile_w) / pw_tile_w,//layer_num_of_ifm_tiles_w;
                (7 + pw_tile_h) / pw_tile_h,//layer_num_of_ofm_tiles_h;
                (7 + pw_tile_w) / pw_tile_w,//layer_num_of_ofm_tiles_w;
                576 * pw_conv_parallelism_out / weights_group_items,//layer_num_of_weight_groups_for_one_pass;
                0,//layer_weights_offset;
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
 const int layer_54_pw_num_fils = 160 / alpha;
const int layer_54_pw_depth = 576;
const int layer_54_pw_filter_dim = 1;
 const int layer_54_pw_ifm_width = 7;
 //****************************
const layer_specs layer_54_pw_specs = {
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
                (576 + pw_tile_d) / pw_tile_d,//layer_num_of_tiles_in_d;
                (160 + pw_conv_parallelism_out) / pw_conv_parallelism_out,//layer_num_of_tiles_out_d;
                (7 + pw_tile_h) / pw_tile_h,//layer_num_of_ifm_tiles_h;
                (7 + pw_tile_w) / pw_tile_w,//layer_num_of_ifm_tiles_w;
                (7 + pw_tile_h) / pw_tile_h,//layer_num_of_ofm_tiles_h;
                (7 + pw_tile_w) / pw_tile_w,//layer_num_of_ofm_tiles_w;
                576 * pw_conv_parallelism_out / weights_group_items,//layer_num_of_weight_groups_for_one_pass;
                8571,//layer_weights_offset;
                1,//bool write_to_tmp;
                0,//bool fused_with_add;
                -128,//fms_dt layer_ifms_zero_point;
                0.12638017535209656,//,fms_dt layer_ofms_scale;
                1,//fms_dt layer_ofms_zero_point;
                1,//rec_scales_dt add_layer_scale_reciprocal;
                0,//biases_dt add_layer_zero_point;
                1,//scales_dt skip_connection_other_layer_scale;
                0//biases_dt skip_connection_other_layer_zero_point;
                };
//****************************
 const int layer_55_pw_num_fils = 960 / alpha;
const int layer_55_pw_depth = 160;
const int layer_55_pw_filter_dim = 1;
 const int layer_55_pw_ifm_width = 7;
 //****************************
const layer_specs layer_55_pw_specs = {
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
                6,//layer_activation;
                (160 + pw_tile_d) / pw_tile_d,//layer_num_of_tiles_in_d;
                (960 + pw_conv_parallelism_out) / pw_conv_parallelism_out,//layer_num_of_tiles_out_d;
                (7 + pw_tile_h) / pw_tile_h,//layer_num_of_ifm_tiles_h;
                (7 + pw_tile_w) / pw_tile_w,//layer_num_of_ifm_tiles_w;
                (7 + pw_tile_h) / pw_tile_h,//layer_num_of_ofm_tiles_h;
                (7 + pw_tile_w) / pw_tile_w,//layer_num_of_ofm_tiles_w;
                160 * pw_conv_parallelism_out / weights_group_items,//layer_num_of_weight_groups_for_one_pass;
                10011,//layer_weights_offset;
                0,//bool write_to_tmp;
                0,//bool fused_with_add;
                1,//fms_dt layer_ifms_zero_point;
                0.0235294122248888,//,fms_dt layer_ofms_scale;
                -128,//fms_dt layer_ofms_zero_point;
                1,//rec_scales_dt add_layer_scale_reciprocal;
                0,//biases_dt add_layer_zero_point;
                1,//scales_dt skip_connection_other_layer_scale;
                0//biases_dt skip_connection_other_layer_zero_point;
                };
//****************************
 const int layer_56_dw_num_fils = 960 / alpha;
const int layer_56_dw_depth = 960;
const int layer_56_dw_filter_dim = 3;
 const int layer_56_dw_ifm_width = 7;
 //****************************
const layer_specs layer_56_dw_specs = {
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
                6,//layer_activation;
                (960 + pw_tile_d) / pw_tile_d,//layer_num_of_tiles_in_d;
                (960 + pw_conv_parallelism_out) / pw_conv_parallelism_out,//layer_num_of_tiles_out_d;
                (7 + pw_tile_h) / pw_tile_h,//layer_num_of_ifm_tiles_h;
                (7 + pw_tile_w) / pw_tile_w,//layer_num_of_ifm_tiles_w;
                (7 + pw_tile_h) / pw_tile_h,//layer_num_of_ofm_tiles_h;
                (7 + pw_tile_w) / pw_tile_w,//layer_num_of_ofm_tiles_w;
                960 * pw_conv_parallelism_out / weights_group_items,//layer_num_of_weight_groups_for_one_pass;
                0,//layer_weights_offset;
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
 const int layer_57_pw_num_fils = 160 / alpha;
const int layer_57_pw_depth = 960;
const int layer_57_pw_filter_dim = 1;
 const int layer_57_pw_ifm_width = 7;
 //****************************
const layer_specs layer_57_pw_specs = {
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
                (960 + pw_tile_d) / pw_tile_d,//layer_num_of_tiles_in_d;
                (160 + pw_conv_parallelism_out) / pw_conv_parallelism_out,//layer_num_of_tiles_out_d;
                (7 + pw_tile_h) / pw_tile_h,//layer_num_of_ifm_tiles_h;
                (7 + pw_tile_w) / pw_tile_w,//layer_num_of_ifm_tiles_w;
                (7 + pw_tile_h) / pw_tile_h,//layer_num_of_ofm_tiles_h;
                (7 + pw_tile_w) / pw_tile_w,//layer_num_of_ofm_tiles_w;
                960 * pw_conv_parallelism_out / weights_group_items,//layer_num_of_weight_groups_for_one_pass;
                12411,//layer_weights_offset;
                1,//bool write_to_tmp;
                1,//bool fused_with_add;
                -128,//fms_dt layer_ifms_zero_point;
                0.09578371047973633,//,fms_dt layer_ofms_scale;
                -6,//fms_dt layer_ofms_zero_point;
                6.874613571747451,//rec_scales_dt add_layer_scale_reciprocal;
                4,//biases_dt add_layer_zero_point;
                0.12638017535209656,//scales_dt skip_connection_other_layer_scale;
                1//biases_dt skip_connection_other_layer_zero_point;
                };
//****************************
 const int layer_59_pw_num_fils = 960 / alpha;
const int layer_59_pw_depth = 160;
const int layer_59_pw_filter_dim = 1;
 const int layer_59_pw_ifm_width = 7;
 //****************************
const layer_specs layer_59_pw_specs = {
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
                6,//layer_activation;
                (160 + pw_tile_d) / pw_tile_d,//layer_num_of_tiles_in_d;
                (960 + pw_conv_parallelism_out) / pw_conv_parallelism_out,//layer_num_of_tiles_out_d;
                (7 + pw_tile_h) / pw_tile_h,//layer_num_of_ifm_tiles_h;
                (7 + pw_tile_w) / pw_tile_w,//layer_num_of_ifm_tiles_w;
                (7 + pw_tile_h) / pw_tile_h,//layer_num_of_ofm_tiles_h;
                (7 + pw_tile_w) / pw_tile_w,//layer_num_of_ofm_tiles_w;
                160 * pw_conv_parallelism_out / weights_group_items,//layer_num_of_weight_groups_for_one_pass;
                14811,//layer_weights_offset;
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
 const int layer_60_dw_num_fils = 960 / alpha;
const int layer_60_dw_depth = 960;
const int layer_60_dw_filter_dim = 3;
 const int layer_60_dw_ifm_width = 7;
 //****************************
const layer_specs layer_60_dw_specs = {
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
                6,//layer_activation;
                (960 + pw_tile_d) / pw_tile_d,//layer_num_of_tiles_in_d;
                (960 + pw_conv_parallelism_out) / pw_conv_parallelism_out,//layer_num_of_tiles_out_d;
                (7 + pw_tile_h) / pw_tile_h,//layer_num_of_ifm_tiles_h;
                (7 + pw_tile_w) / pw_tile_w,//layer_num_of_ifm_tiles_w;
                (7 + pw_tile_h) / pw_tile_h,//layer_num_of_ofm_tiles_h;
                (7 + pw_tile_w) / pw_tile_w,//layer_num_of_ofm_tiles_w;
                960 * pw_conv_parallelism_out / weights_group_items,//layer_num_of_weight_groups_for_one_pass;
                0,//layer_weights_offset;
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
 const int layer_61_pw_num_fils = 160 / alpha;
const int layer_61_pw_depth = 960;
const int layer_61_pw_filter_dim = 1;
 const int layer_61_pw_ifm_width = 7;
 //****************************
const layer_specs layer_61_pw_specs = {
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
                (960 + pw_tile_d) / pw_tile_d,//layer_num_of_tiles_in_d;
                (160 + pw_conv_parallelism_out) / pw_conv_parallelism_out,//layer_num_of_tiles_out_d;
                (7 + pw_tile_h) / pw_tile_h,//layer_num_of_ifm_tiles_h;
                (7 + pw_tile_w) / pw_tile_w,//layer_num_of_ifm_tiles_w;
                (7 + pw_tile_h) / pw_tile_h,//layer_num_of_ofm_tiles_h;
                (7 + pw_tile_w) / pw_tile_w,//layer_num_of_ofm_tiles_w;
                960 * pw_conv_parallelism_out / weights_group_items,//layer_num_of_weight_groups_for_one_pass;
                17211,//layer_weights_offset;
                0,//bool write_to_tmp;
                1,//bool fused_with_add;
                -128,//fms_dt layer_ifms_zero_point;
                0.2120801955461502,//,fms_dt layer_ofms_scale;
                5,//fms_dt layer_ofms_zero_point;
                3.3522585543733454,//rec_scales_dt add_layer_scale_reciprocal;
                4,//biases_dt add_layer_zero_point;
                0.14546272158622742,//scales_dt skip_connection_other_layer_scale;
                4//biases_dt skip_connection_other_layer_zero_point;
                };
//****************************
 const int layer_63_pw_num_fils = 960 / alpha;
const int layer_63_pw_depth = 160;
const int layer_63_pw_filter_dim = 1;
 const int layer_63_pw_ifm_width = 7;
 //****************************
const layer_specs layer_63_pw_specs = {
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
                6,//layer_activation;
                (160 + pw_tile_d) / pw_tile_d,//layer_num_of_tiles_in_d;
                (960 + pw_conv_parallelism_out) / pw_conv_parallelism_out,//layer_num_of_tiles_out_d;
                (7 + pw_tile_h) / pw_tile_h,//layer_num_of_ifm_tiles_h;
                (7 + pw_tile_w) / pw_tile_w,//layer_num_of_ifm_tiles_w;
                (7 + pw_tile_h) / pw_tile_h,//layer_num_of_ofm_tiles_h;
                (7 + pw_tile_w) / pw_tile_w,//layer_num_of_ofm_tiles_w;
                160 * pw_conv_parallelism_out / weights_group_items,//layer_num_of_weight_groups_for_one_pass;
                19611,//layer_weights_offset;
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
 const int layer_64_dw_num_fils = 960 / alpha;
const int layer_64_dw_depth = 960;
const int layer_64_dw_filter_dim = 3;
 const int layer_64_dw_ifm_width = 7;
 //****************************
const layer_specs layer_64_dw_specs = {
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
                6,//layer_activation;
                (960 + pw_tile_d) / pw_tile_d,//layer_num_of_tiles_in_d;
                (960 + pw_conv_parallelism_out) / pw_conv_parallelism_out,//layer_num_of_tiles_out_d;
                (7 + pw_tile_h) / pw_tile_h,//layer_num_of_ifm_tiles_h;
                (7 + pw_tile_w) / pw_tile_w,//layer_num_of_ifm_tiles_w;
                (7 + pw_tile_h) / pw_tile_h,//layer_num_of_ofm_tiles_h;
                (7 + pw_tile_w) / pw_tile_w,//layer_num_of_ofm_tiles_w;
                960 * pw_conv_parallelism_out / weights_group_items,//layer_num_of_weight_groups_for_one_pass;
                0,//layer_weights_offset;
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
 const int layer_65_pw_num_fils = 320 / alpha;
const int layer_65_pw_depth = 960;
const int layer_65_pw_filter_dim = 1;
 const int layer_65_pw_ifm_width = 7;
 //****************************
const layer_specs layer_65_pw_specs = {
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
                (960 + pw_tile_d) / pw_tile_d,//layer_num_of_tiles_in_d;
                (320 + pw_conv_parallelism_out) / pw_conv_parallelism_out,//layer_num_of_tiles_out_d;
                (7 + pw_tile_h) / pw_tile_h,//layer_num_of_ifm_tiles_h;
                (7 + pw_tile_w) / pw_tile_w,//layer_num_of_ifm_tiles_w;
                (7 + pw_tile_h) / pw_tile_h,//layer_num_of_ofm_tiles_h;
                (7 + pw_tile_w) / pw_tile_w,//layer_num_of_ofm_tiles_w;
                960 * pw_conv_parallelism_out / weights_group_items,//layer_num_of_weight_groups_for_one_pass;
                22011,//layer_weights_offset;
                0,//bool write_to_tmp;
                0,//bool fused_with_add;
                -128,//fms_dt layer_ifms_zero_point;
                0.1458444595336914,//,fms_dt layer_ofms_scale;
                40,//fms_dt layer_ofms_zero_point;
                1,//rec_scales_dt add_layer_scale_reciprocal;
                0,//biases_dt add_layer_zero_point;
                1,//scales_dt skip_connection_other_layer_scale;
                0//biases_dt skip_connection_other_layer_zero_point;
                };
//****************************
 const int layer_66_pw_num_fils = 1280 / alpha;
const int layer_66_pw_depth = 320;
const int layer_66_pw_filter_dim = 1;
 const int layer_66_pw_ifm_width = 7;
 //****************************
const layer_specs layer_66_pw_specs = {
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
                6,//layer_activation;
                (320 + pw_tile_d) / pw_tile_d,//layer_num_of_tiles_in_d;
                (1280 + pw_conv_parallelism_out) / pw_conv_parallelism_out,//layer_num_of_tiles_out_d;
                (7 + pw_tile_h) / pw_tile_h,//layer_num_of_ifm_tiles_h;
                (7 + pw_tile_w) / pw_tile_w,//layer_num_of_ifm_tiles_w;
                (7 + pw_tile_h) / pw_tile_h,//layer_num_of_ofm_tiles_h;
                (7 + pw_tile_w) / pw_tile_w,//layer_num_of_ofm_tiles_w;
                320 * pw_conv_parallelism_out / weights_group_items,//layer_num_of_weight_groups_for_one_pass;
                26811,//layer_weights_offset;
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
#endif
