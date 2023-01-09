#ifndef UTILS
#define UTILS

#include <stdint.h>
#include <string>
#include "../basic_defs/basic_defs_glue.h"

#include "../model/headers/model_glue.h"

using namespace std;


void fill_layers_weights_cpu(weights_dt *weights,
							 weights_dt weights_buffer[][max_conv_d],
							 int starting_filter, const int layer_depth,
							 const int layer_weights_offset,
							 const int layer_num_fils);

void fill_layer_0_s_weights(
		layer_0_weights_dt weights_0[layer_0_s_num_fils][layer_0_s_depth][3][3]);

void fill_dw_layer_weights(
		const dw_weights_dt src[max_conv_d][max_conv_h * max_conv_w],
		dw_weights_dt dst[max_conv_d][max_conv_h * max_conv_w],
		const int conv_d, const int conv_h, const int conv_w);

void fill_weights_tile_off_chip(weights_grp_dt *weights,
		weights_dt weights_tile[pw_conv_parallelism_out][max_conv_d],
		int starting_filter, const int layer_depth,
		const int num_of_weight_groups, const int layer_weights_offset);

void fill_layer_weight_groups_tile_off_chip(weights_grp_dt *weights,
		weights_grp_dt weight_groups_buffer[num_of_weight_groups_in_the_largest_weight_tile],
		int starting_filter,
		const int layer_depth, const int num_of_weight_groups,
		const int layer_weights_offset,
		const int layer_num_fils);
		
void fill_weights_tile_from_weight_groups_tile(weights_grp_dt weight_groups_buffer[num_of_weight_groups_in_the_largest_weight_tile],
		weights_dt weights_tile[pw_conv_parallelism_out][max_conv_d],
		int starting_filter,
		const int layer_depth, const int num_of_weight_groups,
		const int layer_weights_offset);

void fill_fused_zero_points_buffer(const biases_dt fused_zero_points[],
		biases_dt fused_zero_points_buffer[pw_conv_parallelism_out],
		int starting_d, int layer, const int current_layer_fused_parameters_offset);

void fill_fused_scales_buffer(const fused_scales_dt fused_scales[],
		fused_scales_dt fused_scales_buffer[],
		const fused_scales_log_2_shifts_dt fused_scales_log_2_shifts[],
		fused_scales_log_2_shifts_dt fused_scales_log_2_shifts_buffer[],
		const relu_6_fused_scales_dt relu_6_fused_scales[],
		relu_6_fused_scales_dt relu_6_fused_scales_buffer[], int starting_d,
		int layer, const int current_layer_fused_parameters_offset);

void fill_fused_scales_and_zero_points(
		const fused_scales_dt layer_fused_scales[],
		fused_scales_dt fused_scales[],
		const fused_scales_log_2_shifts_dt layer_fused_scales_log_2_shifts[],
		fused_scales_log_2_shifts_dt fused_scales_log_2_shifts[],
		const relu_6_fused_scales_dt layer_relu_6_fused_scales[],
		relu_6_fused_scales_dt relu_6_fused_scales[],
		const biases_dt layer_fused_zero_points[],
		biases_dt fused_zero_points[], const int layer_num_filters);

//void v1_3_fill_layers_weights(layer_0_weights_dt weights_0[layer_0_s_num_fils][layer_0_s_depth][layer_0_s_filter_dim][layer_0_s_filter_dim],
//                         dw_weights_dt dw_weights_1[layer_1_dw_depth][v1_layer_1_dw_filter_size][v1_layer_1_dw_filter_size],
//                         weights_dt pw_weights_2[v1_layer_2_pw_num_fils][v1_layer_2_pw_depth]);
//
//void v1_4_fill_layers_weights(layer_0_weights_dt weights_0[layer_0_s_num_fils][layer_0_s_depth][layer_0_s_filter_dim][layer_0_s_filter_dim],
//                         dw_weights_dt dw_weights_1[layer_1_dw_depth][v1_layer_1_dw_filter_size][v1_layer_1_dw_filter_size],
//                         dw_weights_dt dw_weights_2[v1_layer_1_dw_depth][v1_layer_1_dw_filter_size][v1_layer_1_dw_filter_size],
//                         weights_dt pw_weights_2[v1_layer_2_pw_num_fils][v1_layer_2_pw_depth]);

//void v1_7_fill_layers_weights(layer_0_weights_dt weights_0[layer_0_s_num_fils][layer_0_s_depth][layer_0_s_filter_dim][layer_0_s_filter_dim],
//	dw_weights_dt dw_weights_1[v1_layer_1_dw_depth][v1_layer_4_dw_filter_size][v1_layer_4_dw_filter_size],
//	dw_weights_dt dw_weights_2[v1_layer_1_dw_depth][v1_layer_1_dw_filter_size][v1_layer_1_dw_filter_size],
//	dw_weights_dt dw_weights_3[v1_layer_4_dw_depth][v1_layer_4_dw_filter_size][v1_layer_4_dw_filter_size],
//	weights_dt pw_weights_2[v1_layer_2_pw_num_fils][v1_layer_2_pw_depth],
//	weights_dt pw_weights_3[v1_layer_2_pw_num_fils][v1_layer_2_pw_depth],
//	weights_dt pw_weights_4[v1_layer_2_pw_num_fils][v1_layer_2_pw_depth]);

// void read_image(string file_name, uint8_t *image);
// void read_image(string file_name, float *image);
// void read_image_m(string file_name, float image[image_depth][image_height][image_width], bool dummy);
// void read_weights(string file_name, weights_dt weights[32][1][3][3]);
// void print_image(int width, int height, float *image);
// void print_image_m(float image[image_depth][image_height][image_width]);
//
// void read_weights(string file_name, weights_dt weights[32][9]);
// void read_weights_m(string file_name, weights_dt weights[num_fils][conv_d], bool dummy);

#endif
