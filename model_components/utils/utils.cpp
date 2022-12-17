#include "utils.h"
#include <stdint.h>
#include <string>
#include <fstream>
#include <iomanip>
#include <iostream>

using namespace std;

void fill_layer_weight_groups_tile_off_chip(weights_grp_dt *weights,
		weights_grp_dt weight_groups_buffer[num_of_weight_groups_in_the_largest_weight_tile],
		int starting_filter, const int layer_depth,
		const int num_of_weight_groups, const int layer_weights_offset,
		const int layer_num_fils) {
#pragma HLS INLINE off

	const int current_fill_offset = layer_weights_offset
			+ starting_filter * layer_depth / weights_group_items;

	if (starting_filter < layer_num_fils) {
		fill_weights_loop: for (int weight_grp_index = 0;
				weight_grp_index < num_of_weight_groups; weight_grp_index++) {
			weight_groups_buffer[weight_grp_index] = weights[current_fill_offset
					+ weight_grp_index];
		}
	}
}

void fill_weights_tile_from_weight_groups_tile(
		weights_grp_dt weight_groups_buffer[num_of_weight_groups_in_the_largest_weight_tile],
		weights_dt weights_tile[pw_conv_parallelism_out][max_conv_d],
		int starting_filter, const int layer_depth,
		const int num_of_weight_groups, const int layer_weights_offset) {
#pragma HLS INLINE off

//assumes pw_parallelism_out * filter depth is divisable by weight group number

	fill_weights_loop: for (int weight_grp_index = 0;
			weight_grp_index < num_of_weight_groups; weight_grp_index++) {
		weights_grp_dt chunck = weight_groups_buffer[weight_grp_index];
		for (int within_filter_index = 0;
				within_filter_index
						< num_of_weights_in_the_same_filter_and_group;
				within_filter_index++) {
#pragma HLS UNROLL
			for (int filter_index = 0; filter_index < pw_conv_parallelism_out;
					filter_index++) {
#pragma HLS UNROLL
				weights_tile[filter_index][weight_grp_index
						* num_of_weights_in_the_same_filter_and_group
						+ within_filter_index] = (weights_dt) chunck(
						(within_filter_index * pw_conv_parallelism_out
								+ filter_index) * weights_dt_width
								+ weights_dt_offset,
						(within_filter_index * pw_conv_parallelism_out
								+ filter_index) * weights_dt_width);
			}
		}
	}
}

void fill_weights_tile_off_chip(weights_grp_dt *weights,
		weights_dt weights_tile[pw_conv_parallelism_out][max_conv_d],
		int starting_filter, const int layer_depth,
		const int num_of_weight_groups, const int layer_weights_offset) {
//assumes pw_parallelism_out * filter depth is divisable by weight group number
	const int current_fill_offset = layer_weights_offset
			+ starting_filter * layer_depth / weights_group_items;

	fill_weights_loop: for (int weight_grp_index = 0;
			weight_grp_index < num_of_weight_groups; weight_grp_index++) {
		weights_grp_dt chunck = weights[current_fill_offset + weight_grp_index];
		for (int within_filter_index = 0;
				within_filter_index
						< num_of_weights_in_the_same_filter_and_group;
				within_filter_index++) {
#pragma HLS UNROLL
			for (int filter_index = 0; filter_index < pw_conv_parallelism_out;
					filter_index++) {
#pragma HLS UNROLL
				weights_tile[filter_index][weight_grp_index
						* num_of_weights_in_the_same_filter_and_group
						+ within_filter_index] = (weights_dt) chunck(
						(within_filter_index * pw_conv_parallelism_out
								+ filter_index) * weights_dt_width
								+ weights_dt_offset,
						(within_filter_index * pw_conv_parallelism_out
								+ filter_index) * weights_dt_width);
			}
		}
	}
}

void fill_layer_0_weights(
		layer_0_weights_dt weights_0[layer_0_num_fils][layer_0_depth][3][3]) {
	for (int i = 0; i < layer_0_num_fils; i++) {
		for (int j = 0; j < layer_0_depth; j++) {
			for (int k = 0; k < 3; k++) {
				for (int l = 0; l < 3; l++) {
					weights_0[i][j][k][l] = i * j * k * l % 8;
				}
			}
		}
	}
}

void fill_dw_layer_weights(
		const dw_weights_dt src[max_conv_d][max_conv_h * max_conv_w],
		dw_weights_dt dst[max_conv_d][max_conv_h * max_conv_w],
		const int conv_d, const int conv_h, const int conv_w) {
#pragma HLS INLINE OFF
	for (int d = 0; d < conv_d; d++) {
		if (d < conv_d) {
			for (int h = 0; h < conv_h; h++) {
				if (h < conv_h) {
					for (int w = 0; w < conv_w; w++) {
						if (w < conv_w) {
							dst[d][h * conv_w + w] = src[d][h * conv_w + w];
						}
					}
				}
			}
		}
	}
}

void fill_fused_zero_points_buffer(const biases_dt fused_zero_points[],
		biases_dt fused_zero_points_buffer[], int starting_d, int layer) {
	//const int starting_index = layers_fused_parameters_offsets[layer];
	for (int i = 0; i < pw_conv_parallelism_out; i++) {
		fused_zero_points_buffer[i] = fused_zero_points[starting_d + i];
	}
}

void fill_fused_scales_buffer(const fused_scales_dt fused_scales[],
		fused_scales_dt fused_scales_buffer[],
		const fused_scales_log_2_shifts_dt fused_scales_log_2_shifts[],
		fused_scales_log_2_shifts_dt fused_scales_log_2_shifts_buffer[],
		const relu_6_fused_scales_dt relu_6_fused_scales[],
		relu_6_fused_scales_dt relu_6_fused_scales_buffer[], int starting_d,
		int layer) {
	//const int starting_index = layers_fused_parameters_offsets[layer];
	for (int i = 0; i < pw_conv_parallelism_out; i++) {
		fused_scales_buffer[i] = fused_scales[starting_d + i];
		fused_scales_log_2_shifts_buffer[i] =
				fused_scales_log_2_shifts[starting_d + i];
		relu_6_fused_scales_buffer[i] = relu_6_fused_scales[starting_d + i];
	}
}

void fill_fused_scales_and_zero_points(
		const fused_scales_dt layer_fused_scales[],
		fused_scales_dt fused_scales[],
		const fused_scales_log_2_shifts_dt layer_fused_scales_log_2_shifts[],
		fused_scales_log_2_shifts_dt fused_scales_log_2_shifts[],
		const relu_6_fused_scales_dt layer_relu_6_fused_scales[],
		relu_6_fused_scales_dt relu_6_fused_scales[],
		const biases_dt layer_fused_zero_points[],
		biases_dt fused_zero_points[], const int layer_num_filters) {
#pragma HLS INLINE off

	for (int i = 0; i < max_conv_d; i++) {
		if (i >= layer_num_filters) {
			break;
		}
		fused_scales[i] = layer_fused_scales[i];
		fused_scales_log_2_shifts[i] = layer_fused_scales_log_2_shifts[i];
		relu_6_fused_scales[i] = layer_relu_6_fused_scales[i];
		fused_zero_points[i] = layer_fused_zero_points[i];
	}
}
