#include "utils.h"
#include <stdint.h>
#include <string>
#include <fstream>
#include <iomanip>
#include <iostream>

using namespace std;

void fill_weights_tile_off_chip(weights_grp_dt *weights,
		weights_dt weights_tile[pw_conv_parallelism_out][max_conv_d],
		int starting_filter, const int layer, const int layer_num_fils,
		const int layer_depth, const int num_of_weight_groups,
		const int layer_weights_offset) {
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
				if (layer == 3) {
					cout << (within_filter_index * pw_conv_parallelism_out
							+ filter_index) * weights_dt_width
							+ weights_dt_offset <<" " << weights_dt_offset << "****c**" << "\n";
					cout<<(weights_dt)chunck(
							(within_filter_index * pw_conv_parallelism_out
									+ filter_index) * weights_dt_width
									+ weights_dt_offset,
							(within_filter_index * pw_conv_parallelism_out
									+ filter_index) * weights_dt_width)<<"\n";
				}
				weights_tile[filter_index][weight_grp_index
						* num_of_weights_in_the_same_filter_and_group
						+ within_filter_index] = (weights_dt)chunck(
						(within_filter_index * pw_conv_parallelism_out
								+ filter_index) * weights_dt_width
								+ weights_dt_offset,
						(within_filter_index * pw_conv_parallelism_out
								+ filter_index) * weights_dt_width);
			}
		}
	}
	if (layer == 3) {
	cout<<(weights_dt)weights[0](7, 0)<<" @zero\n";
	cout<<(weights_dt)weights[0](8, 15)<<" @one\n";
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
		const dw_weights_dt src[max_conv_d][max_conv_h][max_conv_w],
		dw_weights_dt dst[max_conv_d][max_conv_h][max_conv_w], const int conv_d,
		const int conv_h, const int conv_w) {
#pragma HLS INLINE OFF
	for (int d = 0; d < max_conv_d; d++) {
		if (d < conv_d) {
			for (int h = 0; h < max_conv_h; h++) {
				if (h < conv_h) {
					for (int w = 0; w < max_conv_w; w++) {
						if (w < conv_w) {
							dst[d][h][w] = src[d][h][w];
						}
					}
				}
			}
		}
	}
}

void fill_fused_zero_points(const biases_dt fused_zero_points[],
		biases_dt fused_zero_points_buffer[pw_conv_parallelism_out],
		int starting_d, int layer) {
	const int starting_index = layers_fused_parameters_offsets[layer];
	for (int i = 0; i < pw_conv_parallelism_out; i++) {
		fused_zero_points_buffer[i] = fused_zero_points[starting_index
				+ starting_d + i];
	}
}

void fill_fused_scales(const scales_dt fused_scales[],
		scales_dt fused_scales_buffer[pw_conv_parallelism_out], int starting_d,
		int layer) {
	const int starting_index = layers_fused_parameters_offsets[layer];
	for (int i = 0; i < pw_conv_parallelism_out; i++) {
		fused_scales_buffer[i] = fused_scales[starting_index + starting_d + i];
	}
}

//void _5_fill_layers_weights(
//		layer_0_weights_dt weights_0[layer_0_num_fils][layer_0_depth][3][3],
//		dw_weights_dt dw_weights_1[layer_1_dw_depth][3][3],
//		weights_dt pw_weights_1[layer_1_pw_num_fils][layer_1_pw_depth],
//		weights_dt pw_weights_2[layer_2_pw_num_fils][layer_2_pw_depth],
//		weights_dt pw_weights_3[layer_3_pw_num_fils][layer_3_pw_depth]) {
//	//**********dw**************
//	for (int f = 0; f < layer_0_num_fils; f++) {
//		for (int i = 0; i < layer_0_depth; i++) {
//			for (int j = 0; j < 3; j++) {
//				for (int k = 0; k < 3; k++) {
//					weights_0[f][i][j][k] = (i + j + k) % 8;
//				}
//			}
//		}
//	}
//
//	for (int i = 0; i < layer_1_dw_depth; i++) {
//		for (int j = 0; j < 3; j++) {
//			for (int k = 0; k < 3; k++) {
//				dw_weights_1[i][j][k] = (i + j + k) % 8;
//			}
//		}
//	}
//	//**********dw**************
//
//	//**********pw**************
//	for (int i = 0; i < layer_1_pw_num_fils; i++) {
//		for (int j = 0; j < layer_1_pw_depth; j++) {
//			pw_weights_1[i][j] = (i + j) % 8;
//		}
//	}
//	for (int i = 0; i < layer_2_pw_num_fils; i++) {
//		for (int j = 0; j < layer_2_pw_depth; j++) {
//			pw_weights_2[i][j] = (i + j) % 8;
//		}
//	}
//	for (int i = 0; i < layer_3_pw_num_fils; i++) {
//		for (int j = 0; j < layer_3_pw_depth; j++) {
//			pw_weights_3[i][j] = (i + j) % 8;
//		}
//	}
//}
//
//void v1_3_fill_layers_weights(
//		layer_0_weights_dt weights_0[layer_0_num_fils][layer_0_depth][layer_0_filter_size][layer_0_filter_size],
//		dw_weights_dt dw_weights_1[layer_1_dw_depth][v1_layer_1_dw_filter_size][v1_layer_1_dw_filter_size],
//		weights_dt pw_weights_1[v1_layer_2_pw_num_fils][v1_layer_2_pw_depth]) {
//	for (int f = 0; f < layer_0_num_fils; f++) {
//		for (int i = 0; i < layer_0_depth; i++) {
//			for (int j = 0; j < layer_0_filter_size; j++) {
//				for (int k = 0; k < layer_0_filter_size; k++) {
//					weights_0[f][i][j][k] = (i + j + k) % 8;
//				}
//			}
//		}
//	}
//
//	for (int i = 0; i < v1_layer_1_dw_depth; i++) {
//		for (int j = 0; j < v1_layer_1_dw_filter_size; j++) {
//			for (int k = 0; k < v1_layer_1_dw_num_fils; k++) {
//				dw_weights_1[i][j][k] = (i + j + k) % 8;
//			}
//		}
//	}
//	//**********dw**************
//
//	//**********pw**************
//	for (int i = 0; i < v1_layer_2_pw_num_fils; i++) {
//		for (int j = 0; j < v1_layer_2_pw_depth; j++) {
//			pw_weights_1[i][j] = (i + j) % 8;
//		}
//	}
//}
//
//void v1_4_fill_layers_weights(
//		layer_0_weights_dt weights_0[layer_0_num_fils][layer_0_depth][layer_0_filter_size][layer_0_filter_size],
//		dw_weights_dt dw_weights_1[layer_1_dw_depth][v1_layer_1_dw_filter_size][v1_layer_1_dw_filter_size],
//		dw_weights_dt dw_weights_2[v1_layer_2_dw_depth][v1_layer_2_dw_filter_size][v1_layer_2_dw_filter_size],
//		weights_dt pw_weights_2[v1_layer_2_pw_num_fils][v1_layer_2_pw_depth]) {
//	v1_3_fill_layers_weights(weights_0, dw_weights_1, pw_weights_2);
//	for (int i = 0; i < v1_layer_2_dw_depth; i++) {
//		for (int j = 0; j < v1_layer_2_dw_filter_size; j++) {
//			for (int k = 0; k < v1_layer_2_dw_num_fils; k++) {
//				dw_weights_2[i][j][k] = (i + j + k) % 8;
//			}
//		}
//	}
//}
//
//void v1_7_layer_1_dw(
//		layer_0_weights_dt weights_0[layer_0_num_fils][layer_0_depth][layer_0_filter_size][layer_0_filter_size],
//		dw_weights_dt dw_weights_1[v1_layer_1_dw_depth][v1_layer_3_dw_filter_size][v1_layer_3_dw_filter_size],
//		dw_weights_dt dw_weights_2[v1_layer_2_dw_depth][v1_layer_2_dw_filter_size][v1_layer_2_dw_filter_size],
//		dw_weights_dt dw_weights_3[v1_layer_3_dw_depth][v1_layer_3_dw_filter_size][v1_layer_3_dw_filter_size],
//		weights_dt pw_weights_2[v1_layer_2_pw_num_fils][v1_layer_2_pw_depth],
//		weights_dt pw_weights_3[v1_layer_3_pw_num_fils][v1_layer_3_pw_depth],
//		weights_dt pw_weights_4[v1_layer_4_pw_num_fils][v1_layer_4_pw_depth]) {
//	v1_4_fill_layers_weights(weights_0, dw_weights_1, dw_weights_2,
//			pw_weights_2);
//
//	v1_3_fill_layers_weights(weights_0, dw_weights_1, pw_weights_2);
//
//	for (int i = 0; i < v1_layer_3_dw_depth; i++) {
//		for (int j = 0; j < v1_layer_3_dw_filter_size; j++) {
//			for (int k = 0; k < v1_layer_3_dw_num_fils; k++) {
//				dw_weights_3[i][j][k] = (i + j + k) % 8;
//			}
//		}
//	}
//
//	for (int i = 0; i < v1_layer_3_pw_num_fils; i++) {
//		for (int j = 0; j < v1_layer_3_pw_depth; j++) {
//			pw_weights_3[i][j] = (i + j) % 8;
//		}
//	}
//	for (int i = 0; i < v1_layer_4_pw_num_fils; i++) {
//		for (int j = 0; j < v1_layer_4_pw_depth; j++) {
//			pw_weights_4[i][j] = (i + j) % 8;
//		}
//	}
//
//}

//void _7_fill_layers_weights(
//		layer_0_weights_dt weights_0[layer_0_num_fils][layer_0_depth][layer_0_filter_size][layer_0_filter_size],
//		dw_weights_dt dw_weights_1[layer_1_dw_depth][layer_1_dw_filter_size][layer_1_dw_filter_size],
//		dw_weights_dt dw_weights_3[layer_3_dw_depth][layer_3_dw_filter_size][layer_3_dw_filter_size],
//		weights_dt pw_weights_1[layer_1_pw_num_fils][layer_1_pw_depth],
//		weights_dt pw_weights_2[layer_2_pw_num_fils][layer_2_pw_depth],
//		weights_dt pw_weights_3[layer_3_pw_num_fils][layer_3_pw_depth],
//		weights_dt pw_weights_4[layer_4_pw_num_fils][layer_4_pw_depth]) {
//	//**********dw**************
//	for (int f = 0; f < layer_0_num_fils; f++) {
//		for (int i = 0; i < layer_0_depth; i++) {
//			for (int j = 0; j < layer_0_filter_size; j++) {
//				for (int k = 0; k < layer_0_filter_size; k++) {
//					weights_0[f][i][j][k] = (i + j + k) % 8;
//				}
//			}
//		}
//	}
//
//	for (int i = 0; i < layer_1_dw_depth; i++) {
//		for (int j = 0; j < layer_1_dw_filter_size; j++) {
//			for (int k = 0; k < layer_1_dw_filter_size; k++) {
//				dw_weights_1[i][j][k] = (i + j + k) % 8;
//			}
//		}
//	}
//
//	for (int i = 0; i < layer_3_dw_depth; i++) {
//		for (int j = 0; j < layer_3_dw_filter_size; j++) {
//			for (int k = 0; k < layer_3_dw_filter_size; k++) {
//				dw_weights_3[i][j][k] = (i + j + k) % 8;
//			}
//		}
//	}
//	//**********dw**************
//
//	//**********pw**************
//	for (int i = 0; i < layer_1_pw_num_fils; i++) {
//		for (int j = 0; j < layer_1_pw_depth; j++) {
//			pw_weights_1[i][j] = (i + j) % 8;
//		}
//	}
//	for (int i = 0; i < layer_2_pw_num_fils; i++) {
//		for (int j = 0; j < layer_2_pw_depth; j++) {
//			pw_weights_2[i][j] = (i + j) % 8;
//		}
//	}
//	for (int i = 0; i < layer_3_pw_num_fils; i++) {
//		for (int j = 0; j < layer_3_pw_depth; j++) {
//			pw_weights_3[i][j] = (i + j) % 8;
//		}
//	}
//	for (int i = 0; i < layer_4_pw_num_fils; i++) {
//		for (int j = 0; j < layer_4_pw_depth; j++) {
//			pw_weights_3[i][j] = (i + j) % 8;
//		}
//	}
//}
