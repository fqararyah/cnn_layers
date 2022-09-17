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

	fill_weights_loop: for (int filter_index = 0;
			filter_index < pw_conv_parallelism_out; filter_index++) {
		for (int j = 0; j < max_num_of_weight_groups_in_depth; j++) {
			if (j < 1 || j < num_of_weight_groups) // j < 1 for the case in which the depth is less than weights_group_items
					{
				weights_grp_dt chunck = weights[layer_weights_offset
						+ (filter_index + starting_filter) * layer_depth + j];
				for (int k = 0; k < weights_group_items; k++) {
#pragma HLS UNROLL
					if (j * weights_group_items + k < layer_depth) {
						weights_tile[filter_index][j * weights_group_items + k] =
								chunck(k * weights_dt_width + weights_dt_offset,
										k * weights_dt_width);
					}
				}
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
