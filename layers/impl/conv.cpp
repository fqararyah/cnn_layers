
#include "../headers/conv.h"
#include "../headers/pw_conv.h"
#include "../headers/norm_act.h"
#include "../../utils/utils.h"
#include "../../client/quantization_and_biases.h"

void fill_channels_buffer_0(
		fms_dt channels[input_image_depth][input_image_height][input_image_width],
		fms_dt channels_tile[layer_0_depth][pw_tile_h][pw_tile_w],
		int tile_indx, int c_w) {
#pragma HLS INLINE
	const int starting_indx = tile_indx * pw_tile_size;

	for (int t_w = 0; t_w < pw_tile_w; t_w++) {
		for (int t_h = 0; t_h < pw_tile_h; t_h++) {
			for (int t_d = 0; t_d < layer_0_depth; t_d++) {
#pragma HLS UNROLL
				channels_tile[t_d][t_h][t_w] = channels[starting_indx
						+ t_d * pw_tile_hw + t_h * pw_tile_w
						+ t_w * layer_0_strides + c_w];
			}
		}
	}
}

void write_results_tile_0(
		pss_dt results_tile[pw_conv_parallelism_out][pw_tile_h][pw_tile_w],
		fms_dt results[max_fms_size], int tile_indx, const int layer_conv_d,
		int starting_d, int layer) {

	biases_dt fused_zero_points_buffer[pw_conv_parallelism_out];
	scales_dt fused_scales_buffer[pw_conv_parallelism_out];
	fill_fused_zero_points(fused_zero_points, fused_zero_points_buffer, starting_d, layer);
	fill_fused_scales(fused_scales, fused_scales_buffer, starting_d, layer);

	fms_quantization_scheme normalization = {0,0,0,0};

	for (int tile_offset = 0; tile_offset < pw_conv_parallelism_out / pw_tile_d;
			tile_offset++) {
#pragma HLS PIPELINE
//#pragma HLS dependence variable = results inter false
//#pragma HLS dependence variable = results intra false
		const int current_tile_indx = (tile_indx + tile_offset) * pw_tile_size;
		for (int t_h = 0; t_h < pw_tile_h; t_h++) {
#pragma HLS UNROLL
			for (int t_w = 0; t_w < pw_tile_w; t_w++) {
#pragma HLS UNROLL
				for (int t_d = 0; t_d < pw_tile_d; t_d++) {
#pragma HLS UNROLL
					if (t_d < layer_conv_d) {
						const int in_tile_index = tile_offset * pw_tile_d + t_d;
						normalization.fused_zero_point = fused_zero_points_buffer[in_tile_index];
						normalization.fused_scales = fused_scales_buffer[in_tile_index];
						normalization.ofm_zero_point = conv_fms_zero_points[layer + 1];
						fms_dt scaled_val = conv_relu_norm(
								results_tile[t_d][t_h][t_w], normalization);
						results[current_tile_indx + t_d * pw_tile_hw
								+ t_h * pw_tile_w + t_w] = scaled_val;
					}
				}
			}
		}
	}
}

void layer_0_using_pw(
		weights_dt weights_0[layer_0_num_fils][layer_0_depth][3][3],
		fms_dt channels[input_image_depth][input_image_height][input_image_width],
		fms_dt result[max_fms_size], const int layer, const int layer_conv_d,
		const int layer_num_fils, const int num_of_tiles_d_in,
		const int num_of_tiles_d_out, const int num_of_tiles_h,
		const int num_of_tiles_w) {
#pragma HLS INLINE off

	fms_quantization_scheme normalization = {0,0,0,0};
	weights_dt weights_tile[pw_conv_parallelism_out][max_conv_d];

#pragma HLS ARRAY_PARTITION variable = weights_tile complete dim = 1
#pragma HLS ARRAY_PARTITION variable = weights_tile cyclic factor = pw_conv_parallelism_in dim = 2

	const int num_of_tiles_hw = num_of_tiles_h * num_of_tiles_w;

	conv2_ots_loop: for (int td_o = 0; td_o < num_of_tiles_d_out; td_o++) {
		for (int c_h = 0; c_h < 3; c_h++) {
			for (int c_w = 0; c_w < 3; c_w++) {
				for (int fil = td_o * pw_conv_parallelism_out;
						fil < (td_o + 1) * pw_conv_parallelism_out; fil++) {
					for (int d = 0; d < 3; d++) {
						weights_tile[fil][d] = weights_0[fil][d][c_h][c_w];
					}
				}
				conv2_ith_loop: for (int t_in_h = 0; t_in_h < num_of_tiles_h;
						t_in_h += layer_0_strides) {
					//############width loop##############
					conv2_itw_loop: for (int t_in_w = 0;
							t_in_w < num_of_tiles_w; t_in_w +=
									layer_0_strides) {
						pss_dt results_tile[pw_conv_parallelism_out][pw_tile_h][pw_tile_w] =
								{ 0 };
#pragma HLS ARRAY_PARTITION variable = results_tile complete dim = 0
						//############depth loop##############

						fms_dt channels_buffer[layer_0_depth][pw_tile_h][pw_tile_w];
#pragma HLS ARRAY_PARTITION variable = channels_buffer complete dim = 0
						fill_channels_buffer_0(channels, channels_buffer,
								(t_in_h + c_h) * num_of_tiles_w + t_in_w, c_w);
						pw_conv_eng(channels_buffer, weights_tile, results_tile,
								0, td_o * pw_conv_parallelism_out, layer_conv_d,
								layer_num_fils);
						write_results_tile_0(results_tile, result,
								td_o * (pw_conv_parallelism_out / pw_tile_d)
										* num_of_tiles_hw
										+ t_in_h / layer_0_strides
												* num_of_tiles_w
										+ t_in_w / layer_0_strides,
								layer_num_fils, td_o * pw_conv_parallelism_out, layer);
					}
				}
			}
		}
	}
}
