#include "../headers/pw_conv.h"

void pw_fill_channels_buffer(fms_dt channels[max_fms_size],
		fms_dt channels_tile[pw_tile_d][pw_tile_h][pw_tile_w], int tile_indx,
		int starting_d, const int layer_conv_d) {
#pragma HLS INLINE
	const int starting_indx = tile_indx * pw_tile_size;

	for (int t_d = 0; t_d < pw_tile_d; t_d++) {
#pragma HLS UNROLL
		for (int t_h = 0; t_h < pw_tile_h; t_h++) {
#pragma HLS UNROLL
			for (int t_w = 0; t_w < pw_tile_w; t_w++) {
#pragma HLS UNROLL
				if (starting_d + t_d < layer_conv_d) {
					channels_tile[t_d][t_h][t_w] = channels[starting_indx
							+ t_d * pw_tile_hw + t_h * pw_tile_w + t_w];
				}
			}
		}
	}
}

void pw_write_results_tile(
		pss_dt results_tile[pw_conv_parallelism_out][pw_tile_h][pw_tile_w],
		fms_dt results[max_fms_size], int tile_indx,
		fms_dt tmp_channels[max_tmp_fms_size], int starting_d,
		const int layer_conv_d, int read_write,
		const normalization_scheme normalization) {
	// read_write = 1 when the current layer is the one that is directly connected to the OFMs that have a residual connection to a previous layer
	// read_write = 2 when the current layer has a residual connection
	int num_of_tiles_processed_in_parallel = pw_conv_parallelism_out
			/ pw_tile_d;
	if (pw_conv_parallelism_out < pw_tile_d) {
		num_of_tiles_processed_in_parallel = 1;
	} else if (pw_conv_parallelism_out % pw_tile_d != 0) {
		num_of_tiles_processed_in_parallel = 1
				+ pw_conv_parallelism_out / pw_tile_d;
	}

	for (int tile_offset = 0; tile_offset < num_of_tiles_processed_in_parallel;
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
					if (t_d + starting_d < layer_conv_d) {
						fms_dt scaled_val = pw_relu_norm(
								results_tile[t_d][t_h][t_w], normalization);
						if (read_write == 0 || read_write == 2) {
							results[current_tile_indx + t_d * pw_tile_hw
									+ t_h * pw_tile_w + t_w] = scaled_val;
						}
						if (read_write == 1) {
							results[current_tile_indx + t_d * pw_tile_hw
									+ t_h * pw_tile_w + t_w] = scaled_val
									+ tmp_channels[current_tile_indx
											+ t_d * pw_tile_hw + t_h * pw_tile_w
											+ t_w];
						}
						if (read_write == 2) {
							tmp_channels[current_tile_indx + t_d * pw_tile_hw
									+ t_h * pw_tile_w + t_w] = scaled_val;
						}
					}
				}
			}
		}
	}
}

void pw_conv_eng(fms_dt channels_tile[pw_tile_d][pw_tile_h][pw_tile_w],
		weights_dt weights_tile[pw_conv_parallelism_out][max_conv_d],
		pss_dt results_tile[pw_conv_parallelism_out][pw_tile_h][pw_tile_w],
		int starting_conv_d, int starting_filter, const int layer_conv_d,
		const int layer_num_fils) {
#pragma HLS INLINE
	pw_conv_eng_loops: for (int f_d = 0; f_d < pw_conv_parallelism_out; f_d++) {
#pragma HLS UNROLL
		for (int t_h = 0; t_h < pw_tile_h; t_h++) {
#pragma HLS UNROLL
			for (int t_w = 0; t_w < pw_tile_w; t_w++) {
#pragma HLS UNROLL
				pss_dt tmp = 0;
				for (int t_d = 0; t_d < pw_conv_parallelism_in; t_d++) {
#pragma HLS UNROLL
					if (starting_conv_d + t_d < layer_conv_d
							&& starting_filter + f_d < layer_num_fils) {
						tmp += channels_tile[t_d][t_h][t_w]
								* weights_tile[f_d][starting_conv_d + t_d];
					}
				}
				results_tile[f_d][t_h][t_w] += tmp;
			}
		}
	}
}

void pw_conv_pipeline(fms_dt channels[max_fms_size],
		fms_dt results[max_fms_size],
		weights_dt weights_tile[pw_conv_parallelism_out][max_conv_d],
		pss_dt results_tile[pw_conv_parallelism_out][pw_tile_h][pw_tile_w],
		const int layer_num_fils, const int layer_conv_d,
		const int num_of_tiles_hw, const int num_of_tiles_w, int td_o,
		int t_in_h, int t_in_w, const int direction,
		const int num_of_tiles_d_in) {
#pragma HLS INLINE OFF
	conv2_itd_loop: for (int td_i = 0; td_i < num_of_tiles_d_in; td_i++) {
#pragma HLS PIPELINE
		fms_dt channels_buffer[pw_tile_d][pw_tile_h][pw_tile_w];
#pragma HLS ARRAY_PARTITION variable = channels_buffer complete dim = 0
		if (direction) {
			pw_fill_channels_buffer(channels, channels_buffer,
					td_i * num_of_tiles_hw + t_in_h * num_of_tiles_w + t_in_w,
					td_i * pw_tile_d, layer_conv_d);
		} else {
			pw_fill_channels_buffer(results, channels_buffer,
					td_i * num_of_tiles_hw + t_in_h * num_of_tiles_w + t_in_w,
					td_i * pw_tile_d, layer_conv_d);
		}
		pw_conv_eng(channels_buffer, weights_tile, results_tile,
				td_i * pw_tile_d, td_o * pw_conv_parallelism_out, layer_conv_d,
				layer_num_fils);
	}
}

void pw_conv(weights_grp_dt *weights, fms_dt channels[max_fms_size],
		fms_dt result[max_fms_size], const int layer, const int layer_conv_d,
		const int layer_num_fils, const int num_of_tiles_d_in,
		const int num_of_tiles_d_out, const int num_of_tiles_h,
		const int num_of_tiles_w, fms_dt tmp_channels[max_tmp_fms_size],
		int read_write, const int num_of_weight_groups,
		const normalization_scheme normalization, const int direction,
		const int layer_weights_offset) {
#pragma HLS INLINE off

	weights_dt weights_tile[pw_conv_parallelism_out][max_conv_d];

#pragma HLS ARRAY_PARTITION variable = weights_tile cyclic factor = pw_conv_parallelism_out/2 dim = 1
#pragma HLS ARRAY_PARTITION variable = weights_tile cyclic factor = weights_group_items dim = 2

	const int num_of_tiles_hw = num_of_tiles_h * num_of_tiles_w;

	conv2_ots_loop: for (int td_o = 0; td_o < num_of_tiles_d_out; td_o++) {
		fill_weights_tile_off_chip(weights, weights_tile,
				td_o * pw_conv_parallelism_out, layer, layer_num_fils,
				layer_conv_d, num_of_weight_groups, layer_weights_offset);
		//############width loop##############
		conv2_itw_loop: for (int t_in_w = 0; t_in_w < num_of_tiles_w;
				t_in_w++) {
			conv2_ith_loop: for (int t_in_h = 0; t_in_h < num_of_tiles_h;
					t_in_h++) {
				pss_dt results_tile[pw_conv_parallelism_out][pw_tile_h][pw_tile_w] =
						{ 0 };
#pragma HLS ARRAY_PARTITION variable = results_tile complete dim = 0
				//############depth loop##############
				//conv2_otd_loop: for (int td_i_o = 0;
				//		td_i_o < current_depth_to_median_depth; td_i_o++) {
				pw_conv_pipeline(channels, result, weights_tile, results_tile,
						layer_num_fils, layer_conv_d, num_of_tiles_hw,
						num_of_tiles_w, td_o, t_in_h, t_in_w, direction,
						num_of_tiles_d_in);
				//} // end depth loop###########
				if (direction) {
					pw_write_results_tile(results_tile, channels,
							td_o * (pw_conv_parallelism_out / pw_tile_d)
									* num_of_tiles_hw + t_in_h * num_of_tiles_w
									+ t_in_w, tmp_channels,
							td_o * pw_conv_parallelism_out, layer_num_fils,
							read_write, normalization);
				} else {
					pw_write_results_tile(results_tile, result,
							td_o * (pw_conv_parallelism_out / pw_tile_d)
									* num_of_tiles_hw + t_in_h * num_of_tiles_w
									+ t_in_w, tmp_channels,
							td_o * pw_conv_parallelism_out, layer_num_fils,
							read_write, normalization);
				}
			}
		}
	}
}
