#include "../headers/layers_imp_common_includes.h"
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
//					cout
//							<< starting_indx + t_d * pw_tile_hw
//									+ t_h * pw_tile_w + t_w << " > "
//							<<
//							channels[starting_indx + t_d * pw_tile_hw
//									+ t_h * pw_tile_w + t_w]<<", ";
				}
			}
			//	cout<<"\n";
		}
	}
}

void read_and_scale_tile_from_tmp_channels(
		fms_dt tmp_channels[max_tmp_fms_size],
		pss_f_dt tmp_channels_scaled_tile[pw_conv_parallelism_out][pw_tile_h][pw_tile_w],
		int starting_d, const int layer_num_filters, const int num_of_tiles_hw,
		int tile_index, int read_write, const int layer) {
#pragma HLS INLINE off

	if (read_write == 1 || read_write == 3) {
		int num_of_tiles_processed_in_parallel = pw_conv_parallelism_out
				/ pw_tile_d;
		if (pw_conv_parallelism_out < pw_tile_d) {
			num_of_tiles_processed_in_parallel = 1;
		} else if (pw_conv_parallelism_out % pw_tile_d != 0) {
			num_of_tiles_processed_in_parallel = 1
					+ pw_conv_parallelism_out / pw_tile_d;
		}

		scales_dt skip_connection_other_layer_scale;
		biases_dt skip_connection_other_layer_zero_point;

		if (add_layers_fms_scales[layer - skip_connection_depth + 1] == 0) {
			skip_connection_other_layer_scale = conv_fms_scales[layer
					- skip_connection_depth + 1];
			skip_connection_other_layer_zero_point = conv_fms_zero_points[layer
					- skip_connection_depth + 1];
		} else {
			skip_connection_other_layer_scale = add_layers_fms_scales[layer
					- skip_connection_depth + 1];
			skip_connection_other_layer_zero_point =
					add_layers_fms_zero_points[layer - skip_connection_depth + 1];
		}

		read_a_tile_from_tmp_channels_tile_o_d: for (int tile_offset = 0;
				tile_offset < num_of_tiles_processed_in_parallel;
				tile_offset++) {
			const int current_tile_indx = (tile_index
					+ tile_offset * num_of_tiles_hw) * pw_tile_size;
			pw_write_results_tile_d: for (int t_d = 0; t_d < pw_tile_d; t_d++) {
				if (t_d + starting_d < layer_num_filters) {
#pragma HLS PIPELINE II=2
					const int in_tile_index = tile_offset * pw_tile_d + t_d;
					read_a_tile_from_tmp_channels_tile_h: for (int t_h = 0;
							t_h < pw_tile_h; t_h++) {
#pragma HLS UNROLL
						read_a_tile_from_tmp_channels_w: for (int t_w = 0;
								t_w < pw_tile_w; t_w++) {
#pragma HLS UNROLL
							const int to_read_from_index = current_tile_indx
									+ t_d * pw_tile_hw + t_h * pw_tile_w + t_w;

							tmp_channels_scaled_tile[tile_offset * pw_tile_d
									+ t_d][t_h][t_w] =
									skip_connection_other_layer_scale
											* (tmp_channels[to_read_from_index]
													- skip_connection_other_layer_zero_point);
//							if(tile_index == 0){
//								cout<<to_read_from_index<<" > "<<skip_connection_other_layer_scale<< " * (" << tmp_channels[to_read_from_index] << " - " << skip_connection_other_layer_zero_point <<
//										" = "<< tmp_channels_scaled_tile[tile_offset * pw_tile_d
//																			+ t_d][t_h][t_w] <<")\n";
//							}
						}

					}
				}
			}
		}
	}
}

void scale_pss_tile(
		pss_dt pss_tile[pw_conv_parallelism_out][pw_tile_h][pw_tile_w],
		pss_f_dt pss_tile_scaled[pw_conv_parallelism_out][pw_tile_h][pw_tile_w],
		const int layer_relu, fused_scales_dt fused_scales[],
		fused_scales_log_2_shifts_dt fused_scales_log_2_shifts[],
		relu_6_fused_scales_dt relu_6_fused_scales[],
		biases_dt fused_zero_points[], int starting_d, const int layer,
		int read_write) {
#pragma HLS INLINE

	int num_of_tiles_processed_in_parallel = pw_conv_parallelism_out
			/ pw_tile_d;
	if (pw_conv_parallelism_out < pw_tile_d) {
		num_of_tiles_processed_in_parallel = 1;
	} else if (pw_conv_parallelism_out % pw_tile_d != 0) {
		num_of_tiles_processed_in_parallel = 1
				+ pw_conv_parallelism_out / pw_tile_d;
	}

	biases_dt fused_zero_points_buffer[pw_conv_parallelism_out];
	fused_scales_dt fused_scales_buffer[pw_conv_parallelism_out];
	fused_scales_log_2_shifts_dt fused_scales_log_2_shifts_buffer[pw_conv_parallelism_out];
	relu_6_fused_scales_dt relu_6_fused_scales_buffer[pw_conv_parallelism_out];
	fill_fused_zero_points_buffer(fused_zero_points, fused_zero_points_buffer,
			starting_d, layer);
	fill_fused_scales_buffer(fused_scales, fused_scales_buffer, fused_scales_log_2_shifts, fused_scales_log_2_shifts_buffer,
			relu_6_fused_scales, relu_6_fused_scales_buffer, starting_d, layer);

	fms_quantization_scheme normalization = { 0, 0, 0, 0 };
	normalization.ofm_zero_point = conv_fms_zero_points[layer + 1];
	normalization.ofm_scale_rec = conv_fms_scales_rec[layer + 1];
	normalization.ofm_scale = conv_fms_scales[layer + 1];

	pss_to_fms_tile_o_d: for (int tile_offset = 0;
			tile_offset < num_of_tiles_processed_in_parallel; tile_offset++) {
		tile_d: for (int t_d = 0; t_d < pw_tile_d; t_d++) {
#pragma HLS UNROLL
			const int in_tile_index = tile_offset * pw_tile_d + t_d;
			normalization.fused_zero_point =
					fused_zero_points_buffer[in_tile_index];
			normalization.fused_scales = fused_scales_buffer[in_tile_index];
			normalization.fused_scales_log_2_shift = fused_scales_log_2_shifts_buffer[in_tile_index];
			normalization.relu_6_fused_scale =
					relu_6_fused_scales_buffer[in_tile_index];
			tile_h: for (int t_h = 0; t_h < pw_tile_h; t_h++) {
#pragma HLS UNROLL
				tile_w: for (int t_w = 0; t_w < pw_tile_w; t_w++) {
#pragma HLS PIPELINE
					pss_f_dt scaled_tmp;
					if (read_write == 0 || read_write == 2) {
						scaled_tmp =
								pw_relu_norm(
										pss_tile[tile_offset * pw_tile_d + t_d][t_h][t_w],
										normalization, layer_relu);
					} else if (read_write == 1 || read_write == 3) {
						scaled_tmp =
								pw_relu_norm_no_q_no_relu(
										pss_tile[tile_offset * pw_tile_d + t_d][t_h][t_w],
										normalization, layer_relu);
					}
					pss_tile_scaled[tile_offset * pw_tile_d + t_d][t_h][t_w] =
							scaled_tmp;

				}
			}
		}
	}
}

void pw_write_results_tile(
		pss_f_dt scaled_result_tile[pw_conv_parallelism_out][pw_tile_h][pw_tile_w],
		fms_dt results[max_fms_size], int tile_indx,
		fms_dt tmp_channels[max_tmp_fms_size],
		pss_f_dt tmp_channels_scaled_tile[pw_conv_parallelism_out][pw_tile_h][pw_tile_w],
		int starting_d, const int layer_num_filters, int read_write,
		const int layer_relu, int layer, const int num_of_tiles_hw) {
#pragma HLS INLINE off

// read_write = 1 when the current layer is the one that is directly connected to the OFMs that have a residual connection to a previous layer
// read_write = 2 when the current layer has a residual connection

	int num_of_tiles_processed_in_parallel = pw_conv_parallelism_out
			/ pw_tile_d;

	rec_scales_dt add_layer_scale_reciprocal = add_layers_fms_scales_rec[layer
			+ 1];
	biases_dt add_layer_zero_point = add_layers_fms_zero_points[layer + 1];

	pw_write_results_tile_o_d: for (int tile_offset = 0;
			tile_offset < num_of_tiles_processed_in_parallel; tile_offset++) {
#pragma HLS PIPELINE
		const int current_tile_indx =
				(tile_indx + tile_offset * num_of_tiles_hw) * pw_tile_size;
		pw_write_results_tile_d: for (int t_d = 0; t_d < pw_tile_d; t_d++) {
#pragma HLS UNROLL
			if (t_d + starting_d < layer_num_filters) {
				pw_write_results_tile_h: for (int t_h = 0; t_h < pw_tile_h;
						t_h++) {
#pragma HLS UNROLL
					pw_write_results_tile_w: for (int t_w = 0; t_w < pw_tile_w;
							t_w++) {
#pragma HLS UNROLL
						const int to_write_at_index = current_tile_indx
								+ t_d * pw_tile_hw + t_h * pw_tile_w + t_w;

						pss_f_dt scaled_val = scaled_result_tile[tile_offset
								* pw_tile_d + t_d][t_h][t_w];
//						if (layer == 39) {
//							cout << tile_indx << " " << current_tile_indx << " "
//									<< to_write_at_index << "\n";
//						}
						if (read_write == 0 || read_write == 2) {
							results[to_write_at_index] = (fms_dt) scaled_val;
							if (read_write == 2) {	//2: expansion
								tmp_channels[to_write_at_index] =
										(fms_dt) scaled_val;
//							if (current_tile_indx + t_d * pw_tile_hw
//									+ t_h * pw_tile_w + t_w >= 56 * 56 * 24)
//								cout << layer << ": " << tile_indx << " "
//										<< current_tile_indx << " " << t_d
//										<< " " << t_h << " " << t_w << " "
//										<< current_tile_indx + t_d * pw_tile_hw
//												+ t_h * pw_tile_w + t_w << "\n";
							}
//							if (layer == 22 && to_write_at_index == 0) {
//								cout << "\n************\n";
//								cout << results_tile[t_d][t_h][t_w]
//										<< "***results_tile[t_d][t_h][t_w]***\n";
//								cout << normalization.fused_zero_point
//										<< " ***fused_zero_point***\n";
//								cout << normalization.fused_scales
//										<< " ****fused_scales**\n";
//								cout << normalization.ofm_zero_point
//										<< " ***ofm_zero_point***\n";
//								cout << normalization.ofm_scale
//										<< " ***ofm_scale***\n";
//								cout
//										<< pw_relu_norm(
//												results_tile[tile_offset
//														* pw_tile_d + t_d][t_h][t_w],
//												normalization, layer_relu)
//										<< "****scaled_val***\n";
//								cout<<to_write_at_index<<"***index***\n";
//								cout << "\n************\n";
//							}
						} else if (read_write == 1 || read_write == 3) {//1: projection
							pss_f_dt distant_val =
									tmp_channels_scaled_tile[tile_offset
											* pw_tile_d + t_d][t_h][t_w];
							pss_f_dt tmp = (scaled_val + distant_val)
									* add_layer_scale_reciprocal
									+ add_layer_zero_point;

							fms_dt to_write_val = clamp(tmp);
							results[to_write_at_index] = to_write_val;
//							if (layer == 9) {
//								cout <<"( "
//										<< scaled_val << " + "
//										<< distant_val << ") * "
//										<< add_layer_scale_reciprocal
//										<< " + "
//										<< add_layer_zero_point<<"\n";
//							}
							if (read_write == 3) {
								tmp_channels[to_write_at_index] = to_write_val;
							}

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
				if (starting_conv_d == 0) {
					results_tile[f_d][t_h][t_w] = 0;
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
		int layer, const int layer_num_fils, const int layer_conv_d,
		const int num_of_tiles_hw, const int num_of_tiles_w, int td_o,
		int t_in_h, int t_in_w, const int direction,
		const int num_of_tiles_d_in) {
#pragma HLS INLINE OFF

	conv2_itd_loop: for (int td_i = 0; td_i < num_of_tiles_d_in; td_i++) {
#pragma HLS PIPELINE
		fms_dt channels_buffer[pw_tile_d][pw_tile_h][pw_tile_w];
#pragma HLS ARRAY_PARTITION variable = channels_buffer complete dim = 0
		if (direction) {
			pw_fill_channels_buffer(results, channels_buffer,
					td_i * num_of_tiles_hw + t_in_h * num_of_tiles_w + t_in_w,
					td_i * pw_tile_d, layer_conv_d);
		} else {
			pw_fill_channels_buffer(channels, channels_buffer,
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
		int read_write, const int num_of_weight_groups, const int direction,
		const int layer_weights_offset, const int layer_relu,
		fused_scales_dt fused_scales[],
		fused_scales_log_2_shifts_dt fused_scales_log_2_shifts[],
		relu_6_fused_scales_dt relu_6_fused_scales[],
		biases_dt fused_zero_points[]) {
#pragma HLS INLINE off

	weights_dt weights_tile[pw_conv_parallelism_out][max_conv_d];
#pragma HLS ARRAY_PARTITION variable = weights_tile complete dim = 1
#pragma HLS ARRAY_PARTITION variable = weights_tile cyclic dim = 2 factor= num_of_weights_in_the_same_filter_and_group

	fms_quantization_scheme normalization = { 0, 0, 0, 0 };

	pss_f_dt tmp_channels_scaled_tile[pw_conv_parallelism_out][pw_tile_h][pw_tile_w];
#pragma HLS ARRAY_PARTITION variable = tmp_channels_scaled_tile complete dim = 3

	const int num_of_tiles_hw = num_of_tiles_h * num_of_tiles_w;

	conv2_ots_loop: for (int td_o = 0; td_o < num_of_tiles_d_out; td_o++) {
		fill_weights_tile_off_chip(weights, weights_tile,
				td_o * pw_conv_parallelism_out,
				layer_conv_d, num_of_weight_groups, layer_weights_offset);
		conv2_ith_loop: for (int t_in_h = 0; t_in_h < num_of_tiles_h;
				t_in_h++) {
			//############width loop##############
			conv2_itw_loop: for (int t_in_w = 0; t_in_w < num_of_tiles_w;
					t_in_w++) {
				pss_dt results_tile[pw_conv_parallelism_out][pw_tile_h][pw_tile_w];
				pss_f_dt scaled_result_tile[pw_conv_parallelism_out][pw_tile_h][pw_tile_w];
#pragma HLS ARRAY_PARTITION variable = results_tile complete dim = 0
#pragma HLS ARRAY_PARTITION variable = scaled_result_tile complete dim = 0

				//############depth loop##############
				//conv2_otd_loop: for (int td_i_o = 0;
				//		td_i_o < current_depth_to_median_depth; td_i_o++) {

				int tile_index = td_o * (pw_conv_parallelism_out / pw_tile_d)
						* num_of_tiles_hw + t_in_h * num_of_tiles_w + t_in_w;

				read_and_scale_tile_from_tmp_channels(tmp_channels,
						tmp_channels_scaled_tile,
						td_o * pw_conv_parallelism_out, layer_num_fils,
						num_of_tiles_hw, tile_index, read_write, layer);

				pw_conv_pipeline(channels, result, weights_tile, results_tile,
						layer, layer_num_fils, layer_conv_d, num_of_tiles_hw,
						num_of_tiles_w, td_o, t_in_h, t_in_w, direction,
						num_of_tiles_d_in);

				scale_pss_tile(results_tile, scaled_result_tile, layer_relu,
						fused_scales, fused_scales_log_2_shifts, relu_6_fused_scales, fused_zero_points,
						td_o * pw_conv_parallelism_out, layer, read_write);
//
				if (direction) {
					pw_write_results_tile(scaled_result_tile, channels,
							tile_index, tmp_channels, tmp_channels_scaled_tile,
							td_o * pw_conv_parallelism_out, layer_num_fils,
							read_write, layer_relu, layer, num_of_tiles_hw);
				} else {
					pw_write_results_tile(scaled_result_tile, result,
							tile_index, tmp_channels, tmp_channels_scaled_tile,
							td_o * pw_conv_parallelism_out, layer_num_fils,
							read_write, layer_relu, layer, num_of_tiles_hw);
				}
			}
		}
	}
}
