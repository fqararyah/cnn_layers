#include "../headers/pw_conv.h"
#include "../headers/norm_act.h"
#include "../../utils/utils.h"
#include "../../client/quantization_and_biases.h"
#include "../../tests/test_utils.h"

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
		const int layer_conv_d, int read_write, const int layer_relu, int layer,
		const int num_of_tiles_hw) {
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

	biases_dt fused_zero_points_buffer[pw_conv_parallelism_out];
	scales_dt fused_scales_buffer[pw_conv_parallelism_out];
	fill_fused_zero_points(fused_zero_points, fused_zero_points_buffer,
			starting_d, layer);
	fill_fused_scales(fused_scales, fused_scales_buffer, starting_d, layer);
	scales_dt current_layer_scale = conv_fms_scales[layer + 1];
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

	biases_dt current_layer_zero_point = conv_fms_zero_points[layer + 1];
	scales_dt add_layer_scale = add_layers_fms_scales[layer + 1];
	biases_dt add_layer_zero_point = add_layers_fms_zero_points[layer + 1];

	fms_quantization_scheme normalization = { 0, 0, 0, 0 };
	normalization.ofm_zero_point = conv_fms_zero_points[layer + 1];
	normalization.ofm_scale = conv_fms_scales[layer + 1];

	for (int tile_offset = 0; tile_offset < num_of_tiles_processed_in_parallel;
			tile_offset++) {
#pragma HLS PIPELINE
//#pragma HLS dependence variable = results inter false
//#pragma HLS dependence variable = results intra false
		const int current_tile_indx =
				(tile_indx + tile_offset * num_of_tiles_hw) * pw_tile_size;
		for (int t_d = 0; t_d < pw_tile_d; t_d++) {
#pragma HLS UNROLL
			if (t_d + starting_d < layer_conv_d) {
				const int in_tile_index = tile_offset * pw_tile_d + t_d;
				normalization.fused_zero_point =
						fused_zero_points_buffer[in_tile_index];
				normalization.fused_scales = fused_scales_buffer[in_tile_index];
				for (int t_h = 0; t_h < pw_tile_h; t_h++) {
#pragma HLS UNROLL
					for (int t_w = 0; t_w < pw_tile_w; t_w++) {
#pragma HLS UNROLL

						fms_dt scaled_val =
								pw_relu_norm(
										results_tile[tile_offset * pw_tile_d
												+ t_d][t_h][t_w], normalization,
										layer_relu);
						const int to_write_at_index = current_tile_indx
								+ t_d * pw_tile_hw + t_h * pw_tile_w + t_w;
						if (read_write == 0 || read_write == 2) {
							results[to_write_at_index] = scaled_val;
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
							pss_f_dt tmp =
									(current_layer_scale
											* (scaled_val
													- current_layer_zero_point)
											+ skip_connection_other_layer_scale
													* (tmp_channels[to_write_at_index]
															- skip_connection_other_layer_zero_point))
											/ add_layer_scale
											+ add_layer_zero_point;
							results[to_write_at_index] = (fms_dt) tmp;
							if (read_write == 3) {
								tmp_channels[to_write_at_index] = (fms_dt) tmp;
							}
//							if (layer == 18) {
//								cout << current_layer_scale << " * ( "
//										<< scaled_val << " - "
//										<< current_layer_zero_point << ") + "
//										<< skip_connection_other_layer_scale
//										<< " * ("
//										<< tmp_channels[to_write_at_index]
//										<< " - "
//										<< skip_connection_other_layer_zero_point
//										<< ")) / " << add_layer_scale << " + "
//										<< add_layer_zero_point << "\n";
//							}
						}
						if (read_write == 2) {	//2: expansion
							tmp_channels[to_write_at_index] = scaled_val;
//							if (current_tile_indx + t_d * pw_tile_hw
//									+ t_h * pw_tile_w + t_w >= 56 * 56 * 24)
//								cout << layer << ": " << tile_indx << " "
//										<< current_tile_indx << " " << t_d
//										<< " " << t_h << " " << t_w << " "
//										<< current_tile_indx + t_d * pw_tile_hw
//												+ t_h * pw_tile_w + t_w << "\n";
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
		if (layer == 22) {
//			dumb_pw_channels_tile(
//					"/media/SSD2TB/wd/my_repos/DL_Benchmarking/tflite_scripts_imgnt_accuracy_and_weight_extraction/scratch_out/tile_ch."
//							+ to_string(td_i) + "txt", channels_buffer);
			dumb_pw_pss_tile(
					"/media/SSD2TB/wd/my_repos/DL_Benchmarking/tflite_scripts_imgnt_accuracy_and_weight_extraction/scratch_out/tile_pss"
							+ to_string(td_i) + ".txt", results_tile);
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
		const int layer_weights_offset, const int layer_relu) {
#pragma HLS INLINE off

	weights_dt weights_tile[pw_conv_parallelism_out][max_conv_d];
	fms_quantization_scheme normalization = { 0, 0, 0, 0 };

#pragma HLS ARRAY_PARTITION variable = weights_tile complete
#pragma HLS ARRAY_PARTITION variable = weights_tile cyclic factor = pw_weights_tile_partitioning_factor dim = 2

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
				pss_dt results_tile[pw_conv_parallelism_out][pw_tile_h][pw_tile_w];
				// if (td_o == 0 && t_in_h == 0 && t_in_w == 0 && layer == 4) {
				// 	dumb_pw_pss_tile(
				// 			"/media/SSD2TB/wd/my_repos/DL_Benchmarking/tflite_scripts_imgnt_accuracy_and_weight_extraction/scratch_out/tile_b_pss_f.txt",
				// 			results_tile);
				// 	dumb_pw_weights_tile(
				// 			"/media/SSD2TB/wd/my_repos/DL_Benchmarking/tflite_scripts_imgnt_accuracy_and_weight_extraction/scratch_out/tile_b_w.txt",
				// 			weights_tile, layer_conv_d);
				// }

#pragma HLS ARRAY_PARTITION variable = results_tile complete dim = 0
				//############depth loop##############
				//conv2_otd_loop: for (int td_i_o = 0;
				//		td_i_o < current_depth_to_median_depth; td_i_o++) {
				pw_conv_pipeline(channels, result, weights_tile, results_tile,
						layer, layer_num_fils, layer_conv_d, num_of_tiles_hw,
						num_of_tiles_w, td_o, t_in_h, t_in_w, direction,
						num_of_tiles_d_in);
				if (td_o == 0 && t_in_h == 0 && t_in_w == 0 && layer == 3) {
//					dumb_pw_pss_tile(
//							"/media/SSD2TB/wd/my_repos/DL_Benchmarking/tflite_scripts_imgnt_accuracy_and_weight_extraction/scratch_out/tile_pss_f.txt",
//							results_tile);
//					dumb_pw_weights_tile(
//							"/media/SSD2TB/wd/my_repos/DL_Benchmarking/tflite_scripts_imgnt_accuracy_and_weight_extraction/scratch_out/tile_w.txt",
//							weights_tile, layer_conv_d);
				}
				//} // end depth loop###########
				int tile_index = td_o * (pw_conv_parallelism_out / pw_tile_d)
						* num_of_tiles_hw + t_in_h * num_of_tiles_w + t_in_w;
//				if (layer == 7) {
//					cout<<num_of_tiles_d_out<<" x "<<num_of_tiles_hw<<" x "<<num_of_tiles_w<<"\n";
//				}
				if (direction) {
					pw_write_results_tile(results_tile, channels, tile_index,
							tmp_channels, td_o * pw_conv_parallelism_out,
							layer_num_fils, read_write, layer_relu, layer,
							num_of_tiles_hw);
				} else {
					pw_write_results_tile(results_tile, result, tile_index,
							tmp_channels, td_o * pw_conv_parallelism_out,
							layer_num_fils, read_write, layer_relu, layer,
							num_of_tiles_hw);
				}
			}
		}
	}
}
