#include "../headers/layers_imp_common_includes.h"
#include "../headers/dw_conv.h"
#include "../headers/pw_conv.h"

void fill_ifms_cols(fms_dt channels[max_fms_size],
		fms_dt ifms_buffer[dw_pipeline_depth][dw_max_v2_buffer_height][dw_max_v2_buffer_width],
		const int offset_in_ifms_buffer_h, const int offset_in_ifms_buffer_w,
		const int last_to_fill_in_h, const int tile_index_in_h,
		const int tile_index_in_w, int absolute_offset, const int padding_top,
		const int padding_left, const int padding_right, const int ifms_height,
		const int ifms_width, const int d_in_pipeline,
		const fms_dt zero_point) {
#pragma HLS INLINE

	const int starting_fill_w_offset = tile_index_in_w * dw_tile_w;

	fill_ifms_cols: for (int w = 0; w < max_padding; w++) {
#pragma HLS UNROLL
		if (w >= padding_right) {
			break;
		}
		for (int h = 0; h < dw_tile_h; h++) {
#pragma HLS UNROLL
			if (offset_in_ifms_buffer_h + h >= last_to_fill_in_h) {
				break;
			}
			if (w + starting_fill_w_offset < ifms_width
					&& tile_index_in_w >= 0) {
				ifms_buffer[d_in_pipeline][h + offset_in_ifms_buffer_h][w
						+ offset_in_ifms_buffer_w] = channels[absolute_offset
						+ h * dw_tile_w + w];
			} else {
				ifms_buffer[d_in_pipeline][h + offset_in_ifms_buffer_h][w
						+ offset_in_ifms_buffer_w] = zero_point;
			}
		}
	}
}

void fill_ifms_rows(fms_dt channels[max_fms_size],
		fms_dt ifms_buffer[dw_pipeline_depth][dw_max_v2_buffer_height][dw_max_v2_buffer_width],
		const int offset_in_ifms_buffer_h, const int offset_in_ifms_buffer_w,
		const int last_to_fill_in_w, const int tile_index_in_h,
		const int tile_index_in_w, int absolute_offset, const int padding_top,
		const int padding_bottom, const int padding_left, const int ifms_height,
		const int ifms_width, const int d_in_pipeline,
		const fms_dt zero_point) {
#pragma HLS INLINE

	const int starting_fill_h_offset = tile_index_in_h * dw_tile_h;
	fill_ifms_rows: for (int h = 0; h < max_padding; h++) {
#pragma HLS UNROLL
		if (h >= padding_bottom) {
			break;
		}
		for (int w = 0; w < dw_tile_w; w++) {
#pragma HLS UNROLL
			if (offset_in_ifms_buffer_w + w >= last_to_fill_in_w) {
				break;
			}
			if (h + starting_fill_h_offset < ifms_height
					&& tile_index_in_h >= 0) {
				ifms_buffer[d_in_pipeline][offset_in_ifms_buffer_h + h][offset_in_ifms_buffer_w
						+ w] = channels[absolute_offset + h * dw_tile_w + w];
			} else {
				ifms_buffer[d_in_pipeline][offset_in_ifms_buffer_h + h][offset_in_ifms_buffer_w
						+ w] = zero_point;
			}
		}
	}
}

void fill_ifms_corner(fms_dt channels[max_fms_size],
		fms_dt ifms_buffer[dw_pipeline_depth][dw_max_v2_buffer_height][dw_max_v2_buffer_width],
		int offset_in_ifms_buffer_h, int offset_in_ifms_buffer_w,
		const int tile_index_in_h, const int tile_index_in_w,
		int absolute_offset, const int padding_left, const int padding_top,
		const int padding_right, const int padding_bottom,
		const int ifms_height, const int ifms_width, const int d_in_pipeline,
		const fms_dt zero_point) {
#pragma HLS INLINE

	const int starting_fill_w_offset = tile_index_in_w * dw_tile_w;
	const int starting_fill_h_offset = tile_index_in_h * dw_tile_w;

	fill_ifms_corner: for (int h = 0; h < max_padding; h++) {
#pragma HLS UNROLL
		for (int w = 0; w < max_padding; w++) {
#pragma HLS UNROLL
			if (h >= padding_bottom || w >= padding_right) {
				break;
			}
			if (h + starting_fill_h_offset < ifms_height && tile_index_in_h >= 0
					&& w + starting_fill_w_offset < ifms_width
					&& tile_index_in_w >= 0) {
				ifms_buffer[d_in_pipeline][h + offset_in_ifms_buffer_h][w
						+ offset_in_ifms_buffer_w] = channels[absolute_offset
						+ h * (max_filter_hw_dim - 1) + w];
			} else {
				ifms_buffer[d_in_pipeline][h + offset_in_ifms_buffer_h][w
						+ offset_in_ifms_buffer_w] = zero_point;
			}
		}
	}
}

void dw_fill_channels_tile(fms_dt channels[max_fms_size],
		fms_dt ifms_buffer[dw_pipeline_depth][dw_max_v2_buffer_height][dw_max_v2_buffer_width],
		const int ifms_buffer_offset_h, const int ifms_buffer_offset_w,
		const int global_absolute_tile_offset_in_ifms,
		const int num_of_tiles_hw) {
#pragma HLS INLINE off

	dw_fill_channels_tile: for (int d_in_pipeline = 0;
			d_in_pipeline < dw_pipeline_depth; d_in_pipeline++) {
		const int local_absolute_tile_offset_in_ifms =
				global_absolute_tile_offset_in_ifms
						+ d_in_pipeline * num_of_tiles_hw * dw_tile_size;
		for (int t_h = 0; t_h < dw_tile_h; t_h++) {
#pragma HLS PIPELINE
			for (int t_w = 0; t_w < dw_tile_w; t_w++) {
#pragma HLS UNROLL
				ifms_buffer[d_in_pipeline][t_h + ifms_buffer_offset_h][t_w
						+ ifms_buffer_offset_w] =
						channels[local_absolute_tile_offset_in_ifms
								+ t_h * dw_tile_w + t_w];
			}
		}
	}
}

void dw_conv_fill_from_channels(fms_dt channels[max_fms_size],
		fms_dt ifm_tile[dw_tile_h][dw_tile_w],
		fms_dt padding_top_buffer[max_padding][dw_tile_w],
		fms_dt padding_right_buffer[dw_tile_h][max_padding],
		fms_dt padding_bottom_buffer[max_padding][dw_tile_w],
		fms_dt padding_left_buffer[dw_tile_h][max_padding],
		fms_dt padding_tl_corner[max_padding * max_padding],
		fms_dt padding_tr_corner[max_padding * max_padding],
		fms_dt padding_br_corner[max_padding * max_padding],
		fms_dt padding_bl_corner[max_padding * max_padding],
		const int ifm_width, const int ifm_height, const int ifms_depth,
		const int absolute_tile_offset_in_ifms, int ifm_tile_in_h,
		int ifm_tile_in_w, int tile_offset_in_d, const int num_of_tiles_w,
		const int padding_left, const int padding_top, const int padding_right,
		const int padding_bottom, const fms_dt fms_zero_point) {
#pragma HLS INLINE off
}

void fill_dw_tile(fms_dt channels[max_fms_size],
		fms_dt ifms_buffer[dw_pipeline_depth][dw_max_v2_buffer_height][dw_max_v2_buffer_width],
		const int ifm_width, const int ifm_height, const int ifms_depth,
		const int global_absolute_tile_offset_in_ifms, int ifm_tile_in_h,
		int ifm_tile_in_w, int tile_offset_in_d, const int num_of_tiles_h,
		const int num_of_tiles_w, const int padding_left, const int padding_top,
		const int padding_right, const int padding_bottom,
		fms_dt fms_zero_point) {
#pragma HLS INLINE off

	const int num_of_tiles_hw = num_of_tiles_h * num_of_tiles_w;
	dw_fill_channels_tile(channels, ifms_buffer, padding_top, padding_left,
			global_absolute_tile_offset_in_ifms, num_of_tiles_hw);

	const int right_corners_offset_w =
			(ifm_tile_in_w == num_of_tiles_w - 1) ?
					padding_left + (ifm_width - dw_tile_w * ifm_tile_in_w) :
					padding_left + dw_tile_w;
	const int bottom_corners_offset_h =
			(ifm_tile_in_h == num_of_tiles_h - 1) ?
					padding_top + (ifm_height - dw_tile_h * ifm_tile_in_h) :
					padding_top + dw_tile_h;
	for (int d_in_pipeline = 0; d_in_pipeline < dw_pipeline_depth;
			d_in_pipeline++) {
#pragma HLS PIPELINE II=4

		const int local_absolute_tile_offset_in_ifms =
				global_absolute_tile_offset_in_ifms
						+ d_in_pipeline * num_of_tiles_hw * dw_tile_size;
		const int absolute_offset_padding_top =
				local_absolute_tile_offset_in_ifms
						- num_of_tiles_w * dw_tile_size
						+ ((dw_tile_h - padding_top) * dw_tile_w);
		const int absolute_offset_padding_right =
				local_absolute_tile_offset_in_ifms + dw_tile_size;
		const int absolute_offset_padding_bottom =
				local_absolute_tile_offset_in_ifms
						+ num_of_tiles_w * dw_tile_size;
		const int absolute_offset_padding_left =
				local_absolute_tile_offset_in_ifms - dw_tile_size
						+ (dw_tile_w - padding_left);

		const int absolute_offset_padding_tl_corner =
				absolute_offset_padding_top - dw_tile_size + dw_tile_w
						- padding_left;
		const int absolute_offset_padding_tr_corner =
				absolute_offset_padding_top + dw_tile_size;
		const int absolute_offset_padding_br_corner =
				absolute_offset_padding_bottom + dw_tile_size;
		const int absolute_offset_padding_bl_corner =
				absolute_offset_padding_bottom - dw_tile_size + dw_tile_w
						- padding_left;

		if (padding_top > 0) {
			fill_ifms_corner(channels, ifms_buffer, 0, 0, ifm_tile_in_h - 1,
					ifm_tile_in_w - 1, absolute_offset_padding_tl_corner,
					padding_left, padding_top, padding_right, padding_bottom,
					ifm_height, ifm_width, d_in_pipeline, fms_zero_point);
			fill_ifms_corner(channels, ifms_buffer, 0, right_corners_offset_w,
					ifm_tile_in_h - 1, ifm_tile_in_w + 1,
					absolute_offset_padding_tr_corner, padding_left,
					padding_top, padding_right, padding_bottom, ifm_height,
					ifm_width, d_in_pipeline, fms_zero_point);
		}
		fill_ifms_corner(channels, ifms_buffer, bottom_corners_offset_h,
				right_corners_offset_w, ifm_tile_in_h + 1, ifm_tile_in_w + 1,
				absolute_offset_padding_br_corner, padding_left, padding_top,
				padding_right, padding_bottom, ifm_height, ifm_width,
				d_in_pipeline, fms_zero_point);
		if (padding_left > 0) {
			fill_ifms_corner(channels, ifms_buffer, bottom_corners_offset_h, 0,
					ifm_tile_in_h + 1, ifm_tile_in_w - 1,
					absolute_offset_padding_bl_corner, padding_left,
					padding_top, padding_right, padding_bottom, ifm_height,
					ifm_width, d_in_pipeline, fms_zero_point);
		}

		if (padding_top > 0) {
			fill_ifms_rows(channels, ifms_buffer, 0, padding_left,
					right_corners_offset_w, ifm_tile_in_h - 1, ifm_tile_in_w,
					absolute_offset_padding_top, padding_top, padding_bottom,
					padding_left, ifm_height, ifm_width, d_in_pipeline,
					fms_zero_point);
		}
		fill_ifms_cols(channels, ifms_buffer, padding_top,
				right_corners_offset_w, bottom_corners_offset_h, ifm_tile_in_h,
				ifm_tile_in_w + 1, absolute_offset_padding_right, padding_top,
				padding_left, padding_right, ifm_height, ifm_width,
				d_in_pipeline, fms_zero_point);
		fill_ifms_rows(channels, ifms_buffer, bottom_corners_offset_h,
				padding_left, right_corners_offset_w, ifm_tile_in_h + 1,
				ifm_tile_in_w, absolute_offset_padding_bottom, padding_top,
				padding_bottom, padding_left, ifm_height, ifm_width,
				d_in_pipeline, fms_zero_point);
		if (padding_left > 0) {
			fill_ifms_cols(channels, ifms_buffer, padding_top, 0,
					bottom_corners_offset_h, ifm_tile_in_h, ifm_tile_in_w - 1,
					absolute_offset_padding_left, padding_top, padding_left,
					padding_right, ifm_height, ifm_width, d_in_pipeline,
					fms_zero_point);
		}
	}
}

void dw_conv_engine(
		dw_weights_dt weights[][max_filter_hw_dim * max_filter_hw_dim],
		fms_dt ifms_buffer[dw_pipeline_depth][dw_max_v2_buffer_height][dw_max_v2_buffer_width],
		dw_pss_dt result_tile[dw_pipeline_depth][dw_tile_h][dw_tile_w],
		const int filter_dim, const int strides) {
#pragma HLS INLINE off
	dw_conv_engine: for (int c_h = 0; c_h < max_filter_hw_dim; c_h++) {
		for (int c_w = 0; c_w < max_filter_hw_dim; c_w++) {
			for (int d_in_pipeline = 0; d_in_pipeline < dw_pipeline_depth;
					d_in_pipeline++) {
#pragma HLS PIPELINE
				for (int h = 0; h < dw_tile_h; h++) {
#pragma HLS UNROLL
					for (int w = 0; w < dw_tile_w; w++) {
#pragma HLS UNROLL
						if (c_w >= filter_dim || c_h >= filter_dim
								|| h >= dw_tile_h / strides
								|| w >= dw_tile_w / strides) {
							break;
						}
						if (c_h == 0 && c_w == 0) {
							result_tile[d_in_pipeline][h][w] =
									ifms_buffer[d_in_pipeline][h * strides + c_h][w
											* strides + c_w]
											* weights[d_in_pipeline][c_h
													* filter_dim + c_w];
						} else {
							result_tile[d_in_pipeline][h][w] +=
									ifms_buffer[d_in_pipeline][h * strides + c_h][w
											* strides + c_w]
											* weights[d_in_pipeline][c_h
													* filter_dim + c_w];
						}
					}
				}
			}
		}
	}
}

void normalize_and_write_back_result_tile(fms_dt result[max_fms_size],
		pss_dt result_tile[dw_pipeline_depth][dw_tile_h][dw_tile_w],
		fms_quantization_scheme normalization, const int layer_relu,
		const fused_scales_dt fused_scales_tile[],
		const fused_scales_log_2_shifts_dt fused_scales_log_2_shifts_tile[],
		const relu_6_fused_scales_dt relu_6_fused_scales_tile[],
		const biases_dt fused_zero_points_tile[],
		const int absolute_offset_in_results, const int num_of_tiles_hw,
		const int offset_h_in_tile, const int offset_w_in_tile,
		const int strides) {
#pragma HLS INLINE off

	for (int d_in_pipeline = 0; d_in_pipeline < dw_pipeline_depth;
			d_in_pipeline++) {
		normalization.fused_scales = fused_scales_tile[d_in_pipeline];
		normalization.fused_scales_log_2_shift =
				fused_scales_log_2_shifts_tile[d_in_pipeline];
		normalization.relu_6_fused_scale =
				relu_6_fused_scales_tile[d_in_pipeline];
		normalization.fused_zero_point = fused_zero_points_tile[d_in_pipeline];

		const int to_wrie_absolute_offset = absolute_offset_in_results
				+ d_in_pipeline * num_of_tiles_hw * dw_tile_size
				+ offset_h_in_tile * dw_tile_w + offset_w_in_tile;

		normalize_and_write_back_result_tile: for (int h = 0; h < dw_tile_h;
				h++) {
#pragma HLS PIPELINE
			for (int w = 0; w < dw_tile_w; w++) {
#pragma HLS unroll
				if (h >= dw_tile_h / strides || w >= dw_tile_w / strides) {
					break;
				}
				result[to_wrie_absolute_offset + h * dw_tile_w + w] =
						dw_relu_norm(result_tile[d_in_pipeline][h][w],
								normalization, layer_relu);
			}
		}
	}
}

void fill_dw_weights_and_scales_tiles(const dw_weights_dt weights[][3 * 3],
		dw_weights_dt weights_tile[][3 * 3],
		const fused_scales_dt fused_scales[],
		fused_scales_dt fused_scales_tile[],
		const fused_scales_log_2_shifts_dt fused_scales_log_2_shifts[],
		fused_scales_log_2_shifts_dt fused_scales_log_2_shifts_tile[],
		const relu_6_fused_scales_dt relu_6_fused_scales[],
		relu_6_fused_scales_dt relu_6_fused_scales_tile[],
		const biases_dt fused_zero_points[], biases_dt fused_zero_points_tile[],
		int starting_d, const int current_dw_layer_weights_offset,
		const int current_layer_fused_parameters_offset) {
#pragma HLS INLINE

	const int absolute_current_layer_fused_parameters_offset =
			current_layer_fused_parameters_offset + starting_d;
	const int absolute_current_layer_weights_offset =
			current_dw_layer_weights_offset + starting_d;
	for (int d = 0; d < dw_pipeline_depth; d++) {
#pragma HLS PIPELINE
		for (int i = 0; i < 3 * 3; i++) {
#pragma HLS UNROLL
			weights_tile[d][i] = weights[absolute_current_layer_weights_offset
					+ d][i];
		}
		fused_scales_tile[d] =
				fused_scales[absolute_current_layer_fused_parameters_offset + d];
		fused_scales_log_2_shifts_tile[d] =
				fused_scales_log_2_shifts[absolute_current_layer_fused_parameters_offset
						+ d];
		relu_6_fused_scales_tile[d] =
				relu_6_fused_scales[absolute_current_layer_fused_parameters_offset
						+ d];
		fused_zero_points_tile[d] =
				fused_zero_points[absolute_current_layer_fused_parameters_offset
						+ d];
	}
}

// void dw_fill_ofms_pss_tile(
// 	pss_dt src_pss_tile[dw_tile_h][dw_tile_w],
// 	pss_dt dst_pss_tile[dw_tile_h][dw_tile_w], const int offset_h, const int offset_w, const int strides)
// {
// #pragma HLS PIPELINE

// 	for (int t_h = 0; t_h < dw_tile_h / strides; t_h++)
// 	{
// #pragma HLS UNROLL
// 		for (int t_w = 0; t_w < dw_tile_w / strides; t_w++)
// 		{
// #pragma HLS UNROLL
// 			dst_pss_tile[t_h + offset_h][t_w + offset_w] = src_pss_tile[t_h * strides][t_w * strides];
// 		}
// 	}
// }

void dw_conv_3x3(const dw_weights_dt weights[][3 * 3],
		fms_dt channels[max_fms_size], fms_dt result[max_fms_size],
		const int layer, const int layer_conv_d, const int layer_ifm_width,
		const int layer_ifm_height, const int num_of_tiles_d,
		const int num_of_ofms_tiles_h, const int num_of_ofms_tiles_w,
		const int strides, const int padding_left, const int padding_right,
		const int padding_top, const int direction,
		const fused_scales_dt fused_scales[],
		const fused_scales_log_2_shifts_dt fused_scales_log_2_shifts[],
		const relu_6_fused_scales_dt relu_6_fused_scales[],
		const biases_dt fused_zero_points[],
		const fused_scales_dt fused_scales_part2[],
		const fused_scales_log_2_shifts_dt fused_scales_log_2_shifts_part2[],
		const relu_6_fused_scales_dt relu_6_fused_scales_part2[],
		const biases_dt fused_zero_points_part2[]) {
#pragma HLS INLINE off

	const int padding_bottom = padding_right;
	const int num_of_ifms_tiles_h =
			(layer_ifm_height % dw_tile_h) == 0 ?
					layer_ifm_height / dw_tile_h :
					num_of_ofms_tiles_h * strides;
	const int num_of_ifms_tiles_w =
			(layer_ifm_width % dw_tile_w) == 0 ?
					layer_ifm_width / dw_tile_w : num_of_ofms_tiles_w * strides;

	const int num_of_ifms_tiles_hw = num_of_ifms_tiles_h * num_of_ifms_tiles_w;
	const int num_of_ofms_tiles_hw = num_of_ofms_tiles_h * num_of_ofms_tiles_w;

	fms_quantization_scheme normalization = { 0, 0, 0, 0 };

	const int current_dw_layer_weights_offset = dw_layers_weights_offsets[layer];
	const int current_layer_fused_parameters_offset =
			layers_fused_parameters_offsets[layer];

	dw_weights_dt weights_tile[dw_pipeline_depth][3 * 3];
#pragma HLS ARRAY_PARTITION variable = weights_tile type = complete dim = 2

	fused_scales_dt fused_scales_tile[dw_pipeline_depth];
	fused_scales_log_2_shifts_dt fused_scales_log_2_shifts_tile[dw_pipeline_depth];
	relu_6_fused_scales_dt relu_6_fused_scales_tile[dw_pipeline_depth];
	biases_dt fused_zero_points_tile[dw_pipeline_depth];

	normalization.ofm_zero_point = conv_fms_zero_points[layer + 1];
	normalization.ofm_scale_rec = conv_fms_scales_rec[layer + 1];
	normalization.ofm_scale = conv_fms_scales[layer + 1];

	const fms_dt current_layer_fms_zero_point = conv_fms_zero_points[layer];

	fms_dt ifms_buffer[dw_pipeline_depth][dw_max_v2_buffer_height][dw_max_v2_buffer_width];
	dw_pss_dt engine_result_tile[dw_pipeline_depth][dw_tile_h][dw_tile_w];

#pragma HLS ARRAY_PARTITION variable = ifms_buffer type = complete dim = 2
#pragma HLS ARRAY_PARTITION variable = ifms_buffer type = complete dim = 3
#pragma HLS ARRAY_PARTITION variable = engine_result_tile type = complete dim = 2
#pragma HLS ARRAY_PARTITION variable = engine_result_tile type = complete dim = 3

	for (int dw_pipeline_in_d = 0;
			dw_pipeline_in_d < num_of_tiles_d / dw_pipeline_depth;
			dw_pipeline_in_d++) {
		const int tile_in_d = dw_pipeline_in_d * dw_pipeline_depth;
		if (current_layer_fused_parameters_offset
				< first_quantization_arrays_num_elements) {
			fill_dw_weights_and_scales_tiles(weights, weights_tile,
					fused_scales, fused_scales_tile, fused_scales_log_2_shifts,
					fused_scales_log_2_shifts_tile, relu_6_fused_scales,
					relu_6_fused_scales_tile, fused_zero_points,
					fused_zero_points_tile, tile_in_d * dw_tile_d,
					current_dw_layer_weights_offset,
					current_layer_fused_parameters_offset);
		} else {
			fill_dw_weights_and_scales_tiles(weights, weights_tile,
					fused_scales_part2, fused_scales_tile,
					fused_scales_log_2_shifts_part2,
					fused_scales_log_2_shifts_tile, relu_6_fused_scales_part2,
					relu_6_fused_scales_tile, fused_zero_points_part2,
					fused_zero_points_tile, tile_in_d * dw_tile_d,
					current_dw_layer_weights_offset,
					current_layer_fused_parameters_offset
							- first_quantization_arrays_num_elements);
		}

		//     if (dw_pipeline_in_d == 0)
		// {
		//     for (int d_in_pipeline = 0; d_in_pipeline < dw_pipeline_depth; d_in_pipeline++)
		//     {
		//         cout << fused_scales_tile[d_in_pipeline] << "\n";
		//         cout << (int)fused_scales_log_2_shifts_tile[d_in_pipeline] << "\n";
		//         cout << relu_6_fused_scales_tile[d_in_pipeline] << "\n";
		//         cout << fused_zero_points_tile[d_in_pipeline] << "\n";
		//         cout<<"*********\n";
		//     }
		// }
		for (int ofm_tile_in_h = 0; ofm_tile_in_h < num_of_ofms_tiles_h;
				ofm_tile_in_h++) {
			for (int ofm_tile_in_w = 0; ofm_tile_in_w < num_of_ofms_tiles_w;
					ofm_tile_in_w++) {
				for (int ifm_tile_in_ofm_tile_h = 0;
						ifm_tile_in_ofm_tile_h < strides;
						ifm_tile_in_ofm_tile_h++) {
					dw_innermost_loop: for (int ifm_tile_in_ofm_tile_w = 0;
							ifm_tile_in_ofm_tile_w < strides;
							ifm_tile_in_ofm_tile_w++) {
						const int absolute_offset_in_ofms = (tile_in_d
								* num_of_ofms_tiles_hw
								+ ofm_tile_in_h * num_of_ofms_tiles_w
								+ ofm_tile_in_w) * dw_tile_size;
						int ifm_tile_in_h = ofm_tile_in_h * strides
								+ ifm_tile_in_ofm_tile_h;
						const int ifm_tile_in_w = ofm_tile_in_w * strides
								+ ifm_tile_in_ofm_tile_w;
						const int absolute_offset_in_ifms = (tile_in_d
								* num_of_ifms_tiles_hw
								+ ifm_tile_in_h * num_of_ifms_tiles_w
								+ ifm_tile_in_w) * dw_tile_size;

						//*************************
						fill_dw_tile(channels, ifms_buffer, layer_ifm_width,
								layer_ifm_height, layer_conv_d,
								absolute_offset_in_ifms, ifm_tile_in_h,
								ifm_tile_in_w, tile_in_d * dw_tile_d,
								num_of_ifms_tiles_h, num_of_ifms_tiles_w,
								padding_left, padding_top, padding_right,
								padding_bottom, current_layer_fms_zero_point);
						dw_conv_engine(weights_tile, ifms_buffer,
								engine_result_tile, 3, strides);
						normalize_and_write_back_result_tile(result,
								engine_result_tile, normalization, 6,
								fused_scales_tile,
								fused_scales_log_2_shifts_tile,
								relu_6_fused_scales_tile,
								fused_zero_points_tile, absolute_offset_in_ofms,
								num_of_ofms_tiles_hw,
								ifm_tile_in_ofm_tile_h * dw_tile_h / strides,
								ifm_tile_in_ofm_tile_w * dw_tile_w / strides,
								strides);

						// if (tile_in_d == 0 && ifm_tile_in_h == 0 && ifm_tile_in_w == 0 && layer == 10)
						// {
						//     const int offset_h = ifm_tile_in_ofm_tile_h * dw_tile_h / strides;
						//     const int offset_w = ifm_tile_in_ofm_tile_w * dw_tile_w / strides;
						//     cout << "\n";
						//     for (int h = 0; h < dw_max_v2_buffer_height; h++)
						//     {
						//         for (int w = 0; w < dw_max_v2_buffer_width; w++)
						//         {
						//             cout << (int)ifms_buffer[0][h][w] << " ";
						//         }
						//         cout << "\n";
						//     }
						//     for (int h = 0; h < dw_tile_h / strides; h++)
						//     {
						//         for (int w = 0; w < dw_tile_w / strides; w++)
						//         {
						//             cout << (int)engine_result_tile[0][h][w] << " ";
						//             cout << (int)result[h * dw_tile_w + w] << ", ";
						//         }
						//         cout << "\n";
						//     }
						// }
						//**********************
					}
				}
			}
		}
	}
}
