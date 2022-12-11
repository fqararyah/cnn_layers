#include "../headers/layers_imp_common_includes.h"
#include "../headers/dw_conv.h"

void fill_ifms_tile_depth_col_segment(fms_dt channels[max_fms_size],
		int ifm_tile_in_h, fms_dt col_segment[dw_tile_d][dw_tile_h],
		int absolute_offset, const bool is_padding_col, const int ifms_height,
		const fms_dt zero_point) {
#pragma HLS INLINE

	const int starting_fill_h_offset = ifm_tile_in_h * dw_tile_h;
	for (int d = 0; d < dw_tile_d; d++) {
#pragma HLS UNROLL
		for (int h = 0; h < dw_tile_h; h++) {
#pragma HLS UNROLL
			// cout << absolute_offset + d * dw_tile_hw + h * dw_tile_w << "\n";
			if (!is_padding_col && h + starting_fill_h_offset < ifms_height) {
				col_segment[d][h] = channels[absolute_offset + d * dw_tile_hw
						+ h * dw_tile_w];
			} else {
				col_segment[d][h] = zero_point;
			}
		}
	}
}

void shift_col_segment(fms_dt left_col_segment[dw_tile_d][dw_tile_h],
		fms_dt ifms_buffer[dw_tile_d][dw_max_v2_buffer_height][dw_max_v2_buffer_width],
		const int use_upper_row, const int filter_dim) {
#pragma HLS INLINE

	for (int h = 0; h < dw_tile_h; h++) {
#pragma HLS UNROLL
		for (int d = 0; d < dw_tile_d; d++) {
#pragma HLS UNROLL
			left_col_segment[d][h] =
					ifms_buffer[d][h + use_upper_row][dw_max_v2_buffer_width
							- filter_dim + 1];
		}
	}
}

void padd_ifms_tile_depth_row(fms_dt row[dw_tile_d][max_dw_input_width],
		const fms_dt zero_point, const int padding_left, const int padding_right, const int layer_ifm_width) {

	// padding left
	for (int w = 0; w < padding_left; w++) {
#pragma HLS UNROLL
		for (int d = 0; d < dw_tile_d; d++) {
#pragma HLS UNROLL
			row[d][w] = zero_point;
		}
	}

	// padding right
	for (int w = 0; w < padding_right;
			w++) {
#pragma HLS UNROLL
		for (int d = 0; d < dw_tile_d; d++) {
#pragma HLS UNROLL
			row[d][w + layer_ifm_width + padding_left] = zero_point;
		}
	}

}

void fill_ifms_tile_depth_row_segment(fms_dt channels[max_fms_size],
		fms_dt row[dw_tile_d][max_dw_input_width], const int tile_index_in_w,
		int absolute_offset, const int padding_left, const int padding_right,
		const bool is_padding_row, const int ifms_width,
		const fms_dt zero_point) {
#pragma HLS INLINE

	const int starting_fill_w_offset = tile_index_in_w * dw_tile_w;
	for (int w = 0; w < dw_tile_w; w++) {
#pragma HLS UNROLL
		for (int d = 0; d < dw_tile_d; d++) {
#pragma HLS UNROLL
			if (!is_padding_row && w + starting_fill_w_offset < ifms_width) {
				row[d][w + starting_fill_w_offset + padding_left] =
						channels[absolute_offset + d * dw_tile_hw + w];
//				if (d == 0)
//					cout << w + starting_fill_w_offset + padding_left <<", " << absolute_offset <<" and "<<absolute_offset + d * dw_tile_hw + w
//							<< " >>>> "
//							<< channels[absolute_offset + d * dw_tile_hw + w]
//							<< "\n";
			} else {
				row[d][w + starting_fill_w_offset + padding_left] = zero_point;
			}
		}
	}
}

void shift_segment_between_row_and_buffer(
		fms_dt upper_row[dw_tile_d][max_dw_input_width],
		fms_dt ifms_buffer[dw_tile_d][dw_max_v2_buffer_height][dw_max_v2_buffer_width],
		const int tile_index_in_w, const int padding_left,
		const int filter_dim) {
#pragma HLS INLINE

	const int starting_shift_index = tile_index_in_w * dw_tile_w + padding_left;
	for (int w = 0; w < dw_tile_w; w++) {
#pragma HLS UNROLL
		for (int d = 0; d < dw_tile_d; d++) {
#pragma HLS UNROLL
			upper_row[d][w + starting_shift_index] =
					ifms_buffer[d][dw_max_v2_buffer_height - filter_dim + 1][w
							+ padding_left];
		}
	}
}

void fill_ifms_tile_depth_row(fms_dt channels[max_fms_size],
		fms_dt ifms_row[dw_tile_d][max_dw_input_width], const int ifms_width,
		const int num_of_tiles_w, const int num_of_tiles_hw,
		const int ifms_height, int starting_tile_in_d, int row_index,
		const int padding_left, const int padding_right,
		const bool is_padding_row, const fms_dt zero_point) {
	const int h_offset = num_of_tiles_w * (row_index / dw_tile_h) * dw_tile_size
			+ (row_index % dw_tile_h) * dw_tile_w;
	int absolute_offset = starting_tile_in_d * num_of_tiles_hw * dw_tile_size
			+ h_offset;

	fill_row: for (int tile_in_w = 0; tile_in_w < num_of_tiles_w; tile_in_w++) {
		fill_ifms_tile_depth_row_segment(channels, ifms_row, tile_in_w,
				absolute_offset, padding_left, padding_right, is_padding_row,
				ifms_width, zero_point);
		absolute_offset += dw_tile_size;
	}
}

void dw_fill_ifms_buffer(fms_dt channels[max_fms_size],
		fms_dt ifms_buffer[dw_tile_d][dw_max_v2_buffer_height][dw_max_v2_buffer_width],
		fms_dt upper_row[dw_tile_d][max_dw_input_width],
		fms_dt lower_row[dw_tile_d][max_dw_input_width],
		fms_dt left_col_segment[dw_tile_d][dw_tile_h],
		fms_dt right_col_segment[dw_tile_d][dw_tile_h], int absolute_offset,
		const int ifms_height, const int ifms_width, int ifms_tile_in_w,
		int ifms_tile_in_h, const int use_left_col, const int use_upper_row,
		const int filter_dim) {
#pragma HLS INLINE

	const int starting_fill_w_from_row_offset = ifms_tile_in_w * dw_tile_w;
	const int col_to_fill_from = dw_max_v2_buffer_width - filter_dim + 1;
	const int last_row_to_fill = (dw_max_v2_buffer_height - 1) - 1
			+ use_upper_row;
	const int last_col_to_fill = (dw_max_v2_buffer_width - 1) - 1
			+ use_left_col;
	// fill upper row
	if (use_upper_row) {
		for (int d = 0; d < dw_tile_d; d++) {
#pragma HLS UNROLL
			if (starting_fill_w_from_row_offset == 0) {
				ifms_buffer[d][0][0] = upper_row[d][0];
			} else {
				ifms_buffer[d][0][0] = ifms_buffer[d][0][col_to_fill_from];
			}
			for (int w = use_left_col; w < dw_max_v2_buffer_width; w++) {
#pragma HLS UNROLL
				ifms_buffer[d][0][w] =
						upper_row[d][starting_fill_w_from_row_offset + w];
			}
		}
	}

	// fill lower row
	for (int d = 0; d < dw_tile_d; d++) {
#pragma HLS UNROLL
		if (starting_fill_w_from_row_offset == 0) {
			ifms_buffer[d][last_row_to_fill][0] = lower_row[d][0];
		} else {
			ifms_buffer[d][last_row_to_fill][0] =
					ifms_buffer[d][last_row_to_fill][col_to_fill_from];
		}
		for (int w = use_left_col; w < dw_max_v2_buffer_width; w++) {
#pragma HLS UNROLL
			ifms_buffer[d][last_row_to_fill][w] =
					lower_row[d][starting_fill_w_from_row_offset + w]; //-1 is padding top
//			if (d == 0)
//				cout << starting_fill_w_from_row_offset + w << " >> "
//						<< lower_row[d][starting_fill_w_from_row_offset + w]
//						<< "\n";
		}
	}

	// fill left col
	if (use_left_col) {
		for (int d = 0; d < dw_tile_d; d++) {
#pragma HLS UNROLL
			for (int h = 0; h < dw_tile_h; h++) {
#pragma HLS UNROLL
				ifms_buffer[d][h + use_upper_row][0] = left_col_segment[d][h];
			}
		}
	}

	// fill right col
	for (int d = 0; d < dw_tile_d; d++) {
#pragma HLS UNROLL
		for (int h = 0; h < dw_tile_h; h++) {
#pragma HLS UNROLL
			ifms_buffer[d][h + use_upper_row][last_col_to_fill] =
					right_col_segment[d][h]; //-1 is padding left
		}
	}

	// fill body
	for (int d = 0; d < dw_tile_d; d++) {
#pragma HLS UNROLL
		for (int h = 0; h < dw_tile_h; h++) {
#pragma HLS UNROLL
			for (int w = 0; w < dw_tile_w; w++) {
#pragma HLS UNROLL
				ifms_buffer[d][h + use_upper_row][w + use_left_col] =
						channels[absolute_offset + d * dw_tile_hw
								+ h * dw_tile_w + w];
			}
		}
	}

	if (ifms_tile_in_w == 13) {
		for (int h = 0; h < dw_max_v2_buffer_height; h++) {
			for (int w = 0; w < dw_max_v2_buffer_width; w++) {
				cout << ifms_buffer[0][h][w] << " ";
			}
			cout << "\n";
		}
		cout << "*******";
	}
}

void dw_conv_engine(
		const dw_weights_dt weights[max_conv_d][max_conv_h * max_conv_w],
		fms_dt ifms_buffer[dw_tile_d][dw_max_v2_buffer_height][dw_max_v2_buffer_width],
		fms_dt result[max_fms_size], int w_offset_in_tile,
		const int tile_d_offset, int absolute_offset_in_ofms,
		int ifm_tile_in_ofm_tile_h, int ifm_tile_in_ofm_tile_w,
		const int filter_dim, const int strides,
		fms_quantization_scheme normalization, const int layer_relu,
		fused_scales_dt fused_scales[],
		fused_scales_log_2_shifts_dt fused_scales_log_2_shifts[],
		relu_6_fused_scales_dt relu_6_fused_scales[],
		biases_dt fused_zero_points[]) {
#pragma HLS INLINE

	const int ifm_tile_offset_in_ofm_tile_h = ifm_tile_in_ofm_tile_h
			* (dw_tile_h / strides);
	const int ifm_tile_offset_in_ofm_tile_w = ifm_tile_in_ofm_tile_w
			* (dw_tile_w / strides);
	const int start_writing_index = absolute_offset_in_ofms
			+ ifm_tile_offset_in_ofm_tile_h * dw_tile_w + w_offset_in_tile
			+ ifm_tile_offset_in_ofm_tile_w;

	dw_engine_loops: for (int d = 0; d < dw_tile_d; d++) {
#pragma HLS UNROLL
		normalization.fused_scales = fused_scales[tile_d_offset + d];
		normalization.fused_scales_log_2_shift =
				fused_scales_log_2_shifts[tile_d_offset + d];
		normalization.relu_6_fused_scale =
				relu_6_fused_scales[tile_d_offset + d];
		normalization.fused_zero_point = fused_zero_points[tile_d_offset + d];
		for (int h = 0; h < dw_tile_h / strides; h++) {
#pragma HLS UNROLL
			dw_pss_dt tmp_pss = 0;
			for (int c_h = 0; c_h < filter_dim; c_h++) {
#pragma HLS UNROLL
				for (int c_w = 0; c_w < filter_dim; c_w++) {
#pragma HLS UNROLL
					tmp_pss +=
							weights[tile_d_offset + d][c_h * filter_dim + c_w]
									* ifms_buffer[d][h * strides + c_h][w_offset_in_tile
											* strides + c_w];
//					if (absolute_offset_in_ofms == 0 && h == 0 && d == 0) {
//						cout << weights[tile_d_offset][c_h * filter_dim + c_w]
//								<< " "
//								<< ifms_buffer[d][h * strides + c_h][w_offset_in_tile
//										* strides + c_w] << "\n";
//					}
				}
			}
			result[start_writing_index + d * dw_tile_hw + h * dw_tile_w] =
					dw_relu_norm(tmp_pss, normalization, layer_relu);
		}
	}
}

void dw_conv_pipeline(fms_dt channels[max_fms_size],
		const dw_weights_dt weights[max_conv_d][max_conv_h * max_conv_w],
		fms_dt result[max_fms_size],
		fms_dt upper_row[dw_tile_d][max_dw_input_width],
		fms_dt lower_row[dw_tile_d][max_dw_input_width],
		fms_dt ifms_buffer[dw_tile_d][dw_max_v2_buffer_height][dw_max_v2_buffer_width],
		fms_dt col_segment_left[dw_tile_d][dw_tile_h],
		fms_dt col_segment_right[dw_tile_d][dw_tile_h], const int filter_dim,
		const int strides, const int padding_left, const int padding_right,
		const int padding_top, const int ifm_width, const int ifm_height,
		int ifm_tile_in_h, int ifm_tile_in_w, const int tile_d_offset,
		const int num_of_ifms_tiles_w, const int num_of_ifms_tiles_h,
		int absolute_offset_in_ifms, int absolute_offset_in_ofms,
		int ifm_tile_in_ofm_tile_h, int ifm_tile_in_ofm_tile_w,
		const fms_dt fms_zero_point, fms_quantization_scheme normalization,
		const int layer_relu, fused_scales_dt fused_scales[],
		fused_scales_log_2_shifts_dt fused_scales_log_2_shifts[],
		relu_6_fused_scales_dt relu_6_fused_scales[],
		biases_dt fused_zero_points[]) {
#pragma HLS INLINE

	fill_ifms_tile_depth_col_segment(channels, ifm_tile_in_h, col_segment_right,
			absolute_offset_in_ifms + dw_tile_size,
			ifm_tile_in_w == num_of_ifms_tiles_w - 1, ifm_height,
			fms_zero_point);

	bool is_padding_row = ifm_tile_in_h == num_of_ifms_tiles_h - 1;
	if ((ifm_tile_in_w + 1) < num_of_ifms_tiles_w) {
		fill_ifms_tile_depth_row_segment(channels, lower_row, ifm_tile_in_w + 1,
				absolute_offset_in_ifms + (num_of_ifms_tiles_w * dw_tile_size)
						+ dw_tile_size, padding_left, padding_right,
				is_padding_row, ifm_width, fms_zero_point);
	}

	dw_fill_ifms_buffer(channels, ifms_buffer, upper_row, lower_row,
			col_segment_left, col_segment_right, absolute_offset_in_ifms,
			ifm_height, ifm_width, ifm_tile_in_w, ifm_tile_in_h, strides == 1,
			strides == 1, filter_dim);

	for (int w = 0; w < dw_tile_w / strides; w++) {
#pragma HLS PIPELINE
		dw_conv_engine(weights, ifms_buffer, result, w, tile_d_offset,
				absolute_offset_in_ofms, ifm_tile_in_ofm_tile_h,
				ifm_tile_in_ofm_tile_w, filter_dim, strides, normalization,
				layer_relu, fused_scales, fused_scales_log_2_shifts,
				relu_6_fused_scales, fused_zero_points);
	}

	if (strides == 1) {
		shift_col_segment(col_segment_left, ifms_buffer, strides == 1,
				filter_dim);
		shift_segment_between_row_and_buffer(upper_row, ifms_buffer,
				ifm_tile_in_w, padding_left, filter_dim);
	}
}

void dw_conv_3x3(dw_weights_dt weights[max_conv_d][3 * 3],
		fms_dt channels[max_fms_size], fms_dt result[max_fms_size],
		const int layer, const int layer_conv_d, const int layer_ifm_width,
		const int layer_ifm_height, const int num_of_tiles_d,
		const int num_of_ofms_tiles_h, const int num_of_ofms_tiles_w,
		const int strides, const int padding_left, const int padding_right,
		const int padding_top, const int direction,
		fused_scales_dt fused_scales[],
		fused_scales_log_2_shifts_dt fused_scales_log_2_shifts[],
		relu_6_fused_scales_dt relu_6_fused_scales[],
		biases_dt fused_zero_points[]) {
#pragma HLS INLINE off

	fms_dt upper_row[dw_tile_d][max_dw_input_width];
	fms_dt lower_row[dw_tile_d][max_dw_input_width];

#pragma HLS ARRAY_PARTITION variable=upper_row type=complete dim=0
#pragma HLS ARRAY_PARTITION variable=lower_row type=complete dim=0

	fms_dt col_segment_left[dw_tile_d][dw_tile_h];
	fms_dt col_segment_right[dw_tile_d][dw_tile_h];
#pragma HLS ARRAY_PARTITION variable=col_segment_left type=complete dim=0
#pragma HLS ARRAY_PARTITION variable=col_segment_right type=complete dim=0

	const int num_of_ifms_tiles_h = num_of_ofms_tiles_h * strides;
	const int num_of_ifms_tiles_w = num_of_ofms_tiles_w * strides;

	const int num_of_ifms_tiles_hw = num_of_ifms_tiles_h * num_of_ifms_tiles_w;
	const int num_of_ofms_tiles_hw = num_of_ofms_tiles_h * num_of_ofms_tiles_w;

	fms_quantization_scheme normalization = { 0, 0, 0, 0 };

	normalization.ofm_zero_point = conv_fms_zero_points[layer + 1];
	normalization.ofm_scale_rec = conv_fms_scales_rec[layer + 1];
	normalization.ofm_scale = conv_fms_scales[layer + 1];

	const fms_dt current_layer_fms_zero_point = conv_fms_zero_points[layer];

	padd_ifms_tile_depth_row(upper_row, current_layer_fms_zero_point, padding_left, padding_right, layer_ifm_width);
	padd_ifms_tile_depth_row(lower_row, current_layer_fms_zero_point, padding_left, padding_right, layer_ifm_width);

	for (int tile_in_d = 0; tile_in_d < num_of_tiles_d; tile_in_d++) {
		fill_ifms_tile_depth_row(channels, upper_row, layer_ifm_width,
				num_of_ifms_tiles_w, num_of_ifms_tiles_hw, layer_ifm_height,
				tile_in_d, 0, padding_left, 1, padding_top != 0,
				current_layer_fms_zero_point);
		for (int ofm_tile_in_h = 0; ofm_tile_in_h < num_of_ofms_tiles_h;
				ofm_tile_in_h++) {
			int absolute_offset_in_ofms = tile_in_d * num_of_ofms_tiles_hw
					* dw_tile_size
					+ ofm_tile_in_h * num_of_ofms_tiles_w * dw_tile_size;
			for (int ifm_tile_in_ofm_tile_h = 0;
					ifm_tile_in_ofm_tile_h < strides;
					ifm_tile_in_ofm_tile_h++) {
				int ifm_tile_in_h = ofm_tile_in_h * strides
						+ ifm_tile_in_ofm_tile_h;
				// cout << "\n"<<tile_in_d << " * " << num_of_ifms_tiles_hw << " * " << dw_tile_size << " + (" <<ofm_tile_in_h << " + " <<ifm_tile_in_ofm_tile_h << ") *"
				//      << num_of_ifms_tiles_w << " * " << dw_tile_size<<"\n";
				int absolute_offset_in_ifms = tile_in_d * num_of_ifms_tiles_hw
						* dw_tile_size
						+ (ofm_tile_in_h * strides + ifm_tile_in_ofm_tile_h)
								* num_of_ifms_tiles_w * dw_tile_size;
				fill_ifms_tile_depth_col_segment(channels, 0, col_segment_left,
						absolute_offset_in_ifms, true, layer_ifm_height,
						current_layer_fms_zero_point);

				bool is_padding_row = ifm_tile_in_h
						== (num_of_ifms_tiles_h - 1);
				fill_ifms_tile_depth_row_segment(channels, lower_row, 0,
						absolute_offset_in_ifms
								+ (num_of_ifms_tiles_w * dw_tile_size),
						padding_left, padding_right, is_padding_row,
						layer_ifm_width, current_layer_fms_zero_point);

				for (int ofm_tile_in_w = 0; ofm_tile_in_w < num_of_ofms_tiles_w;
						ofm_tile_in_w++) {
					for (int ifm_tile_in_ofm_tile_w = 0;
							ifm_tile_in_ofm_tile_w < strides;
							ifm_tile_in_ofm_tile_w++) {
						int ifm_tile_in_w = ofm_tile_in_w * strides
								+ ifm_tile_in_ofm_tile_w;
						fms_dt ifms_buffer[dw_tile_d][dw_max_v2_buffer_height][dw_max_v2_buffer_width];

#pragma HLS ARRAY_PARTITION variable=ifms_buffer type=complete dim=0

						dw_conv_pipeline(channels, weights, result, upper_row,
								lower_row, ifms_buffer, col_segment_left,
								col_segment_right, 3, strides, padding_left,
								padding_right, padding_top, layer_ifm_width,
								layer_ifm_height, ifm_tile_in_h, ifm_tile_in_w,
								tile_in_d * dw_tile_d, num_of_ifms_tiles_w,
								num_of_ifms_tiles_h, absolute_offset_in_ifms,
								absolute_offset_in_ofms
										+ (ofm_tile_in_w * dw_tile_size),
								ifm_tile_in_ofm_tile_h, ifm_tile_in_ofm_tile_w,
								current_layer_fms_zero_point, normalization, 6,
								fused_scales, fused_scales_log_2_shifts,
								relu_6_fused_scales, fused_zero_points);
						absolute_offset_in_ifms += dw_tile_size;
					}
				}
			}
		}
	}
}
