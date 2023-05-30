#include "pipeline_main.h"

void fill_input_image_groups_buffer(
		fms_grp_dt channels[input_image_depth
				* input_image_num_fms_groups_in_a_channel],
		fms_grp_dt fms_groups_buffer[input_image_depth][input_image_num_fms_groups_in_width
				* INPUT_IMAGE_ROWS_FILLED_EACH_TIME], const int starting_h,
		const int elements_to_fill_from_an_ifm) { // chain_0_1_layer_0_s_in_rows_at_once * input_image_num_fms_groups_in_width
#pragma HLS INLINE off

	const int start_filling_offset = starting_h
			* input_image_num_fms_groups_in_width;
	int elements_avaiable_in_input_image;

	elements_avaiable_in_input_image = (input_image_height - starting_h)
			* input_image_num_fms_groups_in_width;

	for (int d = 0; d < input_image_depth; d++) {
#pragma HLS PIPELINE off
		const int d_offst = start_filling_offset
				+ d * input_image_num_fms_groups_in_a_channel;
		for (int i = 0; i < elements_to_fill_from_an_ifm; i++) {
#pragma HLS PIPELINE off
			if (i < elements_avaiable_in_input_image) {
				fms_groups_buffer[d][i] = channels[d_offst + i];
			}
		}
	}
}

void input_image_fill_row_from_groups_buffer(
		fms_grp_dt fms_groups_buffer[input_image_depth][input_image_num_fms_groups_in_width
				* INPUT_IMAGE_ROWS_FILLED_EACH_TIME],
		fms_dt channels_buffer_0[input_image_depth][PRE_FIRST_PIPELINE_INPUT_HEIGHT][input_image_width],
		int row, const int channels_buffer_start_filling_h) {
#pragma HLS INLINE

	const int start_filling_offset = row * input_image_num_fms_groups_in_width;
	xxx_loop: for (int o_w = 0; o_w < input_image_num_fms_groups_in_width;
			o_w++) {
#pragma HLS PIPELINE off
		const int o_w_offset = o_w * input_image_group_items;
		for (int d = 0; d < input_image_depth; d++) {
#pragma HLS UNROLL
			fms_grp_dt chunck = fms_groups_buffer[d][start_filling_offset + o_w];
			for (int w = 0; w < input_image_group_items; w++)
				xxy_loop: {
#pragma HLS PIPELINE off
					if (o_w_offset + w < input_image_width) {
#if HW == _FPGA
						if (channels_buffer_start_filling_h + row
								< PRE_FIRST_PIPELINE_INPUT_HEIGHT) {
							channels_buffer_0[d][channels_buffer_start_filling_h
									+ row][o_w_offset + w] = (fms_dt) chunck(
									w * fms_dt_width + fms_dt_offset,
									w * fms_dt_width);
						} else {
							channels_buffer_0[d][channels_buffer_start_filling_h
									+ row - PRE_FIRST_PIPELINE_INPUT_HEIGHT][o_w_offset
									+ w] = (fms_dt) chunck(
									w * fms_dt_width + fms_dt_offset,
									w * fms_dt_width);
						}
#endif
					}
				}
		}
	}
}

// void input_image_shift_channels_buffer_rows(
//     fms_dt channels_buffer_0[input_image_depth][PRE_FIRST_PIPELINE_INPUT_HEIGHT][input_image_width],
//     const int rows_to_shift)
// {
// #pragma HLS INLINE

//     for (int w = 0; w < input_image_width; w++)
//     {
// #pragma HLS PIPELINE
//         for (int d = 0; d < input_image_depth; d++)
//         {
// #pragma HLS UNROLL
//             for (int h = 0; h < rows_to_shift; h++)
//             {
// #pragma HLS UNROLL
//                 channels_buffer_0[d][h][w] = channels_buffer_0[d][h + INPUT_IMAGE_ROWS_FILLED_EACH_TIME][w];
//             }
//         }
//     }
// }

void input_image_padd_bottom_channels_buffer_rows(
		fms_dt channels_buffer_0[input_image_depth][PRE_FIRST_PIPELINE_INPUT_HEIGHT][input_image_width],
		const int padding_starting_h, const fms_dt zero_point) {
#pragma HLS INLINE

	for (int w = 0; w < input_image_width; w++) {
#pragma HLS PIPELINE
		for (int d = 0; d < input_image_depth; d++) {
#pragma HLS UNROLL
			for (int h = 0; h < first_conv_layer_padding_right; h++) // right is = bottom
					{
#pragma HLS UNROLL
				if (h + padding_starting_h < PRE_FIRST_PIPELINE_INPUT_HEIGHT) {
					channels_buffer_0[d][h + padding_starting_h][w] =
							zero_point;
				} else {
					channels_buffer_0[d][h + padding_starting_h
							- PRE_FIRST_PIPELINE_INPUT_HEIGHT][w] = zero_point;
				}
			}
		}
	}
}

void input_image_fill_channels_buffer_from_groups_buffer(
		fms_grp_dt fms_groups_buffer[input_image_depth][input_image_num_fms_groups_in_width
				* INPUT_IMAGE_ROWS_FILLED_EACH_TIME],
		fms_dt channels_buffer_0[input_image_depth][PRE_FIRST_PIPELINE_INPUT_HEIGHT][input_image_width],
		const int starting_h, const int channels_buffer_start_filling_h,
		const fms_dt zero_point) {
#pragma HLS INLINE off

//    const int rows_to_shift = PRE_FIRST_PIPELINE_INPUT_HEIGHT - INPUT_IMAGE_ROWS_FILLED_EACH_TIME;
//    // if (shift)
//    // {
//    //     input_image_shift_channels_buffer_rows(channels_buffer_0, rows_to_shift);
//    // }
//    // const int channels_buffer_start_filling_h =
//    //     starting_h == 0 ? first_conv_layer_specs.padding_top : rows_to_shift;
//    for (int h = 0; h < INPUT_IMAGE_ROWS_FILLED_EACH_TIME; h++)
//    {
//        if (starting_h + h < input_image_height)
//        {
//            input_image_fill_row_from_groups_buffer(fms_groups_buffer,
//                                                    channels_buffer_0, h, channels_buffer_start_filling_h);
//        }
//        else
//        {
//            input_image_padd_bottom_channels_buffer_rows(channels_buffer_0, channels_buffer_start_filling_h + h,
//                                                         zero_point);
//        }
//    }
}

void fill_first_cols_of_first_layer_input(
		fms_dt channels_buffer_0[input_image_depth][PRE_FIRST_PIPELINE_INPUT_HEIGHT][input_image_width],
		fms_dt first_layer_input_buffer[FIRST_CONV_LAYER_BUFFER_SIZE],
		int starting_h) {
#pragma HLS INLINE off

	const int first_layer_input_buffer_hw = first_conv_layer_filter_dim
			* first_conv_layer_filter_dim;
	const int filter_dim = first_conv_layer_specs.filter_size;
	const int strides = first_conv_layer_specs.strides;
	const int filter_dim_minus_strides = filter_dim - strides;

	fill_first_cols_of_first_layer_input: for (int d = 0; d < input_image_depth;
			d++) {
		for (int h = 0; h < PRE_FIRST_PIPELINE_INPUT_HEIGHT; h++) {
			for (int w = 0; w < MAX_FILTER_MINUS_STRIDES; w++) {
				if (w < filter_dim_minus_strides) {
					if (starting_h + h < PRE_FIRST_PIPELINE_INPUT_HEIGHT) {
						first_layer_input_buffer[d * first_layer_input_buffer_hw
								+ h * first_conv_layer_filter_dim + w
								+ (filter_dim - filter_dim_minus_strides)] =
								channels_buffer_0[d][starting_h + h][w];
					} else {
						first_layer_input_buffer[d * first_layer_input_buffer_hw
								+ h * first_conv_layer_filter_dim + w
								+ (filter_dim - filter_dim_minus_strides)] =
								channels_buffer_0[d][starting_h + h
										- PRE_FIRST_PIPELINE_INPUT_HEIGHT][w];
					}
				}
			}
		}
	}
}

void fill_first_layer_input_new_cols(
		fms_dt channels_buffer_0[input_image_depth][PRE_FIRST_PIPELINE_INPUT_HEIGHT][input_image_width],
		fms_dt first_layer_input_new_cols[FIRST_CONV_LAYER_NEW_COLS_BUFFER_SIZE],
		const int starting_w, int starting_h, fms_dt zero_point) {
#pragma HLS INLINE off

	const int buffer_hw = first_conv_layer_filter_dim
			* first_conv_layer_strides;
	for (int d = 0; d < input_image_depth; d++) {
		for (int h = 0; h < first_conv_layer_filter_dim; h++) {
			for (int w = 0; w < first_conv_layer_strides; w++) {
				if (h + starting_h < PRE_FIRST_PIPELINE_INPUT_HEIGHT) {
					if (starting_w + w < input_image_width) {
						first_layer_input_new_cols[d * buffer_hw
								+ h * first_conv_layer_strides + w] =
								channels_buffer_0[d][h + starting_h][starting_w
										+ w];
					} else {
						first_layer_input_new_cols[d * buffer_hw
								+ h * first_conv_layer_strides + w] =
								zero_point;
					}
				} else {
					if (starting_w + w
							< input_image_width
									+ first_conv_layer_padding_left) {
						first_layer_input_new_cols[d * buffer_hw
								+ h * first_conv_layer_strides + w] =
								channels_buffer_0[d][h + starting_h
										- PRE_FIRST_PIPELINE_INPUT_HEIGHT][starting_w
										+ w];
					} else {
						first_layer_input_new_cols[d * buffer_hw
								+ h * first_conv_layer_strides + w] =
								zero_point;
					}
				}
			}
		}
	}
}

void shift_and_fill_first_layer_input(
		fms_dt first_layer_input_new_cols[FIRST_CONV_LAYER_NEW_COLS_BUFFER_SIZE],
		fms_dt first_layer_input[FIRST_CONV_LAYER_BUFFER_SIZE]) {
#pragma HLS INLINE off

	const int to_be_shifted = first_conv_layer_filter_dim
			- first_conv_layer_strides;
	const int first_layer_input_buffer_hw = first_conv_layer_filter_dim
			* first_conv_layer_filter_dim;
	const int first_layer_input_buffer_cols_hw = first_conv_layer_filter_dim
			* first_conv_layer_strides;

	fill_bottleneck_0_input: for (int d = 0; d < input_image_depth; d++) {
#pragma HLS UNROLL
		for (int h = 0; h < first_conv_layer_filter_dim; h++) {
#pragma HLS UNROLL
			for (int w = 0; w < to_be_shifted; w++) {
#pragma HLS UNROLL
				const int to_fill_in_index = d * first_layer_input_buffer_hw
						+ h * first_conv_layer_filter_dim + w;
				first_layer_input[to_fill_in_index] =
						first_layer_input[to_fill_in_index
								+ first_conv_layer_strides];
			}
			for (int w = 0; w < first_conv_layer_strides; w++) {
#pragma HLS UNROLL
				first_layer_input[d * first_layer_input_buffer_hw
						+ h * first_conv_layer_filter_dim + w + to_be_shifted] =
						first_layer_input_new_cols[d
								* first_layer_input_buffer_cols_hw
								+ h * first_conv_layer_strides + w];
			}
		}
	}
}

void input_image_fill_channels_buffer_cpu(
		fms_dt channels[input_image_depth * input_image_hw],
		fms_dt channels_buffer_0[input_image_depth][PRE_FIRST_PIPELINE_INPUT_HEIGHT][input_image_width],
		int starting_h, const int channels_buffer_start_filling_h,
		const fms_dt zero_point) {
#pragma HLS INLINE off

	const int rows_to_shift = FIRST_CONV_LAYER_EXTRA_ROWS_FILLED_FIRST_TIME;
	// if (shift)
	// {
	//     input_image_shift_channels_buffer_rows(channels_buffer_0, rows_to_shift);
	// }
	// const int channels_buffer_start_filling_h =
	//     starting_h == 0 ? first_conv_layer_specs.padding_top : rows_to_shift;
	for (int h = 0; h < INPUT_IMAGE_ROWS_FILLED_EACH_TIME; h++) {
		if (starting_h + h < input_image_height) {
			for (int d = 0; d < input_image_depth; d++) {
				for (int w = 0; w < input_image_width; w++) {
					if (h + channels_buffer_start_filling_h
							< PRE_FIRST_PIPELINE_INPUT_HEIGHT) {
						channels_buffer_0[d][h + channels_buffer_start_filling_h][w] =
								channels[d * input_image_hw
										+ (h + starting_h) * input_image_width
										+ w];
					} else {
						channels_buffer_0[d][h + channels_buffer_start_filling_h
								- PRE_FIRST_PIPELINE_INPUT_HEIGHT][w] =
								channels[d * input_image_hw
										+ (h + starting_h) * input_image_width
										+ w];
					}
				}
			}
		} else {
			input_image_padd_bottom_channels_buffer_rows(channels_buffer_0,
					channels_buffer_start_filling_h + h, zero_point);
		}
	}
}

void shift_slice_of_conv_dw_communication_buffer_intra(
		fms_dt conv_dw_communication_buffer_intra[first_conv_layer_num_fils][layer_2_dw_filter_dim][layer_2_dw_filter_dim],
		const int d) {
	const int strides = layer_2_dw_specs.strides;
	const int filter_dim_minus_strides = layer_2_dw_filter_dim - strides;

	for (int h = 0; h < layer_2_dw_filter_dim; h++) {
		for (int w = 0; w < filter_dim_minus_strides; w++) {
			conv_dw_communication_buffer_intra[d][h][w] =
					conv_dw_communication_buffer_intra[d][h][w + strides];
		}
	}
}

void first_conv_and_dw_layers_pipeline(
		fms_dt first_layer_input[FIRST_CONV_LAYER_BUFFER_SIZE],
		weights_dt dw_layer_weights[layer_2_dw_num_fils][layer_2_dw_filter_dim
				* layer_2_dw_filter_dim],
		fms_dt conv_dw_communication_buffer_inter[first_conv_layer_num_fils][layer_2_dw_filter_dim][layer_2_dw_ifm_width],
		fms_dt conv_dw_communication_buffer_intra[first_conv_layer_num_fils][layer_2_dw_filter_dim][layer_2_dw_filter_dim],
		fms_dt pre_first_pipeline_layers_output[PRE_FIRST_PIPELINE_OUTPUT_DEPTH][PRE_FIRST_PIPELINE_OUTPUT_HEIGHT][PRE_FIRST_PIPELINE_OUTPUT_WIDTH],
		const int abs_conv_starting_h, const int starting_h,
		const int starting_w, const int conv_dw_comm_buffer_writing_row,
		fms_quantization_scheme first_conv_layer_quantization_params[first_conv_layer_num_fils],
		fms_quantization_scheme first_dw_layer_quantization_params[layer_2_dw_num_fils]) {
#pragma HLS INLINE off

	int dw_starting_h = starting_h;
	if (abs_conv_starting_h
			< layer_2_dw_filter_dim - layer_2_dw_specs.strides) {
		dw_starting_h = starting_h
				- (layer_2_dw_filter_dim - layer_2_dw_specs.padding_top - 1);
	}
	const int dw_starting_w = starting_w
			- (layer_2_dw_filter_dim - layer_2_dw_specs.padding_left);

	for (int f = 0; f < first_conv_layer_num_fils; f++) {
#pragma HLS PIPELINE
		if (f > 0) {
			shift_slice_of_conv_dw_communication_buffer_intra(
					conv_dw_communication_buffer_intra, f - 1);
		}
		if (starting_w < layer_2_dw_ifm_width) {
			conv_dw_communication_buffer_inter[f][conv_dw_comm_buffer_writing_row][starting_w] =
					first_layer_conv_kernel(first_layer_input,
							first_layer_weights, f, starting_h, starting_w,
							first_conv_layer_quantization_params[f]);
		}
		if (dw_starting_h >= 0 && dw_starting_w >= 0) {
			pre_first_pipeline_layers_output[f][dw_starting_h][dw_starting_w] =
					first_dw_layer_kernel(conv_dw_communication_buffer_intra,
							dw_layer_weights, layer_2_dw_filter_dim, f,
							first_dw_layer_quantization_params[f],
							layer_2_dw_specs.layer_activation);
		}
	}
	shift_slice_of_conv_dw_communication_buffer_intra(
			conv_dw_communication_buffer_intra, first_conv_layer_num_fils - 1);
}

void fill_conv_dw_communication_buffer_intra_first_time(
		fms_dt conv_dw_communication_buffer_inter[first_conv_layer_num_fils][layer_2_dw_filter_dim][layer_2_dw_ifm_width],
		fms_dt conv_dw_communication_buffer_intra[first_conv_layer_num_fils][layer_2_dw_filter_dim][layer_2_dw_filter_dim],
		const int starting_h, const layer_specs specs_struct) {

	const int filter_dim = specs_struct.filter_size;
	const int ifms_depth = specs_struct.layer_depth;
	const int ifms_width = specs_struct.layer_ifm_width;
	const int padding_top = specs_struct.padding_top;
	const int padding_left = specs_struct.padding_left;
	const fms_dt layer_ifm_zero_point = specs_struct.layer_ifms_zero_point;

	for (int d = 0; d < ifms_depth; d++) {
#pragma HLs PIPELINE
		for (int h = 0; h < filter_dim; h++) {
			for (int w = 0; w < padding_left; w++) {
				conv_dw_communication_buffer_intra[d][h][w] =
						layer_ifm_zero_point;
			}
			for (int w = padding_left; w < filter_dim; w++) {
				if (starting_h + h < layer_2_dw_filter_dim) {
					conv_dw_communication_buffer_intra[d][h][w] =
							conv_dw_communication_buffer_inter[d][starting_h + h][w
									- padding_left];
				} else {
					conv_dw_communication_buffer_intra[d][h][w] =
							conv_dw_communication_buffer_inter[d][starting_h + h
									- layer_2_dw_filter_dim][w - padding_left];
				}
			}
		}
	}
}

void fill_conv_dw_communication_buffer_intra(
		fms_dt conv_dw_communication_buffer_inter[first_conv_layer_num_fils][layer_2_dw_filter_dim][layer_2_dw_ifm_width],
		fms_dt conv_dw_communication_buffer_intra[first_conv_layer_num_fils][layer_2_dw_filter_dim][layer_2_dw_filter_dim],
		const int starting_h, const int starting_w,
		const layer_specs specs_struct) {

	const int ifms_depth = specs_struct.layer_depth;
	const int ifms_width = specs_struct.layer_ifm_width;
	const int filter_dim = specs_struct.filter_size;
	const int strides = specs_struct.strides;
	const int filter_dim_minus_strides = filter_dim - strides;
	const fms_dt layer_ifm_zero_point = specs_struct.layer_ifms_zero_point;

	for (int d = 0; d < ifms_depth; d++) {
#pragma HLs PIPELINE
		for (int h = 0; h < filter_dim; h++) {
			for (int w = filter_dim_minus_strides; w < filter_dim; w++) {
				if (starting_w + w - filter_dim_minus_strides
						< layer_2_dw_ifm_width) {
					if (starting_h + h < layer_2_dw_filter_dim) {
						conv_dw_communication_buffer_intra[d][h][w] =
								conv_dw_communication_buffer_inter[d][h
										+ starting_h][starting_w + w
										- filter_dim_minus_strides];
					} else {
						conv_dw_communication_buffer_intra[d][h][w] =
								conv_dw_communication_buffer_inter[d][h
										+ starting_h - layer_2_dw_filter_dim][starting_w
										+ w - filter_dim_minus_strides];
					}
				} else {
					conv_dw_communication_buffer_intra[d][h][w] =
							layer_ifm_zero_point;
				}
			}
		}
	}
}

void pre_first_pipeline_layers_mob_v2(
		fms_grp_dt channels[input_image_depth
				* input_image_num_fms_groups_in_a_channel],
		fms_dt pre_first_pipeline_layers_output[PRE_FIRST_PIPELINE_OUTPUT_DEPTH][PRE_FIRST_PIPELINE_OUTPUT_HEIGHT][PRE_FIRST_PIPELINE_OUTPUT_WIDTH],
		weights_dt dw_layer_weights[layer_2_dw_num_fils][layer_2_dw_filter_dim
				* layer_2_dw_filter_dim],
		fms_quantization_scheme first_layer_quantization_params[first_conv_layer_num_fils],
		fms_quantization_scheme first_dw_layer_quantization_params[layer_2_dw_num_fils],
		fms_dt conv_dw_communication_buffer_inter[first_conv_layer_num_fils][layer_2_dw_filter_dim][layer_2_dw_ifm_width],
		fms_dt first_layers_input[input_image_depth][PRE_FIRST_PIPELINE_INPUT_HEIGHT][input_image_width],
		int starting_reading_h, const int end_reading_h) {

	const int num_of_ifm_groups_read_each_time =
			input_image_num_fms_groups_in_width
					* INPUT_IMAGE_ROWS_FILLED_EACH_TIME;

	fms_grp_dt fms_groups_buffer[input_image_depth][input_image_num_fms_groups_in_width
			* INPUT_IMAGE_ROWS_FILLED_EACH_TIME];

	int conv_dw_comm_buffer_reading_row;
	const int first_conv_layer_filter_dim_minus_strides =
			first_conv_layer_filter_dim - first_conv_layer_strides;
	const int layer_2_dw_padding_left = layer_2_dw_specs.padding_left;
	const int layer_2_dw_padding_top = layer_2_dw_specs.padding_top;

#if HW == _FPGA
	if (starting_reading_h == 0) {
		fill_input_image_groups_buffer(channels, fms_groups_buffer, 0,
				input_image_num_fms_groups_in_width); // to do
		input_image_fill_channels_buffer_from_groups_buffer(fms_groups_buffer,
				first_layers_input, 0, 0,
				first_conv_layer_specs.layer_ifms_zero_point);

		starting_reading_h += FIRST_CONV_LAYER_EXTRA_ROWS_FILLED_FIRST_TIME;

		fill_input_image_groups_buffer(channels, fms_groups_buffer,
				FIRST_CONV_LAYER_EXTRA_ROWS_FILLED_FIRST_TIME,
				num_of_ifm_groups_read_each_time); // to do
		input_image_fill_channels_buffer_from_groups_buffer(fms_groups_buffer,
				first_layers_input,
				FIRST_CONV_LAYER_EXTRA_ROWS_FILLED_FIRST_TIME,
				FIRST_CONV_LAYER_EXTRA_ROWS_FILLED_FIRST_TIME,
				first_conv_layer_specs.layer_ifms_zero_point);
	} else {
		fill_input_image_groups_buffer(channels, fms_groups_buffer,
				starting_reading_h, num_of_ifm_groups_read_each_time); // to do
		input_image_fill_channels_buffer_from_groups_buffer(fms_groups_buffer,
				first_layers_input, starting_reading_h,
				starting_reading_h % PRE_FIRST_PIPELINE_INPUT_HEIGHT,
				first_conv_layer_specs.layer_ifms_zero_point);
	}
#elif HW == CPU
    if (starting_reading_h == 0)
    {
        input_image_fill_channels_buffer_cpu(channels, first_layers_input, 0, false, first_conv_layer_specs.layer_ifms_zero_point);

        starting_reading_h += FIRST_CONV_LAYER_EXTRA_ROWS_FILLED_FIRST_TIME;

        input_image_fill_channels_buffer_cpu(channels,
                                             first_layers_input, FIRST_CONV_LAYER_EXTRA_ROWS_FILLED_FIRST_TIME,
                                             FIRST_CONV_LAYER_EXTRA_ROWS_FILLED_FIRST_TIME,
                                             first_conv_layer_specs.layer_ifms_zero_point);
    }
    else
    {
        input_image_fill_channels_buffer_cpu(channels,
                                             first_layers_input, starting_reading_h, starting_reading_h % PRE_FIRST_PIPELINE_INPUT_HEIGHT,
                                             first_conv_layer_specs.layer_ifms_zero_point);
    }
    starting_reading_h += INPUT_IMAGE_ROWS_FILLED_EACH_TIME;

#endif

	fms_dt first_layer_input_buffer[FIRST_CONV_LAYER_BUFFER_SIZE];
	fms_dt first_layer_input_buffer_new_cols[FIRST_CONV_LAYER_NEW_COLS_BUFFER_SIZE];

	fms_dt conv_dw_communication_buffer_intra[first_conv_layer_num_fils][layer_2_dw_filter_dim][layer_2_dw_filter_dim];

	const int starting_first_layer_conv_h =
			(starting_reading_h
					- (first_conv_layer_filter_dim
							- first_conv_layer_specs.padding_top))
					/ first_conv_layer_strides;

	int conv_dw_comm_buffer_writing_row = (starting_first_layer_conv_h
			+ layer_2_dw_specs.padding_top) % layer_2_dw_filter_dim;

	int conv_h = 0;
	const int first_row_h_iters = layer_2_dw_filter_dim - layer_2_dw_padding_top
			- 1;
	if (starting_first_layer_conv_h == 0) {
		for (int h = 0; h < first_row_h_iters; h++) {
			int h_in_first_layer_input_buffer = ((starting_reading_h
					+ conv_h * first_conv_layer_strides))
					% PRE_FIRST_PIPELINE_INPUT_HEIGHT;

			fill_first_cols_of_first_layer_input(first_layers_input,
					first_layer_input_buffer, h_in_first_layer_input_buffer);
			fill_first_layer_input_new_cols(first_layers_input,
					first_layer_input_buffer_new_cols,
					first_conv_layer_filter_dim_minus_strides,
					h_in_first_layer_input_buffer,
					first_conv_layer_specs.layer_ifms_zero_point);
			shift_and_fill_first_layer_input(first_layer_input_buffer_new_cols,
					first_layer_input_buffer);
			int first_layer_input_filling_w =
					(first_conv_layer_filter_dim_minus_strides);
			for (int w; w < layer_2_dw_ifm_width; w++) {
				first_conv_and_dw_layers_pipeline(first_layer_input_buffer,
						dw_layer_weights, conv_dw_communication_buffer_inter,
						conv_dw_communication_buffer_intra,
						pre_first_pipeline_layers_output,
						starting_first_layer_conv_h, conv_h, w,
						conv_dw_comm_buffer_writing_row,
						first_layer_quantization_params,
						first_dw_layer_quantization_params);
				fill_first_layer_input_new_cols(first_layers_input,
						first_layer_input_buffer_new_cols,
						first_layer_input_filling_w * first_conv_layer_strides
								+ first_conv_layer_filter_dim_minus_strides,
						h_in_first_layer_input_buffer,
						first_conv_layer_specs.layer_ifms_zero_point);
				shift_and_fill_first_layer_input(
						first_layer_input_buffer_new_cols,
						first_layer_input_buffer);
				first_layer_input_filling_w++;
			}

#if HW == _FPGA
			fill_input_image_groups_buffer(channels, fms_groups_buffer,
					(starting_reading_h + conv_h * first_conv_layer_strides),
					num_of_ifm_groups_read_each_time); // to do
			input_image_fill_channels_buffer_from_groups_buffer(
					fms_groups_buffer, first_layers_input,
					(starting_reading_h + conv_h * first_conv_layer_strides),
					h_in_first_layer_input_buffer,
					first_conv_layer_specs.layer_ifms_zero_point);
#elif HW == CPU
            input_image_fill_channels_buffer_cpu(channels,
                                                 first_layers_input, (starting_reading_h + conv_h * first_conv_layer_strides),
                                                 h_in_first_layer_input_buffer,
                                                 first_conv_layer_specs.layer_ifms_zero_point);
#endif
			conv_h++;
		}
		conv_dw_comm_buffer_writing_row++;
		if (conv_dw_comm_buffer_writing_row == 3) {
			conv_dw_comm_buffer_writing_row = 0;
		}
	}

	conv_dw_comm_buffer_reading_row = conv_dw_comm_buffer_writing_row
			- (layer_2_dw_filter_dim - layer_2_dw_specs.padding_top);
	if (conv_dw_comm_buffer_reading_row < 0) {
		conv_dw_comm_buffer_reading_row = conv_dw_comm_buffer_reading_row + 3;
	}

	pre_first_pipeline_layers_mob_v2: for (int h = 0; h < PRE_FIRST_PIPELINE_OUTPUT_HEIGHT; h++) {
		int h_in_first_layer_input_buffer = ((starting_reading_h
				+ conv_h * first_conv_layer_strides))
				% PRE_FIRST_PIPELINE_INPUT_HEIGHT;

		fill_first_cols_of_first_layer_input(first_layers_input,
				first_layer_input_buffer, h_in_first_layer_input_buffer);
		fill_first_layer_input_new_cols(first_layers_input,
				first_layer_input_buffer_new_cols,
				first_conv_layer_filter_dim_minus_strides,
				h_in_first_layer_input_buffer,
				first_conv_layer_specs.layer_ifms_zero_point);
		shift_and_fill_first_layer_input(first_layer_input_buffer_new_cols,
				first_layer_input_buffer);
		int w = 0;
		int first_layer_input_filling_w = (w
				+ first_conv_layer_filter_dim_minus_strides);
		for (; w < layer_2_dw_filter_dim - layer_2_dw_padding_left; w++) {
			first_conv_and_dw_layers_pipeline(first_layer_input_buffer,
					dw_layer_weights, conv_dw_communication_buffer_inter,
					conv_dw_communication_buffer_intra,
					pre_first_pipeline_layers_output,
					starting_first_layer_conv_h, conv_h, w,
					conv_dw_comm_buffer_writing_row,
					first_layer_quantization_params,
					first_dw_layer_quantization_params);
			fill_first_layer_input_new_cols(first_layers_input,
					first_layer_input_buffer_new_cols,
					first_layer_input_filling_w * first_conv_layer_strides
							+ first_conv_layer_filter_dim_minus_strides,
					h_in_first_layer_input_buffer,
					first_conv_layer_specs.layer_ifms_zero_point);
			shift_and_fill_first_layer_input(first_layer_input_buffer_new_cols,
					first_layer_input_buffer);
			first_layer_input_filling_w++;
		}

		fill_conv_dw_communication_buffer_intra_first_time(
				conv_dw_communication_buffer_inter,
				conv_dw_communication_buffer_intra,
				conv_dw_comm_buffer_reading_row, layer_2_dw_specs);

		int dw_layer_w = w
				- (layer_2_dw_filter_dim - layer_2_dw_specs.padding_left);

		for (; dw_layer_w < input_image_width / first_conv_layer_strides;
				dw_layer_w++) {
			first_conv_and_dw_layers_pipeline(first_layer_input_buffer,
					dw_layer_weights, conv_dw_communication_buffer_inter,
					conv_dw_communication_buffer_intra,
					pre_first_pipeline_layers_output,
					starting_first_layer_conv_h, conv_h, w,
					conv_dw_comm_buffer_writing_row,
					first_layer_quantization_params,
					first_dw_layer_quantization_params);
			fill_first_layer_input_new_cols(first_layers_input,
					first_layer_input_buffer_new_cols,
					first_layer_input_filling_w * first_conv_layer_strides
							+ first_conv_layer_filter_dim_minus_strides,
					h_in_first_layer_input_buffer,
					first_conv_layer_specs.layer_ifms_zero_point);
			fill_conv_dw_communication_buffer_intra(
					conv_dw_communication_buffer_inter,
					conv_dw_communication_buffer_intra,
					conv_dw_comm_buffer_reading_row, w, layer_2_dw_specs);
			shift_and_fill_first_layer_input(first_layer_input_buffer_new_cols,
					first_layer_input_buffer);
			first_layer_input_filling_w++;
			w++;
		}
#if HW == _FPGA
		fill_input_image_groups_buffer(channels, fms_groups_buffer,
				(starting_reading_h + conv_h * first_conv_layer_strides),
				num_of_ifm_groups_read_each_time); // to do
		input_image_fill_channels_buffer_from_groups_buffer(fms_groups_buffer,
				first_layers_input,
				(starting_reading_h + conv_h * first_conv_layer_strides),
				h_in_first_layer_input_buffer,
				first_conv_layer_specs.layer_ifms_zero_point);
#elif HW == CPU
        input_image_fill_channels_buffer_cpu(channels,
                                             first_layers_input, (starting_reading_h + conv_h * first_conv_layer_strides),
                                             h_in_first_layer_input_buffer,
                                             first_conv_layer_specs.layer_ifms_zero_point);
#endif
		conv_dw_comm_buffer_writing_row++;
		if (conv_dw_comm_buffer_writing_row == 3) {
			conv_dw_comm_buffer_writing_row = 0;
		}
		conv_dw_comm_buffer_reading_row = conv_dw_comm_buffer_writing_row
				- (layer_2_dw_filter_dim - layer_2_dw_specs.padding_top);
		if (conv_dw_comm_buffer_reading_row < 0) {
			conv_dw_comm_buffer_reading_row = conv_dw_comm_buffer_reading_row
					+ 3;
		}

		conv_h++;
		if ((conv_h - 1) * first_conv_layer_strides + starting_reading_h
				>= end_reading_h) {
			break;
		}
	}
}
