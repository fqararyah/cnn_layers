#include "../headers/layers_imp_common_includes.h"
#include "../headers/conv.h"
#include "../headers/pw_conv.h"

#if FIBHA_VERSION == 2

void conv_v2_fill_input_image_groups_buffer(
		fms_grp_dt channels[input_image_depth
				* input_image_num_fms_groups_in_a_channel],
		fms_grp_dt fms_groups_buffer[input_image_depth][input_image_num_fms_groups_in_width
				* first_conv_layer_strides], const int starting_h,
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

void conv_v2_input_image_fill_row_from_groups_buffer(
		fms_grp_dt fms_groups_buffer[input_image_depth][input_image_num_fms_groups_in_width
				* first_conv_layer_strides],
		fms_dt channels_buffer_0[input_image_depth][first_conv_layer_filter_dim][input_image_width],
		int row, const int channels_buffer_start_filling_h) {
#pragma HLS INLINE off

	const int start_filling_offset = row * input_image_num_fms_groups_in_width;
	for (int o_w = 0; o_w < input_image_num_fms_groups_in_width; o_w++) {
		const int o_w_offset = o_w * input_image_group_items;
		for (int d = 0; d < input_image_depth; d++) {
			fms_grp_dt chunck = fms_groups_buffer[d][start_filling_offset + o_w];
			for (int w = 0; w < input_image_group_items; w++) {
#pragma HLS PIPELINE
				if (o_w_offset + w < input_image_width) {
#if HW == _FPGA
					channels_buffer_0[d][channels_buffer_start_filling_h + row][o_w_offset
							+ w] = (fms_dt) chunck(
							w * fms_dt_width + fms_dt_offset, w * fms_dt_width);
#endif
				}
			}
		}
	}
}

void shift_channels_buffer_rows(
		fms_dt channels_tile[input_image_depth][first_conv_layer_filter_dim][input_image_width],
		const int rows_to_shift) {
#pragma HLS INLINE

	for (int w = 0; w < input_image_width; w++) {
#pragma HLS PIPELINE
		for (int d = 0; d < input_image_depth; d++) {
#pragma HLS UNROLL
			for (int h = 0; h < rows_to_shift; h++) {
#pragma HLS UNROLL
				channels_tile[d][h][w] = channels_tile[d][h
						+ (first_conv_layer_filter_dim - rows_to_shift)][w];
			}
		}
	}
}

void chain_0_1_padd_bottom_channels_buffer_rows(
		fms_dt channels_buffer_0[input_image_depth][first_conv_layer_filter_dim][input_image_width],
		const fms_dt zero_point) {
#pragma HLS INLINE

	for (int w = 0; w < input_image_width; w++) {
#pragma HLS PIPELINE
		for (int d = 0; d < input_image_depth; d++) {
#pragma HLS UNROLL
			for (int h = first_conv_layer_filter_dim
					- first_conv_layer_specs.padding_bottom;
					h < first_conv_layer_filter_dim; h++) {
#pragma HLS UNROLL
				channels_buffer_0[d][h][w] = zero_point;
			}
		}
	}
}

void conv_v2_input_image_fill_channels_buffer_from_groups_buffer(
		fms_grp_dt fms_groups_buffer[input_image_depth][input_image_num_fms_groups_in_width
				* first_conv_layer_strides],
		fms_dt channels_tile[first_conv_layer_depth][first_conv_layer_filter_dim][first_conv_layer_ifm_width],
		const int starting_h, const int channels_buffer_start_filling_h,
		const fms_dt zero_point) {
#pragma HLS INLINE off

	const int rows_to_shift = first_conv_layer_filter_dim
			- first_conv_layer_strides;
	shift_channels_buffer_rows(channels_tile, rows_to_shift);

	const int to_fill_rows_count = first_conv_layer_filter_dim
			- channels_buffer_start_filling_h;

	for (int h = 0; h < first_conv_layer_strides; h++) {
		if (h < to_fill_rows_count) {
			if (starting_h + h < input_image_height) {
				conv_v2_input_image_fill_row_from_groups_buffer(
						fms_groups_buffer, channels_tile, h,
						channels_buffer_start_filling_h);
			} else {
				chain_0_1_padd_bottom_channels_buffer_rows(channels_tile,
						first_conv_layer_specs.layer_ifms_zero_point);
			}
		}
	}
}

void fill_channels_buffer_cpu(
		fms_grp_dt input_image[input_image_depth
				* input_image_num_fms_groups_in_a_channel],
		fms_dt channels_tile[input_image_depth][first_conv_layer_filter_dim][input_image_width],
		int starting_h, int to_fill_rows_count) {
#pragma HLS INLINE off

	const int rows_to_shift = first_conv_layer_filter_dim
			- first_conv_layer_strides;
	shift_channels_buffer_rows(channels_tile, rows_to_shift);

	const int channels_buffer_start_filling_h = first_conv_layer_filter_dim
			- to_fill_rows_count;
	for (int h = 0; h < first_conv_layer_specs.strides; h++) {
		if (h < to_fill_rows_count) {
			if (starting_h + h < input_image_height) {
				for (int d = 0; d < input_image_depth; d++) {
					for (int w = 0; w < input_image_width; w++) {
						channels_tile[d][h + channels_buffer_start_filling_h][w] =
								input_image[d * input_image_hw
										+ (h + starting_h) * input_image_width
										+ w];
					}
				}
			} else {
				chain_0_1_padd_bottom_channels_buffer_rows(channels_tile,
						first_conv_layer_specs.layer_ifms_zero_point);
			}
		}
	}
}

// Note that this implementation of layer_0 is not not very optimized
void layer_0_s_conv_engine(
		const layer_0_weights_dt weights_1[first_conv_layer_num_fils][first_conv_layer_depth][first_conv_layer_filter_dim][first_conv_layer_filter_dim],
		fms_dt channels_tile[input_image_depth][first_conv_layer_filter_dim][input_image_width],
		fms_dt result[][CHANNELS_TILE_HEIGHT][CHANNELS_TILE_WIDTH],
		int starting_h, const fused_scales_dt fused_scales[],
		const relu_6_fused_scales_dt relu_6_fused_scales[],
		const biases_dt fused_zero_points[]) {

	const biases_dt current_layer_zero_point =
			first_conv_layer_specs.layer_ifms_zero_point;
	for (int f = 0; f < first_conv_layer_num_fils; f++) {

		fms_dt ofm_zero_point = first_conv_layer_specs.layer_ofms_zero_point;
		scales_dt ofm_scale = first_conv_layer_specs.layer_ofms_scale;
		biases_dt fused_zero_point = fused_zero_points[f];
		fused_scales_dt fused_scale = fused_scales[f];
		relu_6_fused_scales_dt relu_6_fused_scale = relu_6_fused_scales[0];
		int layer_relu = first_conv_layer_specs.layer_activation;
		for (int w = 0; w < first_conv_layer_specs.layer_ofm_width; w++) {
#pragma HLS PIPELINE
			pss_dt tmp = 0;
			for (int d = 0; d < first_conv_layer_depth; d++) {
#pragma HLS UNROLL
				for (int c_h = 0; c_h < first_conv_layer_filter_dim; c_h++) {
#pragma HLS UNROLL
					for (int c_w = 0; c_w < first_conv_layer_filter_dim;
							c_w++) {
#pragma HLS UNROLL
						// tmp += weights_1[f][d][c_h][c_w] * channels_tile[d][c_h][w * first_conv_layer_specs.strides + c_w];
						if (w * first_conv_layer_specs.strides + c_w
								< first_conv_layer_ifm_width) {
							tmp += weights_1[f][d][c_h][c_w]
									* channels_tile[d][c_h][w
											* first_conv_layer_specs.strides
											+ c_w];
						} else {
							tmp += weights_1[f][d][c_h][c_w]
									* current_layer_zero_point;
						}
						// if ((starting_h == 9 || starting_h == 8) && w == 0 && d == 0 && f == 0)
						// {
						// 	printf("%d * %d \n", (int)weights_1[f][d][c_h][c_w],
						// 		   (int)channels_tile[d][c_h][w * first_conv_layer_specs.strides + c_w]);
						// }
					}
				}
				// if ((starting_h == 9 || starting_h == 8) && w == 0 && d == 0 && f == 0)
				// {
				// 	printf("\n*************\n");
				// }
			}
			const int tile_in_d = f / pw_tile_d;
			const int tile_in_h = starting_h / pw_tile_h;
			const int tile_in_w = w / pw_tile_w;
			const int tile_index = tile_in_d
					* (first_conv_layer_specs.layer_num_of_ofm_tiles_h
							* first_conv_layer_specs.layer_num_of_ofm_tiles_w)
					+ tile_in_h
							* first_conv_layer_specs.layer_num_of_ofm_tiles_w
					+ tile_in_w;

			// const int in_tile_d = f % pw_tile_d;
			const int in_tile_h = starting_h % pw_tile_h;
			const int in_tile_w = w % pw_tile_w;
			// const int in_tile_index = in_tile_d * pw_tile_hw + in_tile_h * pw_tile_w + in_tile_w;

			result[tile_index][in_tile_h][in_tile_w] = // conv_relu_norm(
														// tmp, normalization, first_conv_layer_specs.layer_activation);
					conv_relu_norm_v2(tmp, fused_zero_point, ofm_zero_point,
							fused_scale, relu_6_fused_scale, layer_relu);
			// if (tile_in_h * pw_tile_h + in_tile_h == 10 && tile_in_w * pw_tile_w + in_tile_w == 20 && tile_in_d == 1)
			// {
			// 	pss_dt pss = tmp;
			// 	pss += fused_zero_point;
			// 	printf("%d >> ", (int)pss);

			// 	pss_f_dt scaled_pss = pss * fused_scale;
			// 	printf("%f >> ", (float)scaled_pss);
			// 	printf("%f >> ", (float)fused_scale);
			// 	if (layer_relu == 6 && scaled_pss > relu_6_fused_scale)
			// 	{
			// 		scaled_pss = relu_6_fused_scale;
			// 	}

			// 	scaled_pss += ofm_zero_point;
			// 	printf("%f >> ", (float)scaled_pss);
			// 	scaled_pss += quant_half - (scaled_pss < 0);
			// 	printf("%f \n ", (float)scaled_pss);
			// }
		}
	}
}

void layer_0_s_3x3(
		fms_grp_dt input_image[input_image_depth
				* input_image_num_fms_groups_in_a_channel],
		fms_dt result[][CHANNELS_TILE_HEIGHT][CHANNELS_TILE_WIDTH],
		const int starting_row_in_ofms, const int to_produce_rows_count) {
	const int rows_filled_first_time = first_conv_layer_filter_dim
			- first_conv_layer_strides;
	fms_dt channels_tile[first_conv_layer_depth][first_conv_layer_filter_dim][first_conv_layer_ifm_width];
	fms_grp_dt fms_groups_buffer[input_image_depth][input_image_num_fms_groups_in_width
			* first_conv_layer_strides];

	const int num_of_ifm_groups_read_each_time =
			input_image_num_fms_groups_in_width * first_conv_layer_strides;

	const int starting_row_in_ifms = starting_row_in_ofms
			* first_conv_layer_strides;
	const int ending_row_in_ofms = starting_row_in_ofms + to_produce_rows_count;
	const int to_fill_rows_count = first_conv_layer_strides;

#if HW == CPU
	fill_channels_buffer_cpu(input_image, channels_tile, starting_row_in_ifms, rows_filled_first_time);
	fill_channels_buffer_cpu(input_image, channels_tile, starting_row_in_ifms + rows_filled_first_time,
							 to_fill_rows_count);
#elif HW == _FPGA
	conv_v2_fill_input_image_groups_buffer(input_image, fms_groups_buffer,
			starting_row_in_ifms, input_image_num_fms_groups_in_width); // to do
	conv_v2_input_image_fill_channels_buffer_from_groups_buffer(
			fms_groups_buffer, channels_tile, starting_row_in_ifms,
			first_conv_layer_filter_dim - rows_filled_first_time,
			first_conv_layer_specs.layer_ifms_zero_point);

	conv_v2_fill_input_image_groups_buffer(input_image, fms_groups_buffer,
			starting_row_in_ifms + rows_filled_first_time,
			num_of_ifm_groups_read_each_time); // to do
	conv_v2_input_image_fill_channels_buffer_from_groups_buffer(
			fms_groups_buffer, channels_tile,
			starting_row_in_ifms + rows_filled_first_time,
			rows_filled_first_time,
			first_conv_layer_specs.layer_ifms_zero_point);
#endif
	// printf("************\n");
	for (int h = starting_row_in_ofms; h < ending_row_in_ofms; h++) {
		const int start_reading_h = (h + 1) * first_conv_layer_strides
				+ rows_filled_first_time;
		// printf("%d\n", h);
		layer_0_s_conv_engine(first_layer_weights, channels_tile, result,
				h - starting_row_in_ofms, first_conv_layer_fused_scales,
				first_conv_layer_relu_6_fused_scales,
				first_conv_layer_fused_zero_points);
#if HW == CPU
		fill_channels_buffer_cpu(input_image, channels_tile, start_reading_h, to_fill_rows_count);
#elif HW == _FPGA
		conv_v2_fill_input_image_groups_buffer(input_image, fms_groups_buffer,
				start_reading_h, num_of_ifm_groups_read_each_time); // to do
		conv_v2_input_image_fill_channels_buffer_from_groups_buffer(
				fms_groups_buffer, channels_tile, start_reading_h,
				rows_filled_first_time,
				first_conv_layer_specs.layer_ifms_zero_point);
#endif
	}
}

#endif
