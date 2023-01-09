#include "bottleneck.h"

void bottleneck_0_fill_projection_kernel_weights(
		const weights_dt layer_weights[][bottleneck_0_expanded_ifms_depth],
		weights_dt kernel_weights[], int d) {
	bottleneck_0_fill_projection_kernel_weights: for (int filter_index = 0;
			filter_index < bottleneck_0_ofms_depth; filter_index++) {
		kernel_weights[filter_index] = layer_weights[filter_index][d];
	}
}

void bottleneck_0_padding_top_right(
		fms_dt previous_pass_dw_input[][bottlenck_0_inter_pass_dw_input_height][bottlenck_0_inter_pass_dw_input_width],
		fms_dt zero_point) {
	bottleneck_0_padding_top_right: for (int d = 0;
			d < bottleneck_0_expanded_ifms_depth; d++) {
		for (int h = 0; h < bottleneck_0_dw_padding_top; h++) {
			for (int w = 0; w < bottlenck_0_inter_pass_dw_input_width; w++) {
				previous_pass_dw_input[d][h][w] = zero_point;
			}
			for (int w = bottleneck_0_ifms_width;
					w < bottlenck_0_inter_pass_dw_input_width; w++) {
				previous_pass_dw_input[d][h][w] = zero_point;
			}
		}
	}
}

void bottleneck_0_do_padding_left(
		fms_dt previous_pass_dw_input[][bottlenck_0_inter_pass_dw_input_height][bottlenck_0_inter_pass_dw_input_width],
		fms_dt dw_lower_buffer[][bottleneck_0_dw_filter_dim],
		fms_dt zero_point) {
	bottleneck_0_do_padding_left: for (int d = 0;
			d < bottleneck_0_expanded_ifms_depth; d++) {
		previous_pass_dw_input[d][1][0] = zero_point;
		dw_lower_buffer[d][bottleneck_0_dw_filter_dim - bottleneck_0_dw_strides] =
				zero_point;
	}
}

void bottleneck_0_update_previous_pass_buffer(
		fms_dt previous_pass_dw_input[][bottlenck_0_inter_pass_dw_input_height][bottlenck_0_inter_pass_dw_input_width],
		fms_dt dw_lower_buffer[][bottleneck_0_dw_filter_dim], int offset_w,
		int starting_h, int d) {
#pragma HLS INLINE
//	bottleneck_0_update_previous_pass_buffer: for (int d = 0; d < bottleneck_0_expanded_ifms_depth; d++)
//	{
//#pragma HLS PIPELINE
	if (starting_h != 0) {
		previous_pass_dw_input[d][0][offset_w] =
				previous_pass_dw_input[d][1][offset_w];
	}
	previous_pass_dw_input[d][1][offset_w] = dw_lower_buffer[d][0];
//		if (offset_w + 1 == bottleneck_0_ifms_width / layer_0_s_strides)//to do, handle once at another place
//		{
//			previous_pass_dw_input[d][1][offset_w + 1] = dw_lower_buffer[d][1];
//			previous_pass_dw_input[d][1][offset_w + 2] = dw_lower_buffer[d][2];
//		}
//	}
}

void bottleneck_0_fill_dw_input(
		fms_dt previous_pass_dw_input[][bottlenck_0_inter_pass_dw_input_height][bottlenck_0_inter_pass_dw_input_width],
		fms_dt dw_lower_buffer[][bottleneck_0_dw_filter_dim],
		fms_dt dw_input_buffer[], fms_dt expansion_layer_result, int filling_d,
		int filling_w_offset, int absolute_starting_h) {

	if (filling_w_offset > 0 && absolute_starting_h != 0) {
		dw_input_buffer[0] =
				previous_pass_dw_input[filling_d][0][filling_w_offset - 1];
		dw_input_buffer[1] =
				previous_pass_dw_input[filling_d][0][filling_w_offset];
		dw_input_buffer[2] =
				previous_pass_dw_input[filling_d][0][filling_w_offset + 1];
		dw_input_buffer[3] =
				previous_pass_dw_input[filling_d][1][filling_w_offset - 1];
		dw_input_buffer[4] =
				previous_pass_dw_input[filling_d][1][filling_w_offset];
		dw_input_buffer[5] =
				previous_pass_dw_input[filling_d][1][filling_w_offset + 1];

		dw_input_buffer[6] = dw_lower_buffer[filling_d][0];
		dw_input_buffer[7] = dw_lower_buffer[filling_d][1];
	}
	// fill lower part of dw_input
	dw_input_buffer[8] = expansion_layer_result;
	dw_lower_buffer[filling_d][2] = expansion_layer_result;
}

void mob_v2_bottleneck_0(fms_dt bottleneck_input[], fms_dt bottleneck_output[],
		fms_dt previous_pass_dw_input[][bottlenck_0_inter_pass_dw_input_height][bottlenck_0_inter_pass_dw_input_width],
		fms_dt dw_lower_buffer[][bottleneck_0_dw_filter_dim], int starting_h,
		int starting_w) {
#pragma HLS INLINE off

	const fms_dt expansion_layer_ofms_zero_point =
			conv_fms_zero_points[bottleneck_0_expansion_layer_index + 1];
	const rec_scales_dt expansion_layer_ofms_scale_rec =
			conv_fms_scales_rec[bottleneck_0_expansion_layer_index + 1];
	const rec_scales_dt expansion_layer_ofms_scale =
			conv_fms_scales[bottleneck_0_expansion_layer_index + 1];

	const fms_dt dw_layer_ofms_zero_point =
			conv_fms_zero_points[bottleneck_0_dw_layer_index + 1];
	const rec_scales_dt dw_layer_ofms_scale_rec =
			conv_fms_scales_rec[bottleneck_0_dw_layer_index + 1];
	const rec_scales_dt dw_layer_ofms_scale =
			conv_fms_scales[bottleneck_0_dw_layer_index + 1];
	const fms_dt current_dw_ifms_zero_point =
			conv_fms_zero_points[bottleneck_0_dw_layer_index];

	fms_quantization_scheme expansion_layer_normalization;
	expansion_layer_normalization.ofm_zero_point =
			expansion_layer_ofms_zero_point;
	expansion_layer_normalization.ofm_scale_rec =
			expansion_layer_ofms_scale_rec;
	expansion_layer_normalization.ofm_scale = expansion_layer_ofms_scale;

	fms_quantization_scheme dw_layer_normalization;
	dw_layer_normalization.ofm_zero_point = dw_layer_ofms_zero_point;
	dw_layer_normalization.ofm_scale_rec = dw_layer_ofms_scale_rec;
	dw_layer_normalization.ofm_scale = dw_layer_ofms_scale;

	pss_dt projection_results_buffer[bottleneck_0_ofms_depth];

	const int current_dw_starting_h = starting_h
			- (bottleneck_0_dw_filter_dim - bottleneck_0_dw_padding_top) + 1; //+1 is the current
	const int current_dw_starting_w = starting_w
			- (bottleneck_0_dw_filter_dim - bottleneck_0_dw_padding_left) + 1;
	const int non_first_step_in_w = starting_w != 0;
	shift_dw_ifms_buffer_horizontally_3x3_s1(dw_lower_buffer,
			bottleneck_0_dw_strides, bottleneck_0_expanded_ifms_depth);

	mob_v2_bottleneck_0: for (int d_in_out = 0;
			d_in_out < bottleneck_0_expanded_ifms_depth; d_in_out++) {
#pragma HLS PIPELINE
		weights_dt projection_kernel_weights[bottleneck_0_ofms_depth];

		expansion_layer_normalization.fused_scales =
				layer_0_s_fused_scales[d_in_out];
		expansion_layer_normalization.fused_scales_log_2_shift =
				layer_0_s_fused_scales_log_2_shifts[d_in_out];
		expansion_layer_normalization.relu_6_fused_scale =
				layer_0_s_fused_scales_log_2_shifts[d_in_out];
		expansion_layer_normalization.fused_zero_point =
				layer_0_s_fused_zero_points[d_in_out];
		expansion_layer_normalization.layer_0_relu_6_fused_scale =
				layer_0_s_relu_6_fused_scales[d_in_out];
		fms_dt expansion_result;
		fms_dt dw_input_buffer[bottleneck_0_dw_filter_dim
				* bottleneck_0_dw_filter_dim];
#pragma HLS ARRAY_PARTITION variable = dw_input_buffer type = complete dim = 0

		if (starting_h < bottleneck_0_ifms_height + bottleneck_0_dw_padding_top
				&& starting_w < bottleneck_0_ifms_width / layer_0_s_strides) {
			pss_dt expansion_pss = conv_kernel(bottleneck_input, weights_0,
					layer_0_s_filter_dim, d_in_out);
			expansion_result = conv_relu_norm(expansion_pss,
					expansion_layer_normalization,
					bottleneck_0_expansion_layer_relu);
			const int ifms_buffer_hw = bottlenck_0_input_buffer_height
					* bottlenck_0_input_buffer_width;
		} else {
			expansion_result = current_dw_ifms_zero_point;
		}
		bottleneck_0_fill_dw_input(previous_pass_dw_input, dw_lower_buffer,
				dw_input_buffer, expansion_result, d_in_out, starting_w,
				starting_h);

		dw_layer_normalization.fused_scales = layer_1_dw_fused_scales[d_in_out];
		dw_layer_normalization.fused_scales_log_2_shift =
				layer_1_dw_fused_scales_log_2_shifts[d_in_out];
		dw_layer_normalization.relu_6_fused_scale =
				layer_1_dw_relu_6_fused_scales[d_in_out];
		dw_layer_normalization.fused_zero_point =
				layer_1_dw_fused_zero_points[d_in_out];

		dw_pss_dt dw_pss = dw_kernel(dw_input_buffer, dw_weights_1,
				bottleneck_0_dw_filter_dim, d_in_out);
		fms_dt dw_result = dw_relu_norm(dw_pss, dw_layer_normalization,
				bottleneck_0_dw_layer_relu);
		if (starting_w
				>= bottleneck_0_dw_filter_dim - bottleneck_0_dw_padding_left and d_in_out > 0) {
			bottleneck_0_update_previous_pass_buffer(previous_pass_dw_input,
					dw_lower_buffer, starting_w - 1, starting_h, d_in_out - 1);
		}
		bottleneck_0_fill_projection_kernel_weights(pw_weights_2,
				projection_kernel_weights, d_in_out);
		projection_kernel(dw_result, bottleneck_0_ofms_depth,
				projection_kernel_weights, projection_results_buffer, d_in_out);
	}
	normalize_projection_kernel_output(projection_results_buffer,
			bottleneck_output, layer_2_pw_fused_scales,
			layer_2_pw_fused_scales_log_2_shifts,
			layer_2_pw_relu_6_fused_scales, layer_2_pw_fused_zero_points,
			bottleneck_0_ofms_depth, layer_2_activation,
			bottleneck_0_projection_layer_index);
	if (starting_w
			>= bottleneck_0_dw_filter_dim - bottleneck_0_dw_padding_left) {
		bottleneck_0_update_previous_pass_buffer(previous_pass_dw_input,
				dw_lower_buffer, starting_w - 1, starting_h, bottleneck_0_expanded_ifms_depth - 1);
	}
}
