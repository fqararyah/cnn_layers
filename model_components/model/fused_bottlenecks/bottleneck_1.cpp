#include "bottleneck.h"

#if CHAIN_LENGTH >= 6 && MODEL_ID == 2 && ! ONLY_SEML

void bottleneck_1_fill_projection_kernel_weights(
	const weights_dt layer_weights[][bottleneck_1_expanded_ifms_depth],
	weights_dt kernel_weights[], int d)
{
#pragma HLS INLINE
bottleneck_1_fill_projection_kernel_weights:
	for (int filter_index = 0;
		 filter_index < bottleneck_1_ofms_depth; filter_index++)
	{
#pragma HLS UNROLL
		kernel_weights[filter_index] = layer_weights[filter_index][d];
	}
}

void bottleneck_1_padding_right(
	fms_dt bottleneck_1_previous_pass_dw_input[bottleneck_1_expanded_ifms_depth][bottleneck_1_inter_pass_dw_input_width],
	const fms_dt zero_point)
{
#pragma HLS INLINE

bottleneck_1_padding_right:
	for (int d = 0;
		 d < bottleneck_1_expanded_ifms_depth; d++)
	{
		for (int w = bottleneck_1_ifms_width;
			 w < bottleneck_1_inter_pass_dw_input_width; w++)
		{
			bottleneck_1_previous_pass_dw_input[d][w] = zero_point;
		}
	}
}

void bottleneck_1_update_previous_pass_buffer(
	fms_dt previous_pass_dw_input[][bottleneck_1_inter_pass_dw_input_width],
	fms_dt dw_lower_buffer[][bottleneck_1_dw_filter_dim * bottleneck_1_dw_strides], int offset_w, int d)
{
#pragma HLS INLINE

	previous_pass_dw_input[d][offset_w] = dw_lower_buffer[d][3];
	previous_pass_dw_input[d][offset_w + 1] = dw_lower_buffer[d][4];
}

void bottleneck_1_fill_dw_input(
	fms_dt previous_pass_dw_input[][bottleneck_1_inter_pass_dw_input_width],
	fms_dt dw_lower_buffer[][bottleneck_1_dw_filter_dim * bottleneck_1_dw_strides], fms_dt dw_input_buffer[],
	fms_dt expansion_result[bottleneck_1_expansion_parallelism_h][bottleneck_1_expansion_parallelism_w],
	int filling_d, int filling_w_offset, int absolute_starting_h)
{

	if (filling_w_offset >= 0 && absolute_starting_h != 0)
	{
		dw_input_buffer[0] = previous_pass_dw_input[filling_d][filling_w_offset];
		dw_input_buffer[1] =
			previous_pass_dw_input[filling_d][filling_w_offset + 1];
		dw_input_buffer[2] = previous_pass_dw_input[filling_d][filling_w_offset + 2];

		dw_input_buffer[3] = dw_lower_buffer[filling_d][0];
		dw_input_buffer[6] = dw_lower_buffer[filling_d][3];
	}
	// fill lower part of dw_input
	dw_input_buffer[4] = expansion_result[0][0];
	dw_input_buffer[5] = expansion_result[0][1];
	dw_input_buffer[7] = expansion_result[1][0];
	dw_input_buffer[8] = expansion_result[1][1];
	dw_lower_buffer[filling_d][1] = expansion_result[0][0];
	dw_lower_buffer[filling_d][2] = expansion_result[0][1];
	dw_lower_buffer[filling_d][4] = expansion_result[1][0];
	dw_lower_buffer[filling_d][5] = expansion_result[1][1];
}

void bottleneck_1_copy_projection_kernel_output_buffer(pss_dt projection_kernel_output_buffer[],
													   pss_dt projection_kernel_output_buffer_prev[])
{
#pragma HLS PIPELINE
	copy_projection_kernel_output_buffer(projection_kernel_output_buffer, projection_kernel_output_buffer_prev, bottleneck_1_ofms_depth);
}

void mob_v2_bottleneck_1(fms_dt bottleneck_input[],
						 pss_dt projection_kernel_output_buffer[bottleneck_1_ofms_depth],
						 pss_dt projection_kernel_output_buffer_prev[bottleneck_1_ofms_depth],
						 fms_dt next_bottleneck_communication_buffer[bottleneck_1_ofms_depth][bottleneck_1_ofms_width],
						 fms_dt previous_pass_dw_input_r[bottleneck_1_expanded_ifms_depth][bottleneck_1_inter_pass_dw_input_width],
						 fms_dt previous_pass_dw_input_w[bottleneck_1_expanded_ifms_depth][bottleneck_1_inter_pass_dw_input_width],
						 fms_dt dw_lower_buffer[bottleneck_1_expanded_ifms_depth][bottleneck_1_dw_filter_dim * bottleneck_1_dw_strides], const int starting_h,
						 const int expansion_kernel_starting_w)
{
#pragma HLS INLINE off

	const fms_dt expansion_layer_ofms_zero_point =
		conv_fms_zero_points[bottleneck_1_expansion_layer_index + 1];
	const rec_scales_dt expansion_layer_ofms_scale_rec =
		conv_fms_scales_rec[bottleneck_1_expansion_layer_index + 1];
	const rec_scales_dt expansion_layer_ofms_scale =
		conv_fms_scales[bottleneck_1_expansion_layer_index + 1];

	const fms_dt dw_layer_ofms_zero_point =
		conv_fms_zero_points[bottleneck_1_dw_layer_index + 1];
	const rec_scales_dt dw_layer_ofms_scale_rec =
		conv_fms_scales_rec[bottleneck_1_dw_layer_index + 1];
	const rec_scales_dt dw_layer_ofms_scale =
		conv_fms_scales[bottleneck_1_dw_layer_index + 1];
	const fms_dt current_dw_ifms_zero_point =
		conv_fms_zero_points[bottleneck_1_dw_layer_index];

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

	const int dw_kernel_starting_w = expansion_kernel_starting_w - (bottleneck_0_dw_filter_dim - bottleneck_0_dw_padding_left) + 1;
	const int prev_projection_kernel_starting_w = dw_kernel_starting_w / bottleneck_1_dw_strides - 1;

	const int first_step_in_w = expansion_kernel_starting_w == 0;

	const int ifms_buffer_hw = bottlenck_0_input_buffer_height * bottlenck_0_input_buffer_width;

mob_v2_bottleneck_1:
	for (int d_in_out = 0;
		 d_in_out < bottleneck_1_expanded_ifms_depth; d_in_out++)
	{
#pragma HLS PIPELINE
		weights_dt projection_kernel_weights[bottleneck_1_ofms_depth];
#pragma HLS ARRAY_PARTITION variable = projection_kernel_weights type = complete

		expansion_layer_normalization.fused_scales =
			layer_3_pw_fused_scales[d_in_out];
		expansion_layer_normalization.fused_scales_log_2_shift =
			layer_3_pw_fused_scales_log_2_shifts[d_in_out];
		expansion_layer_normalization.relu_6_fused_scale =
			layer_3_pw_relu_6_fused_scales[d_in_out];
		expansion_layer_normalization.fused_zero_point =
			layer_3_pw_fused_zero_points[d_in_out];

		fms_dt expansion_result[bottleneck_1_expansion_parallelism_h][bottleneck_1_expansion_parallelism_w];
		fms_dt dw_input_buffer[bottleneck_1_dw_filter_dim * bottleneck_1_dw_filter_dim];
#pragma HLS ARRAY_PARTITION variable = dw_input_buffer type = complete dim = 0

		for (int h = 0; h < bottleneck_1_expansion_parallelism_h; h++)
		{
#pragma HLS UNROLL
			for (int w = 0; w < bottleneck_1_expansion_parallelism_w; w++)
			{
#pragma HLS UNROLL
				if (starting_h + h < bottleneck_1_ifms_height + bottleneck_1_dw_padding_top &&
					expansion_kernel_starting_w + w < bottleneck_1_ifms_width)
				{
					if (first_step_in_w + w < bottleneck_1_expansion_parallelism_w)
					{
						// in first step, do only one and write it to the second col
						pss_dt expansion_pss = expansion_kernel(bottleneck_input,
																pw_weights_3[d_in_out], bottleneck_1_ifms_depth,
																d_in_out, h, w,
																bottleneck_1_expansion_parallelism_w);
						expansion_result[h][first_step_in_w + w] = pw_relu_norm(
							expansion_pss, expansion_layer_normalization,
							bottleneck_1_expansion_layer_relu);
					}
				}
				else
				{
					expansion_result[h][w] = current_dw_ifms_zero_point;
				}
			}
		}

		bottleneck_1_fill_dw_input(previous_pass_dw_input_r, dw_lower_buffer,
								   dw_input_buffer, expansion_result, d_in_out, dw_kernel_starting_w,
								   starting_h);

		dw_layer_normalization.fused_scales = layer_6_dw_fused_scales[d_in_out];
		dw_layer_normalization.fused_scales_log_2_shift =
			layer_6_dw_fused_scales_log_2_shifts[d_in_out];
		dw_layer_normalization.relu_6_fused_scale =
			layer_6_dw_relu_6_fused_scales[d_in_out];
		dw_layer_normalization.fused_zero_point =
			layer_6_dw_fused_zero_points[d_in_out];

		dw_pss_dt dw_pss = dw_kernel(dw_input_buffer, dw_weights_6[d_in_out],
									 bottleneck_1_dw_filter_dim);
		fms_dt dw_result = dw_relu_norm(dw_pss, dw_layer_normalization,
										bottleneck_1_dw_layer_relu);

		if (dw_kernel_starting_w >= 0)
		{
			bottleneck_1_update_previous_pass_buffer(previous_pass_dw_input_w,
													 dw_lower_buffer, dw_kernel_starting_w, d_in_out);
		}
		bottleneck_1_fill_projection_kernel_weights(pw_weights_7,
													projection_kernel_weights, d_in_out);
		projection_kernel(dw_result, bottleneck_1_ofms_depth,
						  projection_kernel_weights, projection_kernel_output_buffer,
						  d_in_out);

		if (prev_projection_kernel_starting_w >= 0 && d_in_out < bottleneck_1_ofms_depth)
		{
			next_bottleneck_communication_buffer[d_in_out][prev_projection_kernel_starting_w] =
				normalize_projection_kernel_output(
					projection_kernel_output_buffer_prev,
					layer_7_pw_fused_scales,
					layer_7_pw_fused_scales_log_2_shifts,
					layer_7_pw_relu_6_fused_scales,
					layer_7_pw_fused_zero_points, d_in_out,
					layer_7_pw_specs.layer_activation,
					bottleneck_1_projection_layer_index);
		}
	}

//	if (dw_kernel_starting_w >= 0)
//	{
//		bottleneck_1_update_previous_pass_buffer(previous_pass_dw_input_w,
//												 dw_lower_buffer, dw_kernel_starting_w,
//												 bottleneck_1_expanded_ifms_depth - 1);
//	}

	shift_dw_ifms_buffer_horizontally_3x3_s2(dw_lower_buffer,
											 bottleneck_1_expanded_ifms_depth);
	bottleneck_1_copy_projection_kernel_output_buffer(projection_kernel_output_buffer,
													  projection_kernel_output_buffer_prev);
}

#endif