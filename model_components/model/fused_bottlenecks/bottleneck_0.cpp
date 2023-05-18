#include "../../basic_defs/simulation_constants.h"

#if FIRST_PART_IMPLEMENTATION == BOTTLENECK_CHAIN_MODE && !ONLY_SEML

#include "bottleneck.h"

void bottleneck_0_fill_projection_kernel_weights(
	const weights_dt layer_weights[][bottleneck_0_expanded_ifms_depth],
	weights_dt kernel_weights[], int d)
{
#pragma HLS INLINE

bottleneck_0_fill_projection_kernel_weights:
	for (int filter_index = 0;
		 filter_index < bottleneck_0_ofms_depth; filter_index++)
	{
#pragma HLS UNROLL
		kernel_weights[filter_index] = layer_weights[filter_index][d];
	}
}

void bottleneck_0_padding_top_right(
	fms_dt previous_pass_dw_input[][bottleneck_0_inter_pass_dw_input_height][bottleneck_0_inter_pass_dw_input_width],
	fms_dt zero_point)
{
bottleneck_0_padding_top_right:
	for (int d = 0;
		 d < bottleneck_0_expanded_ifms_depth; d++)
	{
		// padding_top
		for (int h = 0; h < bottleneck_0_dw_padding_top; h++)
		{
			for (int w = 0; w < bottleneck_0_inter_pass_dw_input_width; w++)
			{
				previous_pass_dw_input[d][h][w] = zero_point;
			}
		}
		// padding_right
		for (int h = bottleneck_0_dw_padding_top; h < bottleneck_0_inter_pass_dw_input_height; h++)
		{
			for (int w = bottleneck_0_inter_pass_dw_input_width - bottleneck_0_dw_padding_right;
				 w < bottleneck_0_inter_pass_dw_input_width; w++)
			{
				previous_pass_dw_input[d][h][w] = zero_point;
			}
		}
	}
}

void bottleneck_0_do_padding_left(
	fms_dt previous_pass_dw_input[][bottleneck_0_inter_pass_dw_input_height][bottleneck_0_inter_pass_dw_input_width],
	fms_dt dw_lower_buffer[][bottleneck_0_dw_filter_dim],
	fms_dt zero_point)
{
bottleneck_0_do_padding_left:
	for (int d = 0;
		 d < bottleneck_0_expanded_ifms_depth; d++)
	{
		previous_pass_dw_input[d][1][0] = zero_point;
		dw_lower_buffer[d][bottleneck_0_dw_filter_dim - bottleneck_0_dw_strides] =
			zero_point;
	}
}

void bottleneck_0_update_previous_pass_buffer(
	fms_dt previous_pass_dw_input_slice[bottleneck_0_inter_pass_dw_input_height][bottleneck_0_inter_pass_dw_input_width],
	fms_dt dw_lower_buffer_slice[bottleneck_0_dw_filter_dim], int offset_w,
	int starting_h)
{
#pragma HLS INLINE
	//	bottleneck_0_update_previous_pass_buffer: for (int d = 0; d < bottleneck_0_expanded_ifms_depth; d++)
	//	{
	//#pragma HLS PIPELINE
	if (starting_h != 0)
	{
		previous_pass_dw_input_slice[0][offset_w] =
			previous_pass_dw_input_slice[1][offset_w];
	}
	previous_pass_dw_input_slice[1][offset_w] = dw_lower_buffer_slice[0];
	//		if (offset_w + 1 == bottleneck_0_ifms_width / first_conv_layer_specs.strides)//to do, handle once at another place
	//		{
	//			previous_pass_dw_input[d][1][offset_w + 1] = dw_lower_buffer[d][1];
	//			previous_pass_dw_input[d][1][offset_w + 2] = dw_lower_buffer[d][2];
	//		}
	//	}
}

void bottleneck_0_fill_dw_input(
	fms_dt previous_pass_dw_input_slice[bottleneck_0_inter_pass_dw_input_height][bottleneck_0_inter_pass_dw_input_width],
	fms_dt dw_lower_buffer_slice[bottleneck_0_dw_filter_dim],
	fms_dt dw_input_buffer[], fms_dt expansion_layer_result,
	int filling_w_offset, int absolute_starting_h)
{

	if (filling_w_offset >= 0 && absolute_starting_h != 0)
	{
		dw_input_buffer[0] =
			previous_pass_dw_input_slice[0][filling_w_offset];
		dw_input_buffer[1] =
			previous_pass_dw_input_slice[0][filling_w_offset + 1];
		dw_input_buffer[2] =
			previous_pass_dw_input_slice[0][filling_w_offset + 2];
		dw_input_buffer[3] =
			previous_pass_dw_input_slice[1][filling_w_offset];
		dw_input_buffer[4] =
			previous_pass_dw_input_slice[1][filling_w_offset + 1];
		dw_input_buffer[5] =
			previous_pass_dw_input_slice[1][filling_w_offset + 2];

		dw_input_buffer[6] = dw_lower_buffer_slice[0];
		dw_input_buffer[7] = dw_lower_buffer_slice[1];
	}
	// fill lower part of dw_input
	dw_input_buffer[8] = expansion_layer_result;
	dw_lower_buffer_slice[2] = expansion_layer_result;
}

void bottleneck_0_copy_projection_kernel_output_buffer(
	pss_dt projection_kernel_output_buffer[],
	pss_dt projection_kernel_output_buffer_prev[])
{
#pragma HLS PIPELINE

	copy_projection_kernel_output_buffer(projection_kernel_output_buffer,
										 projection_kernel_output_buffer_prev, bottleneck_0_ofms_depth);
}

void mob_v2_bottleneck_0(fms_dt bottleneck_input[bottleneck_0_input_buffer_size],
						 pss_dt projection_kernel_output_buffer[bottleneck_0_ofms_depth],
						 pss_dt projection_kernel_output_buffer_prev[bottleneck_0_ofms_depth],
						 fms_dt bottleneck_0_1_communication_buffer[bottleneck_0_ofms_depth][bottleneck_0_rows_at_once][bottleneck_0_ofms_width],
						 fms_dt previous_pass_dw_input[][bottleneck_0_inter_pass_dw_input_height][bottleneck_0_inter_pass_dw_input_width],
						 fms_dt dw_lower_buffer[][bottleneck_0_dw_filter_dim],
						 const int starting_h, const int h_in_communication_buffer,
						 const int expansion_kernel_starting_w)
{
#pragma HLS INLINE off

	const fms_dt expansion_layer_ofms_zero_point = first_conv_layer_specs.layer_ofms_zero_point;
	const rec_scales_dt expansion_layer_ofms_scale = first_conv_layer_specs.layer_ofms_scale;

	const fms_dt dw_layer_ofms_zero_point = layer_2_dw_specs.layer_ofms_zero_point;
	const rec_scales_dt dw_layer_ofms_scale = layer_2_dw_specs.layer_ofms_scale;
	const fms_dt current_dw_ifms_zero_point = layer_2_dw_specs.layer_ifms_zero_point; // todo
												 // conv_fms_zero_points[bottleneck_0_dw_layer_index];

	fms_quantization_scheme expansion_layer_normalization;
	expansion_layer_normalization.ofm_zero_point =
		expansion_layer_ofms_zero_point;
	expansion_layer_normalization.ofm_scale = expansion_layer_ofms_scale;

	fms_quantization_scheme dw_layer_normalization;
	dw_layer_normalization.ofm_zero_point = dw_layer_ofms_zero_point;
	dw_layer_normalization.ofm_scale = dw_layer_ofms_scale;

	const int dw_kernel_starting_w = expansion_kernel_starting_w - (bottleneck_0_dw_filter_dim - bottleneck_0_dw_padding_left) + 1;
	const int prev_projection_kernel_starting_w = dw_kernel_starting_w - 1;

	fms_dt dw_input_buffer[bottleneck_0_dw_filter_dim * bottleneck_0_dw_filter_dim];
#pragma HLS ARRAY_PARTITION variable = dw_input_buffer type = complete dim = 0

mob_v2_bottleneck_0:
	for (int d_in_out = 0;
		 d_in_out < bottleneck_0_expanded_ifms_depth; d_in_out++)
	{
#pragma HLS PIPELINE
		weights_dt projection_kernel_weights[bottleneck_0_ofms_depth];

		expansion_layer_normalization.fused_scales =
			first_conv_layer_fused_scales[d_in_out];
		expansion_layer_normalization.fused_scales_log_2_shift =
			first_conv_layer_fused_scales_log_2_shifts[d_in_out];
		expansion_layer_normalization.relu_6_fused_scale =
			first_conv_layer_fused_scales_log_2_shifts[d_in_out];
		expansion_layer_normalization.fused_zero_point =
			first_conv_layer_fused_zero_points[d_in_out];
		expansion_layer_normalization.layer_0_relu_6_fused_scale =
			first_conv_layer_relu_6_fused_scales[d_in_out];
		fms_dt expansion_result;

		if (starting_h * first_conv_layer_specs.strides < bottleneck_0_ifms_height + first_conv_layer_specs.padding_top &&
			expansion_kernel_starting_w < first_conv_layer_specs.padding_right + bottleneck_0_ifms_width / first_conv_layer_specs.strides - 1)
		{
			pss_dt expansion_pss = conv_kernel(bottleneck_input, first_layer_weights,
											   first_conv_layer_filter_dim, d_in_out);
			expansion_result = conv_relu_norm(expansion_pss,
											  expansion_layer_normalization,
											  bottleneck_0_expansion_layer_relu);
			const int ifms_buffer_hw = bottlenck_0_input_buffer_height * bottlenck_0_input_buffer_width;
		}
		else
		{
			expansion_result = current_dw_ifms_zero_point;
		}
		bottleneck_0_fill_dw_input(previous_pass_dw_input[d_in_out], dw_lower_buffer[d_in_out],
								   dw_input_buffer, expansion_result, dw_kernel_starting_w,
								   starting_h);

		dw_layer_normalization.fused_scales = layer_2_dw_fused_scales[d_in_out];
		dw_layer_normalization.fused_scales_log_2_shift =
			layer_2_dw_fused_scales_log_2_shifts[d_in_out];
		dw_layer_normalization.relu_6_fused_scale =
			layer_2_dw_relu_6_fused_scales[d_in_out];
		dw_layer_normalization.fused_zero_point =
			layer_2_dw_fused_zero_points[d_in_out];

		dw_pss_dt dw_pss = dw_kernel(dw_input_buffer, dw_weights_2[d_in_out],
									 bottleneck_0_dw_filter_dim);
		fms_dt dw_result = dw_relu_norm(dw_pss, dw_layer_normalization,
										bottleneck_0_dw_layer_relu);

		if (dw_kernel_starting_w >= 0 && d_in_out > 0)
		{
			bottleneck_0_update_previous_pass_buffer(previous_pass_dw_input[d_in_out - 1],
													 dw_lower_buffer[d_in_out - 1], dw_kernel_starting_w, starting_h);
		}
		bottleneck_0_fill_projection_kernel_weights(pw_weights_3,
													projection_kernel_weights, d_in_out);
		projection_kernel(dw_result, bottleneck_0_ofms_depth,
						  projection_kernel_weights, projection_kernel_output_buffer,
						  d_in_out);

		if (prev_projection_kernel_starting_w >= 0 && d_in_out < bottleneck_0_ofms_depth)
		{
			// if(starting_h == 1 && d_in_out == 15)cout<<starting_w - 2<<"\n";
			bottleneck_0_1_communication_buffer[d_in_out][h_in_communication_buffer][prev_projection_kernel_starting_w] = normalize_projection_kernel_output(
				projection_kernel_output_buffer_prev,
				layer_3_pw_fused_scales,
				layer_3_pw_fused_scales_log_2_shifts,
				layer_3_pw_relu_6_fused_scales,
				layer_3_pw_fused_zero_points, d_in_out, layer_3_pw_specs.layer_activation, layer_3_pw_specs);
		}
	}

	if (dw_kernel_starting_w >= 0)
	{
		bottleneck_0_update_previous_pass_buffer(previous_pass_dw_input[bottleneck_0_expanded_ifms_depth - 1],
												 dw_lower_buffer[bottleneck_0_expanded_ifms_depth - 1], dw_kernel_starting_w, starting_h);
	}
	shift_dw_ifms_buffer_horizontally_3x3_s1(dw_lower_buffer,
											 bottleneck_0_expanded_ifms_depth);
	bottleneck_0_copy_projection_kernel_output_buffer(
		projection_kernel_output_buffer,
		projection_kernel_output_buffer_prev);
}

#endif