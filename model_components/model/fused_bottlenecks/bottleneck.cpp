#include "bottleneck.h"

void mob_v2_bottleneck_0(fms_dt bottleneck_input[],
						 fms_dt bottleneck_output[],
						 fms_dt r_previous_pass_dw_input[],
						 fms_dt w_previous_pass_dw_input[], int starting_h, int starting_w)
{
#pragma HLS INLINE off

	fms_dt dw_input_buffer[bottleneck_0_dw_filter_dim * bottleneck_0_dw_filter_dim]; // depth = 1
																					 //#pragma HLS ARRAY_PARTITION variable = dw_input_buffer type = complete dim = 0

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

	for (int d_in_out = 0; d_in_out < bottleneck_0_expanded_ifms_depth; d_in_out++)
	{
#pragma HLS PIPELINE
		fms_dt expansion_results_buffer[bottleneck_0_rows_at_once * bottleneck_0_rows_at_once];

		expansion_layer_normalization.fused_scales =
			layer_0_s_fused_scales[d_in_out];
		expansion_layer_normalization.fused_scales_log_2_shift =
			expansion_layer_fused_scales_log_2_shifts[d_in_out];
		expansion_layer_normalization.relu_6_fused_scale =
			layer_0_s_fused_scales_log_2_shifts[d_in_out];
		expansion_layer_normalization.fused_zero_point =
			layer_0_s_fused_zero_points[d_in_out];

		for (int p_h = 0; p_h < bottleneck_0_expansion_parallelism_h; p_h++)
		{
#pragma HLS UNROLL
			// TODO first column, do not run dw
			for (int p_w = 0; p_w < bottleneck_0_expansion_parallelism_w; p_w++)
			{
#pragma HLS UNROLL
				if (starting_h + p_h < bottleneck_0_ifms_height + bottleneck_0_dw_padding_top &&
					starting_w + p_w >= bottleneck_0_dw_padding_left)
				{
					pss_dt expansion_pss = conv_kernel(bottleneck_input, weights_0, layer_0_s_filter_dim,
													   d_in_out);

					expansion_results_buffer[p_h * bottleneck_0_expansion_parallelism_w + p_w] =
						conv_relu_norm(expansion_pss,
									   expansion_layer_normalization,
									   bottleneck_0_expansion_layer_relu);
				}
				else
				{
					expansion_results_buffer[p_h * bottleneck_0_expansion_parallelism_w + p_w] =
						current_dw_ifms_zero_point;
				}
			}
		}

		fill_dw_ifms_buffer_upper_part(dw_input_buffer, r_previous_pass_dw_input,
									   bottleneck_0_dw_strides, bottleneck_0_dw_filter_dim, starting_w * bottleneck_0_dw_strides,
									   bottleneck_0_ifms_width, bottleneck_0_expanded_ifms_depth, d_in_out,
									   bottleneck_0_dw_padding_left);

		fill_dw_ifms_buffer_lower_part(dw_input_buffer,
									   expansion_results_buffer, bottleneck_0_dw_strides, bottleneck_0_dw_filter_dim, d_in_out);
		update_dw_ifms_buffer_upper_part(w_previous_pass_dw_input,
										 expansion_results_buffer, bottleneck_0_dw_strides, bottleneck_0_dw_filter_dim,
										 starting_w * bottleneck_0_dw_strides, bottleneck_0_ifms_width, d_in_out, d_in_out,
										 bottleneck_0_dw_padding_left, 0);

		dw_layer_normalization.fused_scales = layer_2_fused_scales[d_in_out];
		dw_layer_normalization.fused_scales_log_2_shift =
			layer_2_fused_scales_log_2_shifts[d_in_out];
		dw_layer_normalization.relu_6_fused_scale =
			layer_2_relu_6_fused_scales[d_in_out];
		dw_layer_normalization.fused_zero_point =
			layer_2_fused_zero_points[d_in_out];

		dw_pss_dt dw_pss = dw_kernel(dw_input_buffer, dw_weights_2, bottleneck_0_dw_filter_dim,
									 d_in_out);
		fms_dt dw_result = dw_relu_norm(dw_pss, dw_layer_normalization,
										bottleneck_0_dw_layer_relu);

		shift_dw_ifms_buffer_horizontally(dw_input_buffer, bottleneck_0_dw_strides,
										  bottleneck_0_dw_filter_dim, d_in_out);

		projection_kernel(dw_result, bottleneck_0_ofms_depth,
						  pw_weights_3, projection_results_buffer, d_in_out);
		//	}
		normalize_projection_kernel_output(projection_results_buffer,
										   bottleneck_output, layer_3_fused_scales,
										   layer_3_fused_scales_log_2_shifts,
										   layer_3_relu_6_fused_scales,
										   layer_3_fused_zero_points, bottleneck_0_ofms_depth,
										   layer_3_relu, bottleneck_0_projection_layer_index);
	}
}

void mob_v2_bottleneck(fms_dt bottleneck_input[], fms_dt bottleneck_output[],
					   fms_dt r_previous_pass_dw_input[],
					   fms_dt w_previous_pass_dw_input[],
					   const weights_dt expansion_layer_weights[][max_of_bottlenecks_expansion_layers_depths],
					   const dw_weights_dt dw_weights[][max_dw_filter_area_in_a_chain],
					   const weights_dt projection_layer_weights[][max_of_bottlenecks_layers_depths],
					   const fused_scales_dt expansion_layer_fused_scales[],
					   const fused_scales_log_2_shifts_dt expansion_layer_fused_scales_log_2_shifts[],
					   const relu_6_fused_scales_dt expansion_layer_relu_6_fused_scales[],
					   const biases_dt expansion_layer_fused_zero_points[],
					   const fused_scales_dt dw_layer_fused_scales[],
					   const fused_scales_log_2_shifts_dt dw_layer_fused_scales_log_2_shifts[],
					   const relu_6_fused_scales_dt dw_layer_relu_6_fused_scales[],
					   const biases_dt dw_layer_fused_zero_points[],
					   const fused_scales_dt projection_layer_fused_scales[],
					   const fused_scales_log_2_shifts_dt projection_layer_fused_scales_log_2_shifts[],
					   const relu_6_fused_scales_dt projection_layer_relu_6_fused_scales[],
					   const biases_dt projection_layer_fused_zero_points[],
					   const int bottleneck_ifms_depth, const int bottleneck_ifms_height,
					   const int bottleneck_ifms_width, const int bottleneck_ofms_depth,
					   const int bottleneck_ofms_width, const int expanded_ifms_depth,
					   const int dw_filter_dim, const int strides, int starting_h,
					   int starting_w, const int bottleneck_expansion_parallelism_h,
					   const int bottleneck_expansion_parallelism_w,
					   const int bottleneck_expansion_layer_index,
					   const int bottleneck_dw_layer_index,
					   const int bottleneck_projection_layer_index,
					   const int expansion_layer_relu, const int dw_layer_relu,
					   const int projection_layer_relu, const int padding_left,
					   const int padding_right, const int padding_top,
					   const int padding_bottom, const int first_fill_from_left_offset)
{

#pragma HLS INLINE off

	fms_dt dw_input_buffer[dw_filter_dim * dw_filter_dim]; // depth = 1
														   //#pragma HLS ARRAY_PARTITION variable = dw_input_buffer type = complete dim = 0

	const fms_dt expansion_layer_ofms_zero_point =
		conv_fms_zero_points[bottleneck_expansion_layer_index + 1];
	const rec_scales_dt expansion_layer_ofms_scale_rec =
		conv_fms_scales_rec[bottleneck_expansion_layer_index + 1];
	const rec_scales_dt expansion_layer_ofms_scale =
		conv_fms_scales[bottleneck_expansion_layer_index + 1];

	const fms_dt dw_layer_ofms_zero_point =
		conv_fms_zero_points[bottleneck_dw_layer_index + 1];
	const rec_scales_dt dw_layer_ofms_scale_rec =
		conv_fms_scales_rec[bottleneck_dw_layer_index + 1];
	const rec_scales_dt dw_layer_ofms_scale =
		conv_fms_scales[bottleneck_dw_layer_index + 1];
	const fms_dt current_dw_ifms_zero_point =
		conv_fms_zero_points[bottleneck_dw_layer_index];

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

	pss_dt projection_results_buffer[bottleneck_ofms_depth];

	for (int d_in_out = 0; d_in_out < expanded_ifms_depth; d_in_out++)
	{
#pragma HLS PIPELINE
		fms_dt expansion_results_buffer[4]; // TODO  atrides * strides

		expansion_layer_normalization.fused_scales =
			expansion_layer_fused_scales[d_in_out];
		expansion_layer_normalization.fused_scales_log_2_shift =
			expansion_layer_fused_scales_log_2_shifts[d_in_out];
		expansion_layer_normalization.relu_6_fused_scale =
			expansion_layer_relu_6_fused_scales[d_in_out];
		expansion_layer_normalization.fused_zero_point =
			expansion_layer_fused_zero_points[d_in_out];

		for (int p_h = 0; p_h < bottleneck_expansion_parallelism_h; p_h++)
		{
#pragma HLS UNROLL
			// TODO first column, do not run dw
			for (int p_w = 0; p_w < bottleneck_expansion_parallelism_w; p_w++)
			{
#pragma HLS UNROLL
				if (starting_h + p_h < bottleneck_ifms_height + padding_top && starting_w + p_w >= padding_left)
				{
					pss_dt expansion_pss = expansion_kernel(bottleneck_input,
															bottleneck_ifms_depth, expansion_layer_weights,
															d_in_out, p_h, p_w,
															bottleneck_expansion_parallelism_w);
					expansion_results_buffer[p_h * bottleneck_expansion_parallelism_w + p_w] =
						pw_relu_norm(expansion_pss,
									 expansion_layer_normalization,
									 expansion_layer_relu);
				}
				else
				{
					expansion_results_buffer[p_h * bottleneck_expansion_parallelism_w + p_w] =
						current_dw_ifms_zero_point;
				}
			}
		}

		fill_dw_ifms_buffer_upper_part(dw_input_buffer, r_previous_pass_dw_input,
									   strides, dw_filter_dim, starting_w * strides,
									   bottleneck_ifms_width, expanded_ifms_depth, d_in_out,
									   padding_left);

		fill_dw_ifms_buffer_lower_part(dw_input_buffer,
									   expansion_results_buffer, strides, dw_filter_dim, d_in_out);
		update_dw_ifms_buffer_upper_part(w_previous_pass_dw_input,
										 expansion_results_buffer, strides, dw_filter_dim,
										 starting_w * strides, bottleneck_ifms_width, d_in_out, d_in_out,
										 padding_left, first_fill_from_left_offset);

		dw_layer_normalization.fused_scales = dw_layer_fused_scales[d_in_out];
		dw_layer_normalization.fused_scales_log_2_shift =
			dw_layer_fused_scales_log_2_shifts[d_in_out];
		dw_layer_normalization.relu_6_fused_scale =
			dw_layer_relu_6_fused_scales[d_in_out];
		dw_layer_normalization.fused_zero_point =
			dw_layer_fused_zero_points[d_in_out];

		dw_pss_dt dw_pss = dw_kernel(dw_input_buffer, dw_weights, dw_filter_dim,
									 d_in_out);
		fms_dt dw_result = dw_relu_norm(dw_pss, dw_layer_normalization,
										dw_layer_relu);

		shift_dw_ifms_buffer_horizontally(dw_input_buffer, strides,
										  dw_filter_dim, d_in_out);

		projection_kernel(dw_result, bottleneck_ofms_depth,
						  projection_layer_weights, projection_results_buffer, d_in_out);
		//	}
		normalize_projection_kernel_output(projection_results_buffer,
										   bottleneck_output, projection_layer_fused_scales,
										   projection_layer_fused_scales_log_2_shifts,
										   projection_layer_relu_6_fused_scales,
										   projection_layer_fused_zero_points, bottleneck_ofms_depth,
										   projection_layer_relu, bottleneck_projection_layer_index);
	}
}
