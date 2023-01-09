void mob_v2_bottleneck_1(fms_dt bottleneck_input[], fms_dt bottleneck_output[],
						 fms_dt r_previous_pass_dw_input[],
						 fms_dt w_previous_pass_dw_input[],
						 int starting_h,
						 int starting_w,
						 const int first_fill_from_left_offset)
{

	// #pragma HLS INLINE off

	fms_dt dw_input_buffer[bottleneck_1_dw_filter_dim * bottleneck_1_dw_filter_dim]; // depth = 1
																					 ////#pragma HLS ARRAY_PARTITION variable = dw_input_buffer type = complete dim = 0

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

	pss_dt projection_results_buffer[bottleneck_1_ofms_depth];

	for (int d_in_out = 0; d_in_out < bottleneck_1_expanded_ifms_depth; d_in_out++)
	{
		// #pragma HLS PIPELINE
		fms_dt expansion_results_buffer[4]; // TODO  atrides * strides
		weights_dt projection_kernel_weights[bottleneck_1_ofms_depth];

		expansion_layer_normalization.fused_scales =
			layer_3_pw_fused_scales[d_in_out];
		expansion_layer_normalization.fused_scales_log_2_shift =
			layer_3_pw_fused_scales_log_2_shifts[d_in_out];
		expansion_layer_normalization.relu_6_fused_scale =
			layer_3_pw_relu_6_fused_scales[d_in_out];
		expansion_layer_normalization.fused_zero_point =
			layer_3_pw_fused_zero_points[d_in_out];

		for (int p_h = 0; p_h < bottleneck_1_expansion_parallelism_h; p_h++)
		{
			// #pragma HLS UNROLL
			//  TODO first column, do not run dw
			for (int p_w = 0; p_w < bottleneck_1_expansion_parallelism_w; p_w++)
			{
				// #pragma HLS UNROLL
				if (starting_h + p_h < bottleneck_1_ifms_height + bottleneck_1_dw_padding_top &&
					starting_w + p_w >= bottleneck_1_dw_padding_left)
				{
					pss_dt expansion_pss = expansion_kernel(bottleneck_input,
															bottleneck_1_ifms_depth, pw_weights_3[d_in_out],
															d_in_out, p_h, p_w,
															bottleneck_1_expansion_parallelism_w);
					expansion_results_buffer[p_h * bottleneck_1_expansion_parallelism_w + p_w] =
						pw_relu_norm(expansion_pss,
									 expansion_layer_normalization,
									 layer_3_activation);
				}
				else
				{
					expansion_results_buffer[p_h * bottleneck_1_expansion_parallelism_w + p_w] =
						current_dw_ifms_zero_point;
				}
			}
		}

		fill_dw_ifms_buffer_upper_part(dw_input_buffer, r_previous_pass_dw_input,
									   bottleneck_1_dw_strides, bottleneck_1_dw_filter_dim, starting_w * bottleneck_1_dw_strides,
									   bottleneck_1_ifms_width, bottleneck_1_expanded_ifms_depth, d_in_out,
									   bottleneck_1_dw_padding_left);

		fill_dw_ifms_buffer_lower_part(dw_input_buffer,
									   expansion_results_buffer, bottleneck_1_dw_strides, bottleneck_1_dw_filter_dim, d_in_out,
									   0, 0); // DOUBLE CHECK last two params
		update_dw_ifms_buffer_upper_part(w_previous_pass_dw_input,
										 expansion_results_buffer, bottleneck_1_dw_strides, bottleneck_1_dw_filter_dim,
										 starting_w * bottleneck_1_dw_strides, bottleneck_1_ifms_width, d_in_out, d_in_out,
										 bottleneck_1_dw_padding_left, first_fill_from_left_offset, 1); // DOUBLE CHECK last param

		dw_layer_normalization.fused_scales = layer_4_dw_fused_scales[d_in_out];
		dw_layer_normalization.fused_scales_log_2_shift =
			layer_4_dw_fused_scales_log_2_shifts[d_in_out];
		dw_layer_normalization.relu_6_fused_scale =
			layer_4_dw_relu_6_fused_scales[d_in_out];
		dw_layer_normalization.fused_zero_point =
			layer_4_dw_fused_zero_points[d_in_out];

		dw_pss_dt dw_pss = dw_kernel(dw_input_buffer, dw_weights_4, bottleneck_1_dw_filter_dim,
									 d_in_out);
		fms_dt dw_result = dw_relu_norm(dw_pss, dw_layer_normalization,
										6);

		shift_dw_ifms_buffer_horizontally(dw_input_buffer, bottleneck_1_dw_strides,
										  bottleneck_1_dw_filter_dim, d_in_out);

		bottleneck_1_fill_projection_kernel_weights(pw_weights_5, projection_kernel_weights, d_in_out);
		projection_kernel(dw_result, bottleneck_1_ofms_depth,
						  projection_kernel_weights, projection_results_buffer, d_in_out);
		//	}
		normalize_projection_kernel_output(projection_results_buffer,
										   bottleneck_output, layer_5_pw_fused_scales,
										   layer_5_pw_fused_scales_log_2_shifts,
										   layer_5_pw_relu_6_fused_scales,
										   layer_5_pw_fused_zero_points, bottleneck_1_ofms_depth,
										   layer_5_activation, bottleneck_1_projection_layer_index);
	}
}
