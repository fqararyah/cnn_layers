#include "../headers/layers_imp_common_includes.h"
#include "../headers/pw_conv.h"

#if FIBHA_VERSION == 1

void pw_fill_channels_tile(fms_dt channels[max_fms_size],
						   fms_dt channels_tile[pw_tile_d][pw_tile_h][pw_tile_w], const int starting_index,
						   int starting_d, const int layer_conv_d)
{
#pragma HLS INLINE

	for (int t_d = 0; t_d < pw_tile_d; t_d++)
	{
#pragma HLS UNROLL
		for (int t_h = 0; t_h < pw_tile_h; t_h++)
		{
#pragma HLS UNROLL
			for (int t_w = 0; t_w < pw_tile_w; t_w++)
			{
#pragma HLS UNROLL
				channels_tile[t_d][t_h][t_w] = channels[starting_index + t_d * pw_tile_hw + t_h * pw_tile_w + t_w];
			}
		}
	}
}

void scale_pss_tile(fms_dt tmp_channels[max_fms_size],
					pss_dt pss_tile[pw_conv_parallelism_out][pw_tile_h][pw_tile_w],
					fms_dt result_tile_scaled[pw_conv_parallelism_out][pw_tile_h][pw_tile_w],
					const int layer_relu,
					fused_scales_dt fused_scales_buffer[pw_conv_parallelism_out],
					fused_scales_log_2_shifts_dt fused_scales_log_2_shifts_buffer[pw_conv_parallelism_out],
					relu_6_fused_scales_dt relu_6_fused_scales_buffer[pw_conv_parallelism_out],
					biases_dt fused_zero_points_buffer[pw_conv_parallelism_out],
					const int num_of_tiles_hw, const int tile_index, const int layer,
					const layer_specs layer_specs_struct)
{
#pragma HLS INLINE off

	if (tile_index >= 0)
	{
		int num_of_tiles_processed_in_parallel = pw_conv_parallelism_out / pw_tile_d;
		if (pw_conv_parallelism_out < pw_tile_d)
		{
			num_of_tiles_processed_in_parallel = 1;
		}
		else if (pw_conv_parallelism_out % pw_tile_d != 0)
		{
			num_of_tiles_processed_in_parallel = 1 + pw_conv_parallelism_out / pw_tile_d;
		}

		scales_dt skip_connection_other_layer_scale = layer_specs_struct.skip_connection_other_layer_scale;
		biases_dt skip_connection_other_layer_zero_point = layer_specs_struct.skip_connection_other_layer_zero_point;

		rec_scales_dt add_layer_scale_reciprocal = layer_specs_struct.add_layer_scale_reciprocal;
		biases_dt add_layer_zero_point = layer_specs_struct.add_layer_zero_point;

		fms_quantization_scheme normalization;
		normalization.ofm_zero_point = layer_specs_struct.layer_ofms_zero_point;
		normalization.ofm_scale = layer_specs_struct.layer_ofms_scale;

	pss_to_fms_tile_o_d:
		for (int tile_offset = 0;
			 tile_offset < num_of_tiles_processed_in_parallel;
			 tile_offset++)
		{
		tile_d:
			for (int t_d = 0; t_d < pw_tile_d; t_d++)
			{
				const int current_tile_indx = (tile_index + tile_offset * num_of_tiles_hw) * pw_tile_size;
				const int in_tile_index = tile_offset * pw_tile_d + t_d;
				normalization.fused_zero_point =
					fused_zero_points_buffer[in_tile_index];
				normalization.fused_scales = fused_scales_buffer[in_tile_index];
				normalization.fused_scales_log_2_shift =
					fused_scales_log_2_shifts_buffer[in_tile_index];
				normalization.relu_6_fused_scale =
					relu_6_fused_scales_buffer[in_tile_index];
			tile_h:
				for (int t_h = 0; t_h < pw_tile_h; t_h++)
				{
#pragma HLS PIPELINE
				tile_w:
					for (int t_w = 0; t_w < pw_tile_w; t_w++)
					{
#pragma HLS UNROLL
						const int to_read_from_index = current_tile_indx + t_d * pw_tile_hw + t_h * pw_tile_w + t_w;

						fms_dt scaled_val;
						if (layer_specs_struct.fused_with_add == 0)
						{
							scaled_val =
								pw_relu_norm(
									pss_tile[tile_offset * pw_tile_d + t_d][t_h][t_w],
									normalization, layer_relu);
						}
						else
						{
							pss_f_dt tmp_channels_scaled_val =
								skip_connection_other_layer_scale * (tmp_channels[to_read_from_index] - skip_connection_other_layer_zero_point);
							pss_f_dt scaled_tmp =
								pw_relu_norm_no_q_no_relu(
									pss_tile[tile_offset * pw_tile_d + t_d][t_h][t_w],
									normalization, layer_relu);

							pss_f_dt addition_result = (scaled_tmp + tmp_channels_scaled_val) * add_layer_scale_reciprocal + add_layer_zero_point;
							addition_result = addition_result + quant_half - (addition_result < 0);
							scaled_val = clamp(addition_result);
						}

						result_tile_scaled[tile_offset * pw_tile_d + t_d][t_h][t_w] =
							scaled_val;
					}
				}
			}
		}
	}
}

void pw_write_results_tile(
	fms_dt result_tile_scaled[pw_conv_parallelism_out][pw_tile_h][pw_tile_w],
	fms_dt results[max_fms_size], int tile_indx,
	fms_dt tmp_channels[max_tmp_fms_size],
	pss_f_dt tmp_channels_scaled_tile[pw_conv_parallelism_out][pw_tile_h][pw_tile_w],
	int starting_d, const int layer_num_filters,
	const int layer_relu, int layer, const int num_of_tiles_hw,
	const layer_specs layer_specs_struct)
{
#pragma HLS INLINE off

	// read_write = 1 when the current layer is the one that is directly connected to the OFMs that have a residual connection to a previous layer
	// read_write = 2 when the current layer has a residual connection
	if (tile_indx >= 0)
	{
		int num_of_tiles_processed_in_parallel = pw_conv_parallelism_out / pw_tile_d;

		rec_scales_dt add_layer_scale_reciprocal = layer_specs_struct.add_layer_scale_reciprocal;
		biases_dt add_layer_zero_point = layer_specs_struct.add_layer_zero_point;

	pw_write_results_tile_o_d:
		for (int tile_offset = 0;
			 tile_offset < num_of_tiles_processed_in_parallel;
			 tile_offset++)
		{
#pragma HLS PIPELINE
			const int current_tile_indx = (tile_indx + tile_offset * num_of_tiles_hw) * pw_tile_size;
		pw_write_results_tile_d:
			for (int t_d = 0; t_d < pw_tile_d; t_d++)
			{
#pragma HLS UNROLL
				if (t_d + starting_d < layer_num_filters)
				{
				pw_write_results_tile_h:
					for (int t_h = 0; t_h < pw_tile_h;
						 t_h++)
					{
#pragma HLS UNROLL
					pw_write_results_tile_w:
						for (int t_w = 0;
							 t_w < pw_tile_w; t_w++)
						{
#pragma HLS UNROLL
							const int to_write_at_index = current_tile_indx + t_d * pw_tile_hw + t_h * pw_tile_w + t_w;

							fms_dt scaled_val = result_tile_scaled[tile_offset * pw_tile_d + t_d][t_h][t_w];

							results[to_write_at_index] = (fms_dt)scaled_val;
							if (layer_specs_struct.write_to_tmp)
							{ // 2: expansion
								tmp_channels[to_write_at_index] = scaled_val;
							}
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
				 const int layer_num_fils)
{
#pragma HLS INLINE
pw_conv_eng_loops:
	for (int f_d = 0; f_d < pw_conv_parallelism_out; f_d++)
	{
#pragma HLS UNROLL
		for (int t_h = 0; t_h < pw_tile_h; t_h++)
		{
#pragma HLS UNROLL
			for (int t_w = 0; t_w < pw_tile_w; t_w++)
			{
#pragma HLS UNROLL
				if (starting_conv_d == 0)
				{
					results_tile[f_d][t_h][t_w] = 0;
				}
				for (int t_d = 0; t_d < pw_tile_d; t_d++)
				{
#pragma HLS UNROLL
					results_tile[f_d][t_h][t_w] += channels_tile[t_d][t_h][t_w] * weights_tile[f_d][starting_conv_d + t_d];
				}
				// #if DEBUGGING
				// 				if (t_h == 0 && t_w == 0)
				// 					cout << (int)channels_tile[t_h][t_w] << " * " << (int)weights_tile[f_d][starting_conv_d] << "\n";
				// #endif
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
					  int t_in_h, int t_in_w,
					  const int num_of_tiles_d_in)
{
#pragma HLS INLINE OFF

conv2_itd_loop:
	for (int td_i = 0; td_i < num_of_tiles_d_in; td_i++)
	{
#pragma HLS PIPELINE
		fms_dt channels_buffer[pw_tile_d][pw_tile_h][pw_tile_w];
#pragma HLS ARRAY_PARTITION variable = channels_buffer complete dim = 0

		pw_fill_channels_tile(channels, channels_buffer,
							  (td_i * num_of_tiles_hw + t_in_h * num_of_tiles_w + t_in_w) * pw_tile_size,
							  td_i * pw_tile_d, layer_conv_d);

		pw_conv_eng(channels_buffer, weights_tile, results_tile,
					td_i * pw_tile_d, td_o * pw_conv_parallelism_out, layer_conv_d,
					layer_num_fils);
	}
}

void copy_pss_tile(
	pss_dt src_pss_tile[pw_conv_parallelism_out][pw_tile_h][pw_tile_w],
	pss_dt dst_pss_tile[pw_conv_parallelism_out][pw_tile_h][pw_tile_w])
{

#pragma HLS INLINE off

pw_conv_eng_loops:
	for (int f_d = 0; f_d < pw_conv_parallelism_out; f_d++)
	{
#pragma HLS PIPELINE
		for (int t_h = 0; t_h < pw_tile_h; t_h++)
		{
#pragma HLS UNROLL
			for (int t_w = 0; t_w < pw_tile_w; t_w++)
			{
#pragma HLS UNROLL
				dst_pss_tile[f_d][t_h][t_w] = src_pss_tile[f_d][t_h][t_w];
			}
		}
	}
}

void do_conv(weights_dt weights_tile[pw_conv_parallelism_out][max_conv_d],
			 fms_dt channels[max_fms_size], fms_dt result[max_fms_size],
			 const int layer, const int layer_conv_d, const int layer_num_fils,
			 const int num_of_tiles_d_in, const int num_of_tiles_d_out,
			 const int num_of_tiles_h, const int num_of_tiles_w,
			 fms_dt tmp_channels[max_tmp_fms_size],
			 const int num_of_weight_groups,
			 const int layer_weights_offset, const int layer_relu,
			 fused_scales_dt fused_scales_buffer[pw_conv_parallelism_out],
			 fused_scales_log_2_shifts_dt fused_scales_log_2_shifts_buffer[pw_conv_parallelism_out],
			 relu_6_fused_scales_dt relu_6_fused_scales_buffer[pw_conv_parallelism_out],
			 biases_dt fused_zero_points_buffer[pw_conv_parallelism_out], int td_o,
			 const layer_specs layer_specs_struct)
{

#pragma HLS INLINE off

	fms_quantization_scheme normalization = {0, 0, 0, 0};
	const int num_of_tiles_hw = num_of_tiles_h * num_of_tiles_w;

	pss_f_dt tmp_channels_scaled_tile[pw_conv_parallelism_out][pw_tile_h][pw_tile_w];
#pragma HLS ARRAY_PARTITION variable = tmp_channels_scaled_tile complete dim = 3

	pss_dt results_tile[pw_conv_parallelism_out][pw_tile_h][pw_tile_w];
	pss_dt prev_results_tile[pw_conv_parallelism_out][pw_tile_h][pw_tile_w];
	fms_dt scaled_result_tile[pw_conv_parallelism_out][pw_tile_h][pw_tile_w];
#pragma HLS ARRAY_PARTITION variable = results_tile complete dim = 0
#pragma HLS ARRAY_PARTITION variable = prev_results_tile complete dim = 1
#pragma HLS ARRAY_PARTITION variable = prev_results_tile complete dim = 2
#pragma HLS ARRAY_PARTITION variable = scaled_result_tile complete dim = 1
#pragma HLS ARRAY_PARTITION variable = scaled_result_tile complete dim = 2

	int prev_tile_index = -1;
conv2_ith_loop:
	for (int t_in_h = 0; t_in_h < num_of_tiles_h; t_in_h++)
	{
	//############width loop##############
	conv2_itw_loop:
		for (int t_in_w = 0; t_in_w < num_of_tiles_w;
			 t_in_w++)
		{
			//############depth loop##############
			int tile_index = td_o * (pw_conv_parallelism_out / pw_tile_d) * num_of_tiles_hw + t_in_h * num_of_tiles_w + t_in_w;

			scale_pss_tile(tmp_channels, prev_results_tile, scaled_result_tile,
						   layer_relu, fused_scales_buffer,
						   fused_scales_log_2_shifts_buffer,
						   relu_6_fused_scales_buffer, fused_zero_points_buffer,
						   num_of_tiles_hw, prev_tile_index, layer, layer_specs_struct);

			pw_conv_pipeline(channels, result, weights_tile, results_tile,
							 layer, layer_num_fils, layer_conv_d, num_of_tiles_hw,
							 num_of_tiles_w, td_o, t_in_h, t_in_w,
							 num_of_tiles_d_in);
			// #if DEBUGGING
			// if(t_in_h == 0 && t_in_w == 0){
			// 	cout<<results_tile[0][0][0]<<"\n";
			// }
			// #endif
			copy_pss_tile(results_tile, prev_results_tile);

			//

			pw_write_results_tile(scaled_result_tile, result,
								  prev_tile_index, tmp_channels, tmp_channels_scaled_tile,
								  td_o * pw_conv_parallelism_out, layer_num_fils,
								  layer_relu, layer, num_of_tiles_hw, layer_specs_struct);

			prev_tile_index = tile_index;
		}
	}
	scale_pss_tile(tmp_channels, prev_results_tile, scaled_result_tile,
				   layer_relu, fused_scales_buffer,
				   fused_scales_log_2_shifts_buffer,
				   relu_6_fused_scales_buffer, fused_zero_points_buffer,
				   num_of_tiles_hw, prev_tile_index, layer, layer_specs_struct);

	pw_write_results_tile(scaled_result_tile, result,
						  prev_tile_index, tmp_channels, tmp_channels_scaled_tile,
						  td_o * pw_conv_parallelism_out, layer_num_fils,
						  layer_relu, layer, num_of_tiles_hw, layer_specs_struct);
}

void pw_conv(weights_grp_dt *weights, fms_dt channels[max_fms_size],
			 fms_dt result[max_fms_size],
			 fms_dt tmp_channels[max_tmp_fms_size],
			 int layer, const layer_specs layer_specs_struct,
			 const fused_scales_dt fused_scales[],
			 const fused_scales_log_2_shifts_dt fused_scales_log_2_shifts[],
			 const relu_6_fused_scales_dt relu_6_fused_scales[],
			 const biases_dt fused_zero_points[],
			 const fused_scales_dt fused_scales_part2[],
			 const fused_scales_log_2_shifts_dt fused_scales_log_2_shifts_part2[],
			 const relu_6_fused_scales_dt relu_6_fused_scales_part2[],
			 const biases_dt fused_zero_points_part2[])
{

#pragma HLS INLINE off

	weights_dt weights_tile[pw_conv_parallelism_out][max_conv_d];
#pragma HLS ARRAY_PARTITION variable = weights_tile complete dim = 1
#pragma HLS ARRAY_PARTITION variable = weights_tile cyclic dim = 2 factor = num_of_weights_in_the_same_filter_and_group

	const int layer_conv_d = layer_specs_struct.layer_depth;
	const int layer_weights_offset = layer_specs_struct.layer_weights_offset;
	const int layer_num_fils = layer_specs_struct.layer_num_fils;
	const int num_of_tiles_d_out = layer_specs_struct.layer_num_of_tiles_out_d;

#if HW == FPGA
	weights_grp_dt weight_groups_buffer[num_of_weight_groups_in_the_largest_weight_tile];
	fill_layer_weight_groups_tile_off_chip(weights, weight_groups_buffer, 0,
										   layer_specs_struct.layer_depth, layer_specs_struct.layer_num_of_weight_groups_for_one_pass,
										   layer_specs_struct.layer_weights_offset,
										   layer_specs_struct.layer_num_fils);
#elif HW == CPU
	fill_layers_weights_cpu(weights,
							weights_tile,
							0 * pw_conv_parallelism_out, layer_conv_d,
							layer_weights_offset, layer_num_fils);
#endif

	biases_dt fused_zero_points_buffer[pw_conv_parallelism_out];
	fused_scales_dt fused_scales_buffer[pw_conv_parallelism_out];
	fused_scales_log_2_shifts_dt fused_scales_log_2_shifts_buffer[pw_conv_parallelism_out];
	relu_6_fused_scales_dt relu_6_fused_scales_buffer[pw_conv_parallelism_out];

	const int current_layer_fused_parameters_offset = layers_fused_parameters_offsets[layer];

conv2_ots_loop:
	for (int td_o = 0; td_o < num_of_tiles_d_out; td_o++)
	{
		if (current_layer_fused_parameters_offset < first_quantization_arrays_num_elements)
		{
			fill_fused_zero_points_buffer(fused_zero_points,
										  fused_zero_points_buffer, td_o * pw_conv_parallelism_out,
										  layer, current_layer_fused_parameters_offset);
			fill_fused_scales_buffer(fused_scales, fused_scales_buffer,
									 fused_scales_log_2_shifts, fused_scales_log_2_shifts_buffer,
									 relu_6_fused_scales, relu_6_fused_scales_buffer,
									 td_o * pw_conv_parallelism_out, layer, current_layer_fused_parameters_offset);
		}
		else
		{
			fill_fused_zero_points_buffer(fused_zero_points_part2,
										  fused_zero_points_buffer, td_o * pw_conv_parallelism_out,
										  layer, current_layer_fused_parameters_offset - first_quantization_arrays_num_elements);
			fill_fused_scales_buffer(fused_scales_part2, fused_scales_buffer,
									 fused_scales_log_2_shifts_part2, fused_scales_log_2_shifts_buffer,
									 relu_6_fused_scales_part2, relu_6_fused_scales_buffer,
									 td_o * pw_conv_parallelism_out, layer,
									 current_layer_fused_parameters_offset - first_quantization_arrays_num_elements);
		}

#if HW == FPGA
		fill_weights_tile_from_weight_groups_tile(weight_groups_buffer,
												  weights_tile, td_o * pw_conv_parallelism_out,
												  layer_specs_struct.layer_depth,
												  layer_specs_struct.layer_num_of_weight_groups_for_one_pass,
												  layer_specs_struct.layer_weights_offset);
#endif
		do_conv(weights_tile, channels, result, layer, layer_conv_d,
				layer_num_fils,
				layer_specs_struct.layer_num_of_tiles_in_d, num_of_tiles_d_out,
				layer_specs_struct.layer_num_of_ifm_tiles_h, layer_specs_struct.layer_num_of_ifm_tiles_w,
				tmp_channels,
				layer_specs_struct.layer_num_of_weight_groups_for_one_pass, layer_weights_offset,
				layer_specs_struct.layer_activation, fused_scales_buffer,
				fused_scales_log_2_shifts_buffer, relu_6_fused_scales_buffer,
				fused_zero_points_buffer, td_o, layer_specs_struct);
#if HW == FPGA
		fill_layer_weight_groups_tile_off_chip(weights, weight_groups_buffer,
											   (td_o + 1) * pw_conv_parallelism_out,
											   layer_specs_struct.layer_depth,
											   layer_specs_struct.layer_num_of_weight_groups_for_one_pass,
											   layer_specs_struct.layer_weights_offset,
											   layer_specs_struct.layer_num_fils);
#elif HW == CPU
		fill_layers_weights_cpu(weights,
								weights_tile,
								(td_o + 1) * pw_conv_parallelism_out, layer_conv_d,
								layer_weights_offset, layer_num_fils);
#endif
	}
}

#endif