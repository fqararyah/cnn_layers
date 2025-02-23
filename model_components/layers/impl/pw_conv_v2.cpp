
#include "../headers/layers_imp_common_includes.h"
#include "../headers/pw_conv.h"

#if FIBHA_VERSION == 2

void pw_write_results_tile(
	fms_dt result_tile_scaled[pw_conv_parallelism_out][pw_tile_h][pw_tile_w],
	fms_dt result[][CHANNELS_TILE_HEIGHT][CHANNELS_TILE_WIDTH],
	int tile_indx,
	fms_dt tmp_channels[][CHANNELS_TILE_HEIGHT][CHANNELS_TILE_WIDTH],
	pss_f_dt tmp_channels_scaled_tile[pw_conv_parallelism_out][pw_tile_h][pw_tile_w],
	int starting_d,
	const int num_of_ofm_tiles_h,
	const int in_tile_w,
	layer_specs layer_specs_struct)
{
#pragma HLS INLINE off

	// read_write = 1 when the current layer is the one that is directly connected to the OFMs that have a residual connection to a previous layer
	// read_write = 2 when the current layer has a residual connection
	if (tile_indx >= 0)
	{
		const int num_of_tiles_processed_in_parallel = pw_conv_parallelism_out / pw_tile_d;
		const int num_of_tiles_hw = num_of_ofm_tiles_h * layer_specs_struct.layer_num_of_ofm_tiles_w;
		const int layer_num_filters = layer_specs_struct.layer_num_fils;

	pw_write_results_tile_o_d:
		for (int tile_offset = 0;
			 tile_offset < num_of_tiles_processed_in_parallel;
			 tile_offset++)
		{
#pragma HLS PIPELINE
			const int current_tile_indx = tile_indx + tile_offset * num_of_tiles_hw;
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
							fms_dt scaled_val = result_tile_scaled[tile_offset * pw_tile_d + t_d][t_h][t_w];

							if (layer_specs_struct.write_to_result_or_channels)
							{
								result[current_tile_indx][t_h][t_w] = scaled_val;
							}
							if (layer_specs_struct.write_to_tmp)
							{ // 2: expansion
								tmp_channels[current_tile_indx][t_h][t_w] = scaled_val;
							}
						}
					}
				}
			}
		}
	}
}

void scale_pss_tile(fms_dt tmp_channels[][CHANNELS_TILE_HEIGHT][CHANNELS_TILE_WIDTH],
					pss_dt pss_tile[pw_conv_parallelism_out][pw_tile_h][pw_tile_w],
					fms_dt result_tile_scaled[pw_conv_parallelism_out][pw_tile_h][pw_tile_w],
					layer_specs layer_specs_struct,
					const int num_of_ofm_tiles_h,
					const fused_scales_dt fused_scales_buffer[],
					const relu_6_fused_scales_dt relu_6_fused_scale,
					const biases_dt fused_zero_points_buffer[],
					const int tile_index,
					const int starting_d)
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

		const fms_dt ofm_zero_point = layer_specs_struct.layer_ofms_zero_point;
		const scales_dt ofm_scale = layer_specs_struct.layer_ofms_scale;

		const int num_of_tiles_hw = num_of_ofm_tiles_h * layer_specs_struct.layer_num_of_ofm_tiles_w;
		const int layer_relu = layer_specs_struct.layer_activation;

		const bool fused_with_add = layer_specs_struct.fused_with_add;

	pss_to_fms_tile_o_d:
		for (int tile_offset = 0;
			 tile_offset < num_of_tiles_processed_in_parallel;
			 tile_offset++)
		{
		tile_d:
			for (int t_d = 0; t_d < pw_tile_d; t_d++)
			{
				const int current_tile_indx = tile_index + tile_offset * num_of_tiles_hw;
				const int in_tile_index = tile_offset * pw_tile_d + t_d;
				const biases_dt fused_zero_point =
					fused_zero_points_buffer[starting_d + in_tile_index];
				const scales_dt fused_scale = fused_scales_buffer[starting_d + in_tile_index];
			tile_h:
				for (int t_h = 0; t_h < pw_tile_h; t_h++)
				{
#pragma HLS PIPELINE
				tile_w:
					for (int t_w = 0; t_w < pw_tile_w; t_w++)
					{
#pragma HLS UNROLL
						fms_dt scaled_val;
						if (fused_with_add == 0)
						{
#if MODEL_ACTIVATION == RELU6
							scaled_val =
								pw_relu_norm_6_v2(pss_tile[tile_offset * pw_tile_d + t_d][t_h][t_w], fused_zero_point,
												  ofm_zero_point, fused_scale, relu_6_fused_scale, layer_relu);

							// if(layer_specs_struct.layer_index == 1 && tile_index == 0) {
							// 	pss_dt pss = pss_tile[tile_offset * pw_tile_d
							// 			+ t_d][t_h][t_w];
							// 	pss += fused_zero_point;
							// 	printf("%d >> ", (int) pss);

							// 	pss_f_dt scaled_pss = pss * fused_scale;
							// 	printf("%f >> ", (float) scaled_pss);
							// 	printf("%f >> ", (float) fused_scale);
							// 	if (layer_relu == 6
							// 			&& scaled_pss > relu_6_fused_scale) {
							// 		scaled_pss = relu_6_fused_scale;
							// 	}

							// 	scaled_pss += ofm_zero_point;
							// 	printf("%f >> ", (float) scaled_pss);
							// 	scaled_pss += quant_half - (scaled_pss < 0);
							// 	printf("%f \n ", (float) scaled_pss);
							// }

#elif MODEL_ACTIVATION == RELU
							scaled_val =
								relu_norm(
									pss_tile[tile_offset * pw_tile_d + t_d][t_h][t_w],
									normalization, layer_relu);
#endif
						}
						else
						{
							pss_f_dt tmp_channels_scaled_val =
								skip_connection_other_layer_scale *
								(tmp_channels[current_tile_indx][t_h][t_w] - skip_connection_other_layer_zero_point);
							pss_f_dt scaled_tmp =
								pw_relu_norm_no_q_no_relu_v2(
									pss_tile[tile_offset * pw_tile_d + t_d][t_h][t_w],
									fused_zero_point, fused_scale, ofm_scale);
							// pw_relu_norm_no_q_no_relu(
							// 	pss_tile[tile_offset * pw_tile_d + t_d][t_h][t_w],
							// 	normalization, layer_relu);
#if ADD_LAYER_ACTIVATION == 0
							pss_f_dt addition_result = (scaled_tmp + tmp_channels_scaled_val) * add_layer_scale_reciprocal +
													   add_layer_zero_point;
							// if(layer_specs_struct.layer_index == 10 && tile_offset * pw_tile_d + t_d == 0){
							// 	printf("(%f + %f) * %f + %d \n",
							// 			(float)fused_scale, (float)ofm_scale, (float)fused_scale*ofm_scale, (int)add_layer_zero_point);
							// }
							addition_result = addition_result + quant_half - (addition_result < 0);
							scaled_val = clamp(addition_result);
#elif ADD_LAYER_ACTIVATION == RELU
							pss_f_dt addition_result = (scaled_tmp + tmp_channels_scaled_val);
							if (addition_result > 0)
							{
								addition_result = addition_result * add_layer_scale_reciprocal + add_layer_zero_point + quant_half;
								scaled_val = clamp(addition_result);
							}
							else
							{
								scaled_val = add_layer_zero_point;
							}
#endif
						}

						result_tile_scaled[tile_offset * pw_tile_d + t_d][t_h][t_w] =
							scaled_val;
					}
				}
			}
		}
	}
}

void pw_fill_channels_tile(fms_dt channels[][CHANNELS_TILE_HEIGHT][CHANNELS_TILE_WIDTH],
						   fms_dt channels_tile[pw_tile_h][pw_tile_w], const int starting_index,
						   const int layer_conv_d)
{
#pragma HLS INLINE

	for (int t_h = 0; t_h < pw_tile_h; t_h++)
	{
#pragma HLS UNROLL
		for (int t_w = 0; t_w < pw_tile_w; t_w++)
		{
#pragma HLS UNROLL
			channels_tile[t_h][t_w] = channels[starting_index][t_h][t_w];
		}
	}
}

void pw_conv_eng(fms_dt channels_tile[pw_tile_h][pw_tile_w],
				 weights_dt weights_tile[pw_conv_parallelism_out][max_conv_d],
				 pss_dt results_tile[pw_conv_parallelism_out][pw_tile_h][pw_tile_w],
				 int starting_conv_d, int starting_filter, const int layer_conv_d,
				 const int layer_num_fils, const int strides)
{
#pragma HLS INLINE

pw_conv_eng_loops:
	for (int f_d = 0; f_d < pw_conv_parallelism_out; f_d++)
	{
#pragma HLS UNROLL
		for (int t_h = 0; t_h < PW_PARALLELISM_H; t_h++)
		{
#pragma HLS UNROLL
			for (int t_w = 0; t_w < PW_PARALLELISM_W; t_w++)
			{
#pragma HLS UNROLL
				results_tile[f_d][t_h][t_w] += channels_tile[t_h][t_w] * weights_tile[f_d][starting_conv_d];
			}
		}
	}
}

void pw_conv_pipeline(fms_dt channels[][CHANNELS_TILE_HEIGHT][CHANNELS_TILE_WIDTH],
					  fms_dt result[][CHANNELS_TILE_HEIGHT][CHANNELS_TILE_WIDTH],
					  weights_dt weights_tile[pw_conv_parallelism_out][max_conv_d],
					  pss_dt results_tile[pw_conv_parallelism_out][pw_tile_h][pw_tile_w],
					  layer_specs layer_specs_struct,
					  int td_o,
					  int t_in_h, int t_in_w,
					  const int model_configs_list[])
{
#pragma HLS INLINE OFF

	const int num_of_tiles_d_in = model_configs_list[2 * layer_specs_struct.layer_index] == 0
									  ? layer_specs_struct.layer_num_of_tiles_in_d
									  : (model_configs_list[2 * layer_specs_struct.layer_index] +
										 pw_conv_parallelism_in - 1) /
											pw_conv_parallelism_in;

	// cout << layer_specs_struct.layer_index << " " << layer_specs_struct.layer_num_of_tiles_in_d << " "
	// 	 << (model_configs_list[2 * layer_specs_struct.layer_index] +
	// 		 pw_conv_parallelism_in - 1) /
	// 			pw_conv_parallelism_in<<"\n";

	const int num_of_tiles_w = layer_specs_struct.layer_num_of_ifm_tiles_w;
	const int num_of_tiles_hw = layer_specs_struct.layer_num_of_ifm_tiles_h * num_of_tiles_w;
	const int layer_conv_d = layer_specs_struct.layer_depth;
	const int layer_num_fils = layer_specs_struct.layer_num_fils;
	const int strides = layer_specs_struct.strides;

conv2_itd_loop:
	for (int td_i = 0; td_i < num_of_tiles_d_in; td_i++)
	{
#pragma HLS PIPELINE
		fms_dt channels_buffer[pw_tile_h][pw_tile_w];
#pragma HLS ARRAY_PARTITION variable = channels_buffer complete dim = 0
		pw_fill_channels_tile(channels, channels_buffer,
							  td_i * num_of_tiles_hw + t_in_h * num_of_tiles_w + t_in_w,
							  layer_conv_d);

		pw_conv_eng(channels_buffer, weights_tile, results_tile,
					td_i * pw_tile_d, td_o * pw_conv_parallelism_out, layer_conv_d,
					layer_num_fils, strides);
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
				src_pss_tile[f_d][t_h][t_w] = 0;
			}
		}
	}
}

void do_conv(weights_dt weights_tile[pw_conv_parallelism_out][max_conv_d],
			 fms_dt channels[][CHANNELS_TILE_HEIGHT][CHANNELS_TILE_WIDTH],
			 fms_dt result[][CHANNELS_TILE_HEIGHT][CHANNELS_TILE_WIDTH],
			 fms_dt tmp_channels[][CHANNELS_TILE_HEIGHT][CHANNELS_TILE_WIDTH],
			 const int layer, const layer_specs layer_specs_struct,
			 const fused_scales_dt fused_scales_buffer[],
			 relu_6_fused_scales_dt relu_6_fused_scale,
			 const biases_dt fused_zero_points_buffer[], int td_o,
			 const int model_configs_list[],
			 const int to_produce_row_count,
			 const int starting_row = 0)
{

#pragma HLS INLINE off

	const int num_of_ofm_tiles_h = (to_produce_row_count + pw_tile_h - 1) / pw_tile_h; // layer_specs_struct.layer_num_of_ifm_tiles_h;
	const int num_of_tiles_w = layer_specs_struct.layer_num_of_ifm_tiles_w;
	const int num_of_ofm_tiles_w = layer_specs_struct.layer_num_of_ofm_tiles_w;
	const int num_of_ofm_tiles_hw = num_of_ofm_tiles_h * num_of_ofm_tiles_w;
	const int strides = layer_specs_struct.strides;

	fms_quantization_scheme normalization;

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

	copy_pss_tile(results_tile, prev_results_tile); // just to initialize with zeros

	int prev_tile_index = -1;

conv2_ith_loop:
	for (int t_in_h = 0; t_in_h < num_of_ofm_tiles_h; t_in_h++)
	{
		// ############width loop##############
	conv2_itw_loop:
		for (int t_in_w = 0; t_in_w < num_of_tiles_w;
			 t_in_w++)
		{
			// ############depth loop##############
			int tile_index = td_o * (pw_conv_parallelism_out / pw_tile_d) * num_of_ofm_tiles_hw + (t_in_h / strides) * num_of_ofm_tiles_w +
							 (t_in_w / strides);

			pw_conv_pipeline(channels, result, weights_tile, results_tile,
							 layer_specs_struct,
							 td_o, t_in_h, t_in_w, model_configs_list);
			scale_pss_tile(tmp_channels, prev_results_tile, scaled_result_tile,
						   layer_specs_struct, num_of_ofm_tiles_h, fused_scales_buffer,
						   relu_6_fused_scale, fused_zero_points_buffer,
						   prev_tile_index,
						   td_o * pw_conv_parallelism_out);
			copy_pss_tile(results_tile, prev_results_tile);
			pw_write_results_tile(scaled_result_tile, result,
								  prev_tile_index, tmp_channels, tmp_channels_scaled_tile,
								  td_o * pw_conv_parallelism_out,
								  num_of_ofm_tiles_h, 0,
								  layer_specs_struct);
			prev_tile_index = tile_index;
		}
	}
	scale_pss_tile(tmp_channels, prev_results_tile, scaled_result_tile,
				   layer_specs_struct, num_of_ofm_tiles_h, fused_scales_buffer,
				   relu_6_fused_scale, fused_zero_points_buffer,
				   prev_tile_index,
				   td_o * pw_conv_parallelism_out);

	pw_write_results_tile(scaled_result_tile, result,
						  prev_tile_index, tmp_channels, tmp_channels_scaled_tile,
						  td_o * pw_conv_parallelism_out,
						  num_of_ofm_tiles_h,
						  0,
						  layer_specs_struct);
}

void pw_conv(weights_grp_dt *weights,
			 fms_dt channels[][CHANNELS_TILE_HEIGHT][CHANNELS_TILE_WIDTH],
			 fms_dt result[][CHANNELS_TILE_HEIGHT][CHANNELS_TILE_WIDTH],
			 fms_dt tmp_channels[][CHANNELS_TILE_HEIGHT][CHANNELS_TILE_WIDTH],
			 int layer, const layer_specs layer_specs_struct,
			 const fused_scales_dt fused_scales[],
			 const relu_6_fused_scales_dt relu_6_fused_scales[],
			 const biases_dt fused_zero_points[],
			 const int model_configs_list[2 * max_conv_layers],
			 const int to_produce_row_count,
			 const int starting_row)
{
#pragma HLS INLINE off

	weights_dt weights_tile[pw_conv_parallelism_out][max_conv_d];
#pragma HLS ARRAY_PARTITION variable = weights_tile complete dim = 1
#pragma HLS ARRAY_PARTITION variable = weights_tile cyclic dim = 2 factor = num_of_weights_in_the_same_filter_and_group

	const int model_configs_list_num_filters = model_configs_list[2 * layer + 1];
	const int model_configs_list_layer_depth = model_configs_list[2 * layer];

	const int to_fill_weight_groups_in_a_pass = model_configs_list_layer_depth != 0
													? (model_configs_list_layer_depth * pw_conv_parallelism_out) / weights_group_items
													: layer_specs_struct.layer_num_of_weight_groups_for_one_pass;

#if HW == _FPGA
	weights_grp_dt weight_groups_buffer[num_of_weight_groups_in_the_largest_weight_tile];
	fill_layer_weight_groups_tile_off_chip(weights, weight_groups_buffer, 0,
										   layer_specs_struct.layer_depth, to_fill_weight_groups_in_a_pass,
										   layer_specs_struct.layer_weights_offset,
										   layer_specs_struct.layer_num_fils);
#elif HW == CPU
	fill_layers_weights_cpu(weights,
							weights_tile,
							0 * pw_conv_parallelism_out, layer_specs_struct.layer_depth,
							layer_specs_struct.layer_weights_offset, layer_specs_struct.layer_num_fils);
#endif

	relu_6_fused_scales_dt relu_6_fused_scale = relu_6_fused_scales[layer_specs_struct.layer_index];

	const int current_layer_fused_parameters_offset = layers_fused_parameters_offsets[layer];

	const int model_configs_list_limit =
		(model_configs_list_num_filters + pw_conv_parallelism_out - 1) / pw_conv_parallelism_out;

	// cout << layer << " " << layer_specs_struct.layer_num_of_tiles_out_d << " " << model_configs_list_limit << "\n";

conv2_ots_loop:
	for (int td_o = 0; td_o < layer_specs_struct.layer_num_of_tiles_out_d; td_o++)
	{
		if (model_configs_list_limit != 0 && td_o >= model_configs_list_limit)
		{
			break;
		}

#if HW == _FPGA
		fill_weights_tile_from_weight_groups_tile(weight_groups_buffer,
												  weights_tile,
												  layer_specs_struct.layer_depth,
												  to_fill_weight_groups_in_a_pass);
#endif
		// if(layer_specs_struct.layer_index == 3 && td_o == 0){
		// 	printf("%d\n\n", to_fill_weight_groups_in_a_pass);
		// 			for(int ii =0;ii<8;ii++){
		// 				for(int jj=0;jj<16;jj++){
		// 					printf("%d, ", (int)weights_tile[ii][jj]);
		// 				}
		// 				printf("\n");
		// 			}
		// 		}
		do_conv(weights_tile, channels, result, tmp_channels, layer, layer_specs_struct, fused_scales,
				relu_6_fused_scale, fused_zero_points, td_o, model_configs_list, to_produce_row_count, starting_row);
#if HW == _FPGA
		fill_layer_weight_groups_tile_off_chip(weights, weight_groups_buffer,
											   (td_o + 1) * pw_conv_parallelism_out,
											   layer_specs_struct.layer_depth,
											   to_fill_weight_groups_in_a_pass,
											   layer_specs_struct.layer_weights_offset,
											   layer_specs_struct.layer_num_fils);
#elif HW == CPU
		fill_layers_weights_cpu(weights,
								weights_tile,
								(td_o + 1) * pw_conv_parallelism_out, layer_specs_struct.layer_depth,
								layer_specs_struct.layer_weights_offset, layer_specs_struct.layer_num_fils);
#endif
	}
}

#endif
