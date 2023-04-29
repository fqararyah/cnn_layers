#include <iostream>
#include "../headers/sesl.h"
#include "../../../../tests/test_utils.h"

using namespace std;

void _7_layer_5_pw(
	fms_dt channels_buffer[layer_5_pw_depth][layer_5_pw_ifm_width],
	const weights_dt weights[layer_7_pw_specs.layer_num_fils][layer_5_pw_depth],
	fms_dt result[layer_6_pw_depth][layer_6_pw_ifm_width])
{

#pragma HLS INLINE off

	const int current_layer = 5;
	const int current_layer_fused_parameters_offsets =
		layers_fused_parameters_offsets[current_layer];

	const fms_dt current_layer_ofms_zero_point = conv_fms_zero_points[current_layer + 1];
	const rec_scales_dt current_layer_ofms_scale_rec =
		conv_fms_scales_rec[current_layer + 1];
	const scales_dt current_layer_ofms_scale = conv_fms_scales[current_layer + 1];

	// rows for next DW
	for (int o_o_d = 0;
		 o_o_d < layer_7_pw_specs.layer_num_fils / layer_5_pw_parallelism_out; o_o_d++)
	{
		int o_o_d_offset = o_o_d * layer_5_pw_parallelism_out;
	// filters loop
	layer_5_pw_pipeline:
		for (int w = 0; w < layer_5_pw_ifm_width; w++)
		{
#pragma HLS PIPELINE
		// FMs width loop
		layer_5_pw_loops:
			for (int o_d = 0;
				 o_d < layer_5_pw_parallelism_out; o_d++)
			{
#pragma HLS UNROLL
				// parallelized filters loop
				pss_dt tmp = 0;
				for (int d = 0; d < layer_5_pw_parallelism_in; d++)
				{
#pragma HLS UNROLL
					// parallelized depth loop
					tmp += channels_buffer[d][w] * weights[o_o_d_offset + o_d][d];
				}
				fms_quantization_scheme normalization = {0, 0, 0, 0};
				normalization.fused_scales = layer_7_pw_fused_scales[o_o_d_offset + o_d];
				normalization.fused_scales_log_2_shift =
					layer_7_pw_fused_scales_log_2_shifts[o_o_d_offset + o_d];
				normalization.relu_6_fused_scale =
					layer_7_pw_relu_6_fused_scales[o_o_d_offset + o_d];
				normalization.fused_zero_point =
					layer_7_pw_fused_zero_points[o_o_d_offset + o_d];
				normalization.ofm_zero_point = current_layer_ofms_zero_point;
				normalization.ofm_scale_rec = current_layer_ofms_scale_rec;
				normalization.ofm_scale = current_layer_ofms_scale;
				result[o_o_d_offset + o_d][w] = dw_relu_norm(tmp, normalization,
															 layer_7_pw_specs.layer_activation);
			}
		}
	}
}

void write_to_tmp_channel(
	fms_dt channels_buffer[layer_7_pw_specs.layer_num_fils][layer_5_pw_ofm_width],
	fms_dt tmp_channels[max_tmp_fms_size], int starting_h)
{
#pragma HLS INLINE off

	const int num_tiles_hw = layer_5_pw_num_of_tiles_h * layer_5_pw_num_of_tiles_w;

	const int tile_in_h = starting_h / pw_tile_h;
	const int in_tile_h = starting_h % pw_tile_h;

// rows for next DW
// cout<<"\nstarting_h: "<<starting_h<<"\n";
write_to_tmp_channel_loops:
	for (int o_o_d = 0;
		 o_o_d < layer_7_pw_specs.layer_num_fils / layer_5_pw_parallelism_out; o_o_d++)
	{
		int o_o_d_offset = o_o_d * layer_5_pw_parallelism_out;
		// filters loop
		for (int o_d = 0; o_d < layer_5_pw_parallelism_out; o_d++)
		{
#pragma HLS UNROLL
			const int tile_in_z = (o_o_d_offset + o_d) / pw_tile_d;
			const int in_tile_z = (o_o_d_offset + o_d) % pw_tile_d;
		layer_5_pw_pipeline:
			for (int w = 0; w < layer_5_pw_ofm_width;
				 w++)
			{
#pragma HLS PIPELINE
				// FMs width loop
				// parallelized filters loop
				pss_dt tmp = 0;

				const int tile_in_w = w / pw_tile_w;
				const int tile_index = tile_in_z * num_tiles_hw + tile_in_h * layer_5_pw_num_of_tiles_w + tile_in_w;

				const int in_tile_w = w % pw_tile_w;
				const int in_tile_index = in_tile_z * pw_tile_hw + in_tile_h * pw_tile_w + in_tile_w;

				const int index_in_tmp_channels = tile_index * pw_tile_size + in_tile_index;

				tmp_channels[index_in_tmp_channels] =
					channels_buffer[o_o_d_offset + o_d][w];
			}
		}
	}
}

void _7_layer_6_pw(
	fms_dt channels_buffer[layer_6_pw_depth][layer_6_pw_ifm_width],
	const weights_dt weights[layer_8_pw_specs.layer_num_fils][layer_6_pw_depth],
	fms_dt result[max_fms_size], int starting_h)
{

#pragma HLS INLINE off

	//	if (starting_h > 50) {
	//		cout << "***7***\n";
	//		for (int w = 0; w < 5; w++) {
	//			//for (int d = 0; d < layer_6_pw_depth; d++)
	//			cout << channels_buffer[0][w] << " ";
	//			//cout << "\n";
	//		}
	//		cout << "******\n";
	//	}

	const fms_dt current_layer_ofms_zero_point = conv_fms_zero_points[6 + 1];

	const rec_scales_dt current_layer_ofms_scale_rec =
		conv_fms_scales_rec[6 + 1];
	const scales_dt current_layer_ofms_scale = conv_fms_scales[6 + 1];

	const int num_tiles_hw = layer_10_pw_specs.layer_num_of_ofm_tiles_h * layer_10_pw_specs.layer_num_of_ofm_tiles_w;

	const int tile_in_h = starting_h / pw_tile_h;
	const int in_tile_h = starting_h % pw_tile_h;

	// rows for next DW
	// cout<<"\nstarting_h: "<<starting_h<<"\n";
	for (int o_o_d = 0;
		 o_o_d < layer_8_pw_specs.layer_num_fils / layer_6_pw_parallelism_out; o_o_d++)
	{
		int o_o_d_offset = o_o_d * layer_6_pw_parallelism_out;
	// filters loop
	layer_6_pw_loops:
		for (int o_d = 0; o_d < layer_6_pw_parallelism_out;
			 o_d++)
		{
#pragma HLS UNROLL
			const int tile_in_z = (o_o_d_offset + o_d) / pw_tile_d;
			const int in_tile_z = (o_o_d_offset + o_d) % pw_tile_d;
		layer_6_pw_pipeline:
			for (int w = 0; w < layer_6_pw_ifm_width;
				 w++)
			{
#pragma HLS PIPELINE
				// FMs width loop
				// parallelized filters loop
				pss_dt tmp = 0;

				const int tile_in_w = w / pw_tile_w;
				const int tile_index = tile_in_z * num_tiles_hw + tile_in_h * layer_10_pw_specs.layer_num_of_ofm_tiles_w + tile_in_w;

				const int in_tile_w = w % pw_tile_w;
				const int in_tile_index = in_tile_z * pw_tile_hw + in_tile_h * pw_tile_w + in_tile_w;

				const int index_in_result = tile_index * pw_tile_size + in_tile_index;

				for (int d = 0; d < layer_6_pw_parallelism_in; d++)
				{
#pragma HLS UNROLL
					// parallelized depth loop
					tmp += channels_buffer[d][w] * weights[o_o_d_offset + o_d][d];
					//					if (o_o_d + o_d == 0 && w == 0) {
					//						cout << channels_buffer[d][w] << " * "
					//								<< weights[o_o_d_offset + o_d][d] << " ";
					//						if (d == layer_6_pw_parallelism_in - 1)
					//							cout << "\n";
					//					}
				}

				fms_quantization_scheme normalization = {0, 0, 0, 0};
				normalization.fused_scales = layer_8_pw_fused_scales[o_o_d_offset + o_d];
				normalization.fused_scales_log_2_shift =
					layer_8_pw_fused_scales_log_2_shifts[o_o_d_offset + o_d];
				normalization.relu_6_fused_scale =
					layer_8_pw_relu_6_fused_scales[o_o_d_offset + o_d];
				normalization.fused_zero_point =
					layer_8_pw_fused_zero_points[o_o_d_offset + o_d];
				normalization.ofm_zero_point = current_layer_ofms_zero_point;
				normalization.ofm_scale_rec = current_layer_ofms_scale_rec;
				normalization.ofm_scale = current_layer_ofms_scale;
				result[index_in_result] = pw_relu_norm_6(tmp, normalization,
													   layer_6_activation);
				//				if (o_o_d_offset + o_d == 0) {
				////					cout << "\n"<<tmp <<" >> "<<pw_relu_norm_6(tmp, normalization, layer_7_relu)
				////							<< "\n";
				//					cout << pw_relu_norm_6(tmp, normalization, layer_7_relu)
				//							<< " ";
				//				}
			}
		}
		//		if (o_o_d == 0) {
		//			cout << "\n";
		//		}
	}
}

void fill_row(
	fms_grp_dt tmp_buffer[input_image_depth][input_image_num_fms_groups_in_width],
	fms_dt channels_buffer_0[input_image_depth][_7_stages_layer_0_s_in_buffer_height][input_image_width],
	const int input_image_num_fms_groups_in_width, int row)
{

#pragma HLS INLINE OFF

	for (int d = 0; d < input_image_depth; d++)
	{
#pragma HLS UNROLL
		for (int o_w = 0; o_w < input_image_num_fms_groups_in_width; o_w++)
		{
#pragma HLS PIPELINE OFF
			const int o_w_offset = o_w * input_image_group_items;
			fms_grp_dt chunck = tmp_buffer[d][o_w];
			for (int w = 0; w < input_image_group_items; w++)
			{
				if (o_w_offset + w < input_image_width)
				{
					channels_buffer_0[d][row][o_w_offset + w] = (fms_dt)chunck(
						w * fms_dt_width + fms_dt_offset, w * fms_dt_width);
				}
			}
		}
	}
}

void _7_stages_fill_channels_buffer(
	fms_grp_dt channels[input_image_depth * input_image_num_fms_groups_in_a_channel],
	fms_dt channels_buffer_0[input_image_depth][first_conv_layer_filter_dim + (_7_stages_layer_0_s_rows_at_once - 1) * first_conv_layer_specs.strides][input_image_width],
	int starting_h)
{

	const fms_dt current_layer_zero_point = conv_fms_zero_points[0];

	const int buffer_height = first_conv_layer_filter_dim + (_7_stages_layer_0_s_rows_at_once - 1) * first_conv_layer_specs.strides;
	const int rows_to_shift = first_conv_layer_filter_dim - first_conv_layer_specs.strides;

	const int filling_starting_offset =
		starting_h == 0 ? input_image_num_fms_groups_in_width * rows_to_shift : starting_h * input_image_num_fms_groups_in_width;

	const int num_fms_groups_to_fitch = input_image_num_fms_groups_in_width * (buffer_height - rows_to_shift);

	const int shift_starting_point =
		starting_h >= buffer_height - rows_to_shift ? buffer_height - rows_to_shift : starting_h - 1;
	const int fill_starting_row = rows_to_shift;
	const int first_time_offset =
		starting_h == 0 ? (first_conv_layer_filter_dim - first_conv_layer_specs.strides) : 0;

	fms_grp_dt tmp_buffer[input_image_depth][input_image_num_fms_groups_in_width];

	// shift
	if (starting_h != 0)
	{
		for (int w = 0; w < input_image_width; w++)
		{
#pragma HLS PIPELINE
			for (int d = 0; d < input_image_depth; d++)
			{
#pragma HLS UNROLL
				for (int h = 0; h < rows_to_shift; h++)
				{
#pragma HLS UNROLL
					channels_buffer_0[d][h][w] = channels_buffer_0[d][h + shift_starting_point][w];
				}
			}
		}
	}
	else
	{
		// fill first time:
		for (int h = 0; h < rows_to_shift; h++)
		{
			const int h_offset = h * input_image_num_fms_groups_in_width;
			for (int d = 0; d < input_image_depth; d++)
			{
				const int d_offst = h_offset + d * input_image_num_fms_groups_in_a_channel;
				for (int i = 0; i < input_image_num_fms_groups_in_width; i++)
				{
					tmp_buffer[d][i] = channels[d_offst + i];
				}
			}
			fill_row(tmp_buffer, channels_buffer_0,
					 input_image_num_fms_groups_in_width, h);
		}
	}

	// fill

	for (int h = rows_to_shift; h < buffer_height; h++)
	{
		const int h_offset = filling_starting_offset + (h - rows_to_shift) * input_image_num_fms_groups_in_width;
		if ((h - rows_to_shift) + starting_h < input_image_height)
		{
			for (int d = 0; d < input_image_depth; d++)
			{
				const int d_offst = h_offset + d * input_image_num_fms_groups_in_a_channel;
				for (int i = 0; i < input_image_num_fms_groups_in_width; i++)
				{
					tmp_buffer[d][i] = channels[d_offst + i];
				}
			}
			fill_row(tmp_buffer, channels_buffer_0,
					 input_image_num_fms_groups_in_width, h);
		}
		else
		{ // padding bottom
			for (int d = 0; d < input_image_depth; d++)
			{
				for (int w = 0; w < input_image_width; w++)
				{
					channels_buffer_0[d][h][w] = current_layer_zero_point;
				}
			}
		}
	}
}

//****************v2
void _7_stages_fill_ifm_groups_buffer(
	fms_grp_dt channels[input_image_depth * input_image_num_fms_groups_in_a_channel],
	fms_grp_dt fms_groups_buffer[input_image_depth][input_image_num_fms_groups_in_width * _7_stages_layer_0_s_in_rows_at_once], int starting_h,
	const int elements_to_fill_from_an_ifm)
{ //_7_stages_layer_0_s_in_rows_at_once * input_image_num_fms_groups_in_width
#pragma HLS INLINE off

	const int start_filling_offset = starting_h * input_image_num_fms_groups_in_width;
	int elements_avaiable_in_input_image;
	if (starting_h + _7_stages_layer_0_s_in_rows_at_once >= input_image_height)
	{
		elements_avaiable_in_input_image = (input_image_height - starting_h) * input_image_num_fms_groups_in_width;
		if (elements_avaiable_in_input_image < 0)
		{
			elements_avaiable_in_input_image = 0;
		}
	}
	for (int d = 0; d < input_image_depth; d++)
	{
		const int d_offst = start_filling_offset + d * input_image_num_fms_groups_in_a_channel;
		for (int i = 0; i < elements_to_fill_from_an_ifm; i++)
		{
			if (i < elements_avaiable_in_input_image)
			{
				fms_groups_buffer[d][i] = channels[d_offst + i];
			}
		}
	}
}

void fill_row_from_groups_buffer(
	fms_grp_dt fms_groups_buffer[input_image_depth][input_image_num_fms_groups_in_width * _7_stages_layer_0_s_in_rows_at_once],
	fms_dt channels_buffer_0[input_image_depth][_7_stages_layer_0_s_in_buffer_height][input_image_width],
	int row, const int channels_buffer_start_filling_h)
{
#pragma HLS INLINE

	const int start_filling_offset = row * input_image_num_fms_groups_in_width;

	for (int o_w = 0; o_w < input_image_num_fms_groups_in_width; o_w++)
	{
#pragma HLS PIPELINE off
		const int o_w_offset = o_w * input_image_group_items;
		for (int d = 0; d < input_image_depth; d++)
		{
#pragma HLS UNROLL
			fms_grp_dt chunck = fms_groups_buffer[d][start_filling_offset + o_w];
			for (int w = 0; w < input_image_group_items; w++)
			{
#pragma HLS PIPELINE
				if (o_w_offset + w < input_image_width)
				{
					channels_buffer_0[d][channels_buffer_start_filling_h + row][o_w_offset + w] = (fms_dt)chunck(
						w * fms_dt_width + fms_dt_offset, w * fms_dt_width);
				}
			}
		}
	}
}

void _7_stages_shift_channels_buffer_rows(
	fms_dt channels_buffer_0[input_image_depth][_7_stages_layer_0_s_in_buffer_height][input_image_width],
	const int rows_to_shift)
{
#pragma HLS INLINE

	for (int w = 0; w < input_image_width; w++)
	{
#pragma HLS PIPELINE
		for (int d = 0; d < input_image_depth; d++)
		{
#pragma HLS UNROLL
			for (int h = 0; h < rows_to_shift; h++)
			{
#pragma HLS UNROLL
				channels_buffer_0[d][h][w] = channels_buffer_0[d][h + _7_stages_layer_0_s_in_rows_at_once][w];
			}
		}
	}
}

void _7_stages_padd_bottom_channels_buffer_rows(
	fms_dt channels_buffer_0[input_image_depth][_7_stages_layer_0_s_in_buffer_height][input_image_width],
	const fms_dt zero_point)
{
#pragma HLS INLINE

	for (int w = 0; w < input_image_width; w++)
	{
#pragma HLS PIPELINE
		for (int d = 0; d < input_image_depth; d++)
		{
#pragma HLS UNROLL
			for (int h = _7_stages_layer_0_s_in_buffer_height - first_conv_layer_specs.padding_bottom;
				 h < _7_stages_layer_0_s_in_buffer_height; h++)
			{
#pragma HLS UNROLL
				channels_buffer_0[d][h][w] = zero_point;
			}
		}
	}
}

void _7_stages_fill_channels_buffer_from_groups_buffer(
	fms_grp_dt fms_groups_buffer[input_image_depth][input_image_num_fms_groups_in_width * _7_stages_layer_0_s_in_rows_at_once],
	fms_dt channels_buffer_0[input_image_depth][_7_stages_layer_0_s_in_buffer_height][input_image_width],
	int starting_h, bool shift, const fms_dt zero_point)
{
#pragma HLS INLINE off

	const int rows_to_shift = _7_stages_layer_0_s_in_buffer_height - _7_stages_layer_0_s_in_rows_at_once;
	if (shift)
	{
		_7_stages_shift_channels_buffer_rows(channels_buffer_0, rows_to_shift);
	}
	const int channels_buffer_start_filling_h =
		starting_h == 0 ? first_conv_layer_specs.padding_top : rows_to_shift;
	for (int h = 0; h < _7_stages_layer_0_s_in_rows_at_once; h++)
	{
		if (starting_h + h < input_image_height)
		{
			fill_row_from_groups_buffer(fms_groups_buffer, channels_buffer_0, h,
										channels_buffer_start_filling_h);
		}
		else
		{
			_7_stages_padd_bottom_channels_buffer_rows(channels_buffer_0,
													   zero_point);
		}
	}
}
//****************v2

void cnn_pipeline_7_mob_v2(
	fms_grp_dt channels[input_image_depth * input_image_num_fms_groups_in_a_channel],
	fms_dt result[max_fms_size], fms_dt tmp_channels[max_tmp_fms_size])
{
#pragma HLS INLINE off

#pragma HLS ARRAY_PARTITION variable = channels type = complete dim = 1

	//#########################even###############################
	fms_dt channels_buffer_0[input_image_depth][_7_stages_layer_0_s_in_buffer_height][input_image_width];
#pragma HLS ARRAY_PARTITION variable = channels_buffer_0 complete dim = 1
#pragma HLS ARRAY_PARTITION variable = channels_buffer_0 complete dim = 2
	fms_dt _6_layer_0_s_3x3_conv_out_0[layer_1_dw_depth][_7_stages_layer_0_s_rows_at_once][layer_1_dw_ifm_width] =
		{0};
	//##############
	fms_dt _6_layer_1_dw_upper[layer_1_dw_depth][layer_2_dw_specs.filter_size - layer_2_dw_specs.strides][layer_1_dw_ifm_width];
#pragma HLS ARRAY_PARTITION variable = _6_layer_1_dw_upper complete dim = 3

	fms_dt _6_layer_1_dw_lower[layer_1_dw_depth][_7_stages_layer_2_rows_at_once][layer_1_dw_ifm_width];
#pragma HLS ARRAY_PARTITION variable = _6_layer_1_dw_lower complete dim = 3

	fms_dt _6_layer_1_dw_out_0[layer_2_pw_depth][_7_stages_layer_2_rows_at_once][layer_2_pw_ifm_width] =
		{0};
#pragma HLS ARRAY_PARTITION variable = _6_layer_1_dw_out_0 complete dim = 1
	//##############

	fms_dt _6_layer_2_pw_out_0[layer_3_pw_depth][_7_stages_layer_3_rows_at_once][layer_3_pw_ifm_width] =
		{0};

	fms_dt _6_layer_4_dw_upper[layer_4_dw_depth][layer_4_dw_ifm_width];
#pragma HLS ARRAY_PARTITION variable = _6_layer_4_dw_upper cyclic factor = layer_2_pw_parallelism_out dim = 1
#pragma HLS ARRAY_PARTITION variable = _6_layer_4_dw_upper cyclic factor = 6 dim = 2

	fms_dt _6_layer_4_dw_lower[layer_4_dw_depth][layer_6_dw_specs.strides][layer_4_dw_ifm_width];
#pragma HLS ARRAY_PARTITION variable = _6_layer_4_dw_lower cyclic factor = 2 dim = 1
#pragma HLS ARRAY_PARTITION variable = _6_layer_4_dw_lower complete dim = 2
#pragma HLS ARRAY_PARTITION variable = _6_layer_4_dw_lower cyclic factor = 12 dim = 3

	fms_dt _6_layer_3_4_pw_dw_out_0[layer_5_pw_depth][layer_5_pw_ifm_width] = {
		0};

#pragma HLS ARRAY_PARTITION variable = _6_layer_0_s_3x3_conv_out_0 complete dim = 1
#pragma HLS ARRAY_PARTITION variable = _6_layer_0_s_3x3_conv_out_0 complete dim = 2

#pragma HLS ARRAY_PARTITION variable = _6_layer_2_pw_out_0 complete dim = 1

#pragma HLS ARRAY_PARTITION variable = _6_layer_3_4_pw_dw_out_0 cyclic factor = layer_6_pw_parallelism_in / 2 dim = 1

	fms_dt _6_layer_5_pw_out_0[layer_6_pw_depth][layer_6_pw_ifm_width] = {0};
#pragma HLS ARRAY_PARTITION variable = _6_layer_5_pw_out_0 complete dim = 1
	//###########################################################

	//#########################odd###############################
	//	fms_dt channels_buffer_0[input_image_depth][first_conv_layer_filter_dim
	//			+ (_7_stages_layer_2_rows_at_once - 1) * first_conv_layer_specs.strides][input_image_width];

	fms_dt _6_layer_0_s_3x3_conv_out_1[layer_1_dw_depth][_7_stages_layer_0_s_rows_at_once][layer_1_dw_ifm_width] =
		{0};
	//##############

	fms_dt _6_layer_1_dw_out_1[layer_2_pw_depth][_7_stages_layer_2_rows_at_once][layer_2_pw_ifm_width] =
		{0};
	//##############
	fms_dt _6_layer_2_pw_out_1[layer_3_pw_depth][_7_stages_layer_3_rows_at_once][layer_3_pw_ifm_width] =
		{0};

	fms_dt _6_layer_3_4_pw_dw_out_1[layer_5_pw_depth][layer_5_pw_ifm_width] = {
		0};

#pragma HLS ARRAY_PARTITION variable = _6_layer_0_s_3x3_conv_out_1 complete dim = 1
#pragma HLS ARRAY_PARTITION variable = _6_layer_0_s_3x3_conv_out_1 complete dim = 2

#pragma HLS ARRAY_PARTITION variable = _6_layer_1_dw_out_1 complete dim = 1

#pragma HLS ARRAY_PARTITION variable = _6_layer_2_pw_out_1 complete dim = 1

#pragma HLS ARRAY_PARTITION variable = _6_layer_3_4_pw_dw_out_1 cyclic factor = layer_6_pw_parallelism_in / 2 dim = 1

	fms_dt _6_layer_5_pw_out_1[layer_6_pw_depth][layer_6_pw_ifm_width] = {0};
#pragma HLS ARRAY_PARTITION variable = _6_layer_5_pw_out_1 complete dim = 1

	//###########################################################
	int pipeline_active_row = 0;
	const int rows_filled_each_time = _7_stages_layer_0_s_in_rows_at_once;
	const int extra_rows_filled_first_time = _7_stages_layer_0_s_in_buffer_height - rows_filled_each_time;
	const int num_of_ifm_groups_read_each_time =
		input_image_num_fms_groups_in_width * _7_stages_layer_0_s_in_rows_at_once;

	const fms_dt first_conv_layer_fms_zero_point = conv_fms_zero_points[0];
	fms_grp_dt fms_groups_buffer[input_image_depth][input_image_num_fms_groups_in_width * _7_stages_layer_0_s_in_rows_at_once];

	_7_stages_fill_ifm_groups_buffer(channels, fms_groups_buffer,
									 pipeline_active_row, input_image_num_fms_groups_in_width);
	_7_stages_fill_channels_buffer_from_groups_buffer(fms_groups_buffer,
													  channels_buffer_0, pipeline_active_row, false,
													  first_conv_layer_fms_zero_point);
	pipeline_active_row++;

	_7_stages_fill_ifm_groups_buffer(channels, fms_groups_buffer,
									 pipeline_active_row, num_of_ifm_groups_read_each_time);
	_7_stages_fill_channels_buffer_from_groups_buffer(fms_groups_buffer,
													  channels_buffer_0, pipeline_active_row, false,
													  first_conv_layer_fms_zero_point);
	_6_layer_0_s_3x3_conv(channels_buffer_0, weights_1,
						  _6_layer_0_s_3x3_conv_out_0);
	//****************************
	_7_stages_fill_ifm_groups_buffer(channels, fms_groups_buffer,
									 pipeline_active_row * rows_filled_each_time + extra_rows_filled_first_time,
									 input_image_num_fms_groups_in_width * _7_stages_layer_0_s_in_rows_at_once);
	_7_stages_fill_channels_buffer_from_groups_buffer(fms_groups_buffer,
													  channels_buffer_0,
													  pipeline_active_row * rows_filled_each_time + extra_rows_filled_first_time, true,
													  first_conv_layer_fms_zero_point);
	_7_stages_fill_ifm_groups_buffer(channels, fms_groups_buffer,
									 (pipeline_active_row + 1) * rows_filled_each_time + extra_rows_filled_first_time,
									 num_of_ifm_groups_read_each_time);
	pipeline_active_row++;
	_6_layer_0_s_3x3_conv(channels_buffer_0, weights_1,
						  _6_layer_0_s_3x3_conv_out_1);

	const int _6_layer_1_dw_upper_height = layer_2_dw_specs.filter_size - layer_2_dw_specs.strides;
	const fms_dt layer_2_ifms_zero_point = conv_fms_zero_points[2];
	// first time only second row of what _6_layer_1_dw is going to be valid, hence fill with zero points
	for (int d = 0; d < layer_1_dw_depth; d++)
	{
		for (int h = 0; h < _6_layer_1_dw_upper_height; h++)
		{
			for (int w = 0; w < layer_1_dw_ifm_width; w++)
			{
				_6_layer_1_dw_upper[d][h][w] = layer_2_ifms_zero_point;
			}
		}
	}
	_6_layer_1_dw(_6_layer_0_s_3x3_conv_out_0, dw_weights_2, _6_layer_1_dw_upper,
				  _6_layer_1_dw_out_0, 1);
	//##########
	//	_7_stages_fill_channels_buffer(channels, channels_buffer_0, 9);
	_7_stages_fill_channels_buffer_from_groups_buffer(fms_groups_buffer,
													  channels_buffer_0,
													  pipeline_active_row * rows_filled_each_time + extra_rows_filled_first_time, true,
													  first_conv_layer_fms_zero_point);
	_7_stages_fill_ifm_groups_buffer(channels, fms_groups_buffer,
									 (pipeline_active_row + 1) * rows_filled_each_time + extra_rows_filled_first_time,
									 num_of_ifm_groups_read_each_time);
	pipeline_active_row++;
	_6_layer_0_s_3x3_conv(channels_buffer_0, weights_1,
						  _6_layer_0_s_3x3_conv_out_0);
	_6_layer_1_dw(_6_layer_0_s_3x3_conv_out_1, dw_weights_2, _6_layer_1_dw_upper,
				  _6_layer_1_dw_out_1, 0);
	_6_layer_2_pw(_6_layer_1_dw_out_0, pw_weights_3, _6_layer_2_pw_out_0);
	//##########
	//	_7_stages_fill_channels_buffer(channels, channels_buffer_0, 13);
	_7_stages_fill_channels_buffer_from_groups_buffer(fms_groups_buffer,
													  channels_buffer_0,
													  pipeline_active_row * rows_filled_each_time + extra_rows_filled_first_time, true,
													  first_conv_layer_fms_zero_point);
	_7_stages_fill_ifm_groups_buffer(channels, fms_groups_buffer,
									 (pipeline_active_row + 1) * rows_filled_each_time + extra_rows_filled_first_time,
									 num_of_ifm_groups_read_each_time);
	pipeline_active_row++;
	_6_layer_0_s_3x3_conv(channels_buffer_0, weights_1,
						  _6_layer_0_s_3x3_conv_out_1);
	_6_layer_1_dw(_6_layer_0_s_3x3_conv_out_0, dw_weights_2, _6_layer_1_dw_upper,
				  _6_layer_1_dw_out_0, 0);
	_6_layer_2_pw(_6_layer_1_dw_out_1, pw_weights_3, _6_layer_2_pw_out_1);
	_6_layer_3_pw_4_dw(_6_layer_2_pw_out_0, pw_weights_3, dw_weights_6,
					   _6_layer_4_dw_upper, _6_layer_4_dw_lower, _6_layer_3_4_pw_dw_out_0,
					   0);
	//##########//_6_layer_3_pw_5_dw first run does not produce any valid output
	//	_7_stages_fill_channels_buffer(channels, channels_buffer_0, 17);
	_7_stages_fill_channels_buffer_from_groups_buffer(fms_groups_buffer,
													  channels_buffer_0,
													  pipeline_active_row * rows_filled_each_time + extra_rows_filled_first_time, true,
													  first_conv_layer_fms_zero_point);
	_7_stages_fill_ifm_groups_buffer(channels, fms_groups_buffer,
									 (pipeline_active_row + 1) * rows_filled_each_time + extra_rows_filled_first_time,
									 num_of_ifm_groups_read_each_time);
	pipeline_active_row++;
	_6_layer_0_s_3x3_conv(channels_buffer_0, weights_1,
						  _6_layer_0_s_3x3_conv_out_0);
	_6_layer_1_dw(_6_layer_0_s_3x3_conv_out_1, dw_weights_2, _6_layer_1_dw_upper,
				  _6_layer_1_dw_out_1, 0);
	_6_layer_2_pw(_6_layer_1_dw_out_0, pw_weights_3, _6_layer_2_pw_out_0);
	_6_layer_3_pw_4_dw(_6_layer_2_pw_out_1, pw_weights_3, dw_weights_6,
					   _6_layer_4_dw_upper, _6_layer_4_dw_lower, _6_layer_3_4_pw_dw_out_1,
					   1);
	// first call of _6_layer_2_pw_5_dw the pw part produces only 1 valid row, that is why the increment is 1 not 2
	//##########
	_7_stages_fill_channels_buffer_from_groups_buffer(fms_groups_buffer,
													  channels_buffer_0,
													  pipeline_active_row * rows_filled_each_time + extra_rows_filled_first_time, true,
													  first_conv_layer_fms_zero_point);
	_7_stages_fill_ifm_groups_buffer(channels, fms_groups_buffer,
									 (pipeline_active_row + 1) * rows_filled_each_time + extra_rows_filled_first_time,
									 num_of_ifm_groups_read_each_time);
	pipeline_active_row++;
	_6_layer_0_s_3x3_conv(channels_buffer_0, weights_1,
						  _6_layer_0_s_3x3_conv_out_1);
	_6_layer_1_dw(_6_layer_0_s_3x3_conv_out_0, dw_weights_2, _6_layer_1_dw_upper,
				  _6_layer_1_dw_out_0, 0);
	_6_layer_2_pw(_6_layer_1_dw_out_1, pw_weights_3, _6_layer_2_pw_out_1);
	 _6_layer_3_pw_4_dw(_6_layer_2_pw_out_0, pw_weights_3, dw_weights_6,
                       _6_layer_4_dw_upper, _6_layer_4_dw_lower, _6_layer_3_4_pw_dw_out_0,
                       3);
    _7_layer_5_pw(_6_layer_3_4_pw_dw_out_1, pw_weights_7, _6_layer_5_pw_out_1);
    write_to_tmp_channel(_6_layer_5_pw_out_1, tmp_channels, 0);

	//	cout << "***6***\n";
	//	for (int w = 0; w < 5; w++) {
	//		for (int d = 0; d < layer_6_pw_depth; d++)
	//			cout << _6_layer_5_pw_out_0[d][w] << " ";
	//		cout << "\n";
	//	}

	const int pipeline_filling_stages = pipeline_active_row;
	int odd_even = pipeline_filling_stages % 2;
	int _6_layer_2_pw_5_dw_starting_h = (pipeline_active_row - 3) * _7_stages_layer_4_rows_at_once - 1;
main_pipeline_loop:
	for (; pipeline_active_row < switch_point_fms_height; pipeline_active_row++)
	{
		if (odd_even)
		{
			//			_7_stages_fill_channels_buffer(channels, channels_buffer_0,
			//					h * rows_filled_each_time
			//							+ extra_rows_filled_first_time);
			_7_stages_fill_channels_buffer_from_groups_buffer(fms_groups_buffer,
															  channels_buffer_0,
															  pipeline_active_row * rows_filled_each_time + extra_rows_filled_first_time, true,
															  first_conv_layer_fms_zero_point);
			_7_stages_fill_ifm_groups_buffer(channels, fms_groups_buffer,
											 (pipeline_active_row + 1) * rows_filled_each_time + extra_rows_filled_first_time,
											 num_of_ifm_groups_read_each_time);
			_6_layer_0_s_3x3_conv(channels_buffer_0, weights_1,
								  _6_layer_0_s_3x3_conv_out_1);
			_6_layer_1_dw(_6_layer_0_s_3x3_conv_out_0, dw_weights_2,
						  _6_layer_1_dw_upper, _6_layer_1_dw_out_0, 0);
			_6_layer_2_pw(_6_layer_1_dw_out_1, pw_weights_3,
						  _6_layer_2_pw_out_1);
			_6_layer_3_pw_4_dw(_6_layer_2_pw_out_0, pw_weights_3, dw_weights_6,
							   _6_layer_4_dw_upper, _6_layer_4_dw_lower,
							   _6_layer_3_4_pw_dw_out_0, _6_layer_2_pw_5_dw_starting_h);
			_7_layer_5_pw(_6_layer_3_4_pw_dw_out_1, pw_weights_7,
						  _6_layer_5_pw_out_1);
			write_to_tmp_channel(_6_layer_5_pw_out_1, tmp_channels,
								 pipeline_active_row - pipeline_filling_stages + 1);
			_7_layer_6_pw(_6_layer_5_pw_out_0, pw_weights_6, result,
						  pipeline_active_row - pipeline_filling_stages);
		}
		else
		{
			//			_7_stages_fill_channels_buffer(channels, channels_buffer_0,
			//					h * rows_filled_each_time
			//							+ extra_rows_filled_first_time);
			_7_stages_fill_channels_buffer_from_groups_buffer(fms_groups_buffer,
															  channels_buffer_0,
															  pipeline_active_row * rows_filled_each_time + extra_rows_filled_first_time, true,
															  first_conv_layer_fms_zero_point);
			_7_stages_fill_ifm_groups_buffer(channels, fms_groups_buffer,
											 (pipeline_active_row + 1) * rows_filled_each_time + extra_rows_filled_first_time,
											 num_of_ifm_groups_read_each_time);
			_6_layer_0_s_3x3_conv(channels_buffer_0, weights_1,
								  _6_layer_0_s_3x3_conv_out_0);
			_6_layer_1_dw(_6_layer_0_s_3x3_conv_out_1, dw_weights_2,
						  _6_layer_1_dw_upper, _6_layer_1_dw_out_1, 0);
			_6_layer_2_pw(_6_layer_1_dw_out_0, pw_weights_3,
						  _6_layer_2_pw_out_0);
			_6_layer_3_pw_4_dw(_6_layer_2_pw_out_1, pw_weights_3, dw_weights_6,
							   _6_layer_4_dw_upper, _6_layer_4_dw_lower,
							   _6_layer_3_4_pw_dw_out_1, _6_layer_2_pw_5_dw_starting_h);
			_7_layer_5_pw(_6_layer_3_4_pw_dw_out_0, pw_weights_7,
						  _6_layer_5_pw_out_0);
			write_to_tmp_channel(_6_layer_5_pw_out_0, tmp_channels,
								 pipeline_active_row - pipeline_filling_stages + 1);
			_7_layer_6_pw(_6_layer_5_pw_out_1, pw_weights_6, result,
						  pipeline_active_row - pipeline_filling_stages);
		}
		odd_even = 1 - odd_even;
		_6_layer_2_pw_5_dw_starting_h += _7_stages_layer_4_rows_at_once;
	}
	//###########################################################
	// cout << "\n*******0*********\n";
	// for (int h = 0; h < _7_stages_layer_0_s_in_buffer_height; h++) {
	// 	for (int w = 0; w < layer_1_dw_ifm_width; w++) {
	// 		cout << channels_buffer_0[0][h][w] << " ";
	// 	}
	// 	cout << "\n";
	// }
	// cout << "\n*********0*******\n";
	// cout << "\n*******1*********\n";
	// for (int h = 0; h < _7_stages_layer_0_s_in_buffer_height; h++) {
	// 	for (int w = 0; w < layer_1_dw_ifm_width; w++) {
	// 		cout << channels_buffer_0[1][h][w] << " ";
	// 	}
	// 	cout << "\n";
	// }
	// cout << "\n*********1*******\n";
	// pipeline flushing##########################################
	_7_layer_6_pw(_6_layer_5_pw_out_1, pw_weights_6, result,
				  switch_point_fms_height - pipeline_filling_stages);
	// ##########
	_7_layer_5_pw(_6_layer_3_4_pw_dw_out_0, pw_weights_7, _6_layer_5_pw_out_0);
	write_to_tmp_channel(_6_layer_5_pw_out_0, tmp_channels,
						 pipeline_active_row - pipeline_filling_stages + 1);
	_7_layer_6_pw(_6_layer_5_pw_out_0, pw_weights_6, result,
				  switch_point_fms_height - pipeline_filling_stages + 1);
	// ##########
	_6_layer_3_pw_4_dw(_6_layer_2_pw_out_1, pw_weights_3, dw_weights_6,
					   _6_layer_4_dw_upper, _6_layer_4_dw_lower, _6_layer_3_4_pw_dw_out_1,
					   _6_layer_2_pw_5_dw_starting_h);
	_7_layer_5_pw(_6_layer_3_4_pw_dw_out_1, pw_weights_7, _6_layer_5_pw_out_1);
	write_to_tmp_channel(_6_layer_5_pw_out_1, tmp_channels,
						 pipeline_active_row - pipeline_filling_stages + 2);
	_7_layer_6_pw(_6_layer_5_pw_out_1, pw_weights_6, result,
				  switch_point_fms_height - pipeline_filling_stages + 2);
	_6_layer_2_pw_5_dw_starting_h += _7_stages_layer_4_rows_at_once;
	// ##########
	_6_layer_2_pw(_6_layer_1_dw_out_0, pw_weights_3, _6_layer_2_pw_out_0);
	_6_layer_3_pw_4_dw(_6_layer_2_pw_out_0, pw_weights_3, dw_weights_6,
					   _6_layer_4_dw_upper, _6_layer_4_dw_lower, _6_layer_3_4_pw_dw_out_0,
					   _6_layer_2_pw_5_dw_starting_h);
	_7_layer_5_pw(_6_layer_3_4_pw_dw_out_0, pw_weights_7, _6_layer_5_pw_out_0);
	write_to_tmp_channel(_6_layer_5_pw_out_0, tmp_channels,
						 pipeline_active_row - pipeline_filling_stages + 3);
	_7_layer_6_pw(_6_layer_5_pw_out_0, pw_weights_6, result,
				  switch_point_fms_height - pipeline_filling_stages + 3);
	_6_layer_2_pw_5_dw_starting_h += _7_stages_layer_4_rows_at_once;
	// ##########
	_6_layer_1_dw(_6_layer_0_s_3x3_conv_out_1, dw_weights_2, _6_layer_1_dw_upper,
				  _6_layer_1_dw_out_1, 0);
	_6_layer_2_pw(_6_layer_1_dw_out_1, pw_weights_3, _6_layer_2_pw_out_1);
	_6_layer_3_pw_4_dw(_6_layer_2_pw_out_1, pw_weights_3, dw_weights_6,
					   _6_layer_4_dw_upper, _6_layer_4_dw_lower, _6_layer_3_4_pw_dw_out_1,
					   _6_layer_2_pw_5_dw_starting_h);
	_7_layer_5_pw(_6_layer_3_4_pw_dw_out_1, pw_weights_7, _6_layer_5_pw_out_1);
	write_to_tmp_channel(_6_layer_5_pw_out_1, tmp_channels,
						 pipeline_active_row - pipeline_filling_stages + 4);
	_7_layer_6_pw(_6_layer_5_pw_out_1, pw_weights_6, result,
				  switch_point_fms_height - pipeline_filling_stages + 4);
	_6_layer_2_pw_5_dw_starting_h += _7_stages_layer_4_rows_at_once;
	// ##########
	// padding bottom
	//  by the end of the previous _6_layer_1_dw, _6_layer_1_dw_upper contains the last two valid rows of the first layer output.
	// This time, _6_layer_1_dw will produce only one valid row
	for (int d = 0; d < layer_1_dw_depth; d++)
	{
		for (int h = 0; h < layer_2_dw_specs.padding_bottom; h++)
		{
			for (int w = 0; w < layer_1_dw_ifm_width; w++)
			{
				_6_layer_0_s_3x3_conv_out_0[d][h][w] = layer_2_ifms_zero_point;
			}
		}
	}
	_6_layer_1_dw(_6_layer_0_s_3x3_conv_out_0, dw_weights_2, _6_layer_1_dw_upper,
				  _6_layer_1_dw_out_0, 0);
	_6_layer_2_pw(_6_layer_1_dw_out_0, pw_weights_3, _6_layer_2_pw_out_0);
	_6_layer_3_pw_4_dw(_6_layer_2_pw_out_0, pw_weights_3, dw_weights_6,
					   _6_layer_4_dw_upper, _6_layer_4_dw_lower, _6_layer_3_4_pw_dw_out_0,
					   _6_layer_2_pw_5_dw_starting_h);
	_7_layer_5_pw(_6_layer_3_4_pw_dw_out_0, pw_weights_7, _6_layer_5_pw_out_0);
	write_to_tmp_channel(_6_layer_5_pw_out_0, tmp_channels,
						 pipeline_active_row - pipeline_filling_stages + 5);
	_7_layer_6_pw(_6_layer_5_pw_out_0, pw_weights_6, result,
				  switch_point_fms_height - pipeline_filling_stages + 5);
}