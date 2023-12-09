#include "utils.h"

void fill_channels_buffer(
	fms_dt channels[input_image_depth][input_image_height][input_image_width],
	fms_dt channels_buffer_0[input_image_depth][3 + _5_stages_layer_1_rows_at_once - 1][_5_stages_input_image_width],
	int starting_h, int starting_w, int starting_offset)
{

#pragma HLS INLINE off
	for (int w = 0; w < _5_stages_input_image_width; w++)
	{
#pragma HLS PIPELINE
		for (int d = 0; d < layer_1_dw_depth; d++)
		{
#pragma HLS UNROLL
			for (int h = 0; h < max_conv_h + layer_0_strides; h++)
			{
#pragma HLS UNROLL
				channels_buffer_0[d][h][w + starting_offset] = channels[d][starting_h + h][w + starting_w];
			}
		}
	}
	if (starting_w == 0)
	{
		for (int d = 0; d < layer_1_dw_depth; d++)
		{
#pragma HLS UNROLL
			for (int h = 0; h < max_conv_h + layer_0_strides; h++)
			{
#pragma HLS UNROLL
				channels_buffer_0[d][h][0] = 0;
			}
		}
	}
}

void _5_layer_0_3x3_conv_half(
	fms_dt channels_buffer[first_conv_layer_depth][3 + _5_stages_layer_1_rows_at_once - 1][_5_stages_input_image_width],
	layer_0_weights_dt weights[first_conv_layer_num_fils][first_conv_layer_depth][3][3],
	fms_dt result[layer_1_pw_depth][_5_stages_layer_1_pw_input_width])
{
#pragma HLS INLINE off

layer_0_ofms:
	for (int o_o_d = 0;
		 o_o_d < first_conv_layer_num_fils / sesl_layer_0_parallelism_ofms; o_o_d++)
	{
		// depth loop
		int o_o_d_offset = o_o_d * sesl_layer_0_parallelism_ofms;

	layer_0_pipeline:
		for (int w = 0; w < _5_stages_layer_1_pw_input_width; w++)
		{
#pragma HLS PIPELINE
			// FMs width loop
			first_conv_pss_dt tmp = 0;
		layer_0_parallelized_ofms:
			for (int o_d = 0;
				 o_d < sesl_layer_0_parallelism_ofms; o_d++)
			{
				first_conv_pss_dt tmp = 0;
#pragma HLS UNROLL
			// parallelized filters loop
			layer_0_d_loops:
				for (int d = 0; d < first_conv_layer_depth; d++)
				{
#pragma HLS UNROLL
				// parallelized depth loop
				layer_0_ch:
					for (int h = 0; h < 3; h++)
					{
#pragma HLS UNROLL
					// conv height loop
					layer_0_cw:
						for (int c_w = 0; c_w < 3; c_w++)
						{
#pragma HLS UNROLL
							// conv width loop
							tmp += channels_buffer[d][h][w * layer_0_strides + c_w] * weights[o_o_d_offset + o_d][d][h][c_w];
						}
					}
				}
				fms_dt scaled_val = (fms_dt)(((ap_fixed<17, 12>)tmp - zero_point_dw) * ratio_dw_pss_to_fms);
				if (scaled_val > 0)
				{
					result[o_o_d_offset + o_d][w] =
						scaled_val;
				}
			}
		}
	}
}

void _5_layer_1_pw(
	fms_dt channels_buffer[layer_1_pw_depth][_5_stages_layer_1_pw_input_width],
	weights_dt weights[layer_1_pw_num_fils][layer_1_pw_depth],
	fms_dt result[layer_1_dw_depth][_5_stages_layer_1_dw_input_width], int left_offset)
{

#pragma HLS INLINE off
layer_1_pw_main_loop:
	for (int h = 0; h < _5_stages_layer_1_rows_at_once;
		 h++)
	{
		// rows for next DW
		for (int o_o_d = 0;
			 o_o_d < layer_1_pw_num_fils / pw_layer_1_parallelism_out;
			 o_o_d++)
		{
			int o_o_d_offset = o_o_d * pw_layer_1_parallelism_out;
		// filters loop
		o_i_d_loop:
			for (int o_i_d = 0;
				 o_i_d < layer_1_pw_depth / pw_layer_1_parallelism_in;
				 o_i_d++)
			{
				// depth loop
				int o_i_d_offset = o_i_d * pw_layer_1_parallelism_in;
			layer_1_pw_pipeline:
				for (int w = 0; w < _5_stages_layer_1_dw_input_width;
					 w++)
				{
#pragma HLS PIPELINE
				// FMs width loop
				layer_1_pw_loops:
					for (int o_d = 0;
						 o_d < pw_layer_1_parallelism_out; o_d++)
					{
#pragma HLS UNROLL
						// parallelized filters loop
						pss_dt tmp = 0;
						for (int d = 0; d < pw_layer_1_parallelism_in; d++)
						{
#pragma HLS UNROLL
							// parallelized depth loop
							tmp +=
								((fms_dt)channels_buffer[o_i_d_offset + d][h][w]) * weights[o_d][o_i_d_offset + d];
						}
						fms_dt scaled_val = (fms_dt)((((ap_fixed<17, 12>)tmp) - zero_point_dw) * ratio_dw_pss_to_fms);
						if (scaled_val > 0)
						{
							result[o_o_d_offset + o_d][w + left_offset] += scaled_val;
						}
					}
				}
			}
		}
	}
}

void write_row_1(fms_dt src[layer_1_dw_depth][_5_stages_layer_1_dw_input_width],
				 fms_dt upper[layer_1_dw_depth][2][_5_stages_layer_1_dw_input_width],
				 fms_dt lower[layer_1_dw_depth][_5_stages_layer_1_dw_input_width], int left_offset)
{

layer_1_pw_pipeline:
	for (int w = 0; w < _5_stages_layer_1_dw_input_width; w++)
	{
#pragma HLS PIPELINE
		for (int d = 0; d < layer_1_dw_depth; d++)
		{
#pragma HLS UNROLL
			upper[d][0][w] = upper[d][1][w];
			upper[d][1][w] = lower[d][0][w];
			lower[d][0][w] = src[d][0][w];
		}
	}
	if (left_offset == 0)
	{
		for (int d = 0; d < layer_1_dw_depth; d++)
		{
#pragma HLS UNROLL
			upper[d][0][_5_stages_layer_1_dw_input_width - 1] = 0;
			upper[d][1][_5_stages_layer_1_dw_input_width - 1] = 0;
			lower[d][0][_5_stages_layer_1_dw_input_width - 1] = 0;
		}
	}
	else
	{
		for (int d = 0; d < layer_1_dw_depth; d++)
		{
#pragma HLS UNROLL
			upper[d][0][0] = 0;
			upper[d][1][0] = 0;
			lower[d][0][0] = 0;
		}
	}
}

void _5_layer_1_dw(fms_dt upper[layer_1_dw_depth][2][_5_stages_layer_1_dw_input_width],
				   fms_dt lower[layer_1_dw_depth][_5_stages_layer_1_dw_input_width],
				   dw_weights_dt dw_weights[layer_1_dw_depth][3][3],
				   fms_dt result[layer_2_pw_depth][_5_stages_layer_2_pw_input_width])
{

#pragma HLS INLINE off
	for (int o_d = 0; o_d < layer_1_dw_depth / dw_layer_1_parallelism;
		 o_d++)
	{
		// depth loop
		int o_d_indx = o_d * dw_layer_1_parallelism;

	layer_1_dw_pipeline:
		for (int w = 0; w < _5_stages_layer_2_pw_input_width; w +=
																 layer_2_dw_specs.strides)
		{
#pragma HLS PIPELINE
		// FMs width loop
		layer_1_dw_loops:
			for (int d = 0; d < dw_layer_1_parallelism; d++)
			{
#pragma HLS UNROLL
				dw_pss_dt tmp = 0;
				// parallelized depth loop
				for (int c_w = 0; c_w < 3; c_w++)
				{
					// conv width loop
#pragma HLS UNROLL
					tmp += upper[o_d_indx + d][0][w + c_w] * dw_weights[o_d_indx + d][0][c_w];
				}
				for (int c_w = 0; c_w < 3; c_w++)
				{
					// conv width loop
#pragma HLS UNROLL
					tmp += upper[o_d_indx + d][1][w + c_w] * dw_weights[o_d_indx + d][1][c_w];
				}

				for (int c_w = 0; c_w < 3; c_w++)
				{
					// conv width loop
#pragma HLS UNROLL
					tmp += lower[o_d_indx + d][0][w + c_w] * dw_weights[o_d_indx + d][2][c_w];
				}
				fms_dt scaled_val = (fms_dt)((((ap_fixed<17, 12>)tmp) - zero_point_dw) * ratio_dw_pss_to_fms);
				if (scaled_val > 0)
				{
					result[o_d_indx + d][w / layer_2_dw_specs.strides] =
						scaled_val;
				}
			}
		}
	}
}

void _5_layer_2_pw(
	fms_dt channels_buffer[layer_2_pw_depth][_5_stages_layer_2_pw_input_width],
	weights_dt weights[layer_3_pw_num_fils][layer_3_pw_depth],
	fms_dt result[layer_3_pw_num_fils][_5_stages_layer_3_pw_input_width])
{

#pragma HLS INLINE off

	// rows for next DW
	for (int o_o_d = 0;
		 o_o_d < layer_3_pw_specs.num_fils / pw_layer_2_parallelism_out;
		 o_o_d++)
	{
		int o_o_d_offset = o_o_d * pw_layer_2_parallelism_out;
	// filters loop
	o_i_d_loop:
		for (int o_i_d = 0;
			 o_i_d < layer_2_pw_depth / pw_layer_2_parallelism_in;
			 o_i_d++)
		{
			// depth loop
			int o_i_d_offset = o_i_d * pw_layer_2_parallelism_in;
		layer_2_pw_pipeline:
			for (int w = 0; w < _5_stages_layer_3_pw_input_width; w++)
			{
#pragma HLS PIPELINE
			// FMs width loop
			layer_2_pw_loops:
				for (int o_d = 0;
					 o_d < pw_layer_2_parallelism_out; o_d++)
				{
#pragma HLS UNROLL
					// parallelized filters loop
					pss_dt tmp = 0;
					for (int d = 0; d < pw_layer_2_parallelism_in; d++)
					{
#pragma HLS UNROLL
						// parallelized depth loop
						tmp += ((fms_dt)channels_buffer[o_i_d_offset + d][w]) * weights[o_d][o_i_d_offset + d];
					}
					fms_dt scaled_val = (fms_dt)((((ap_fixed<17, 12>)tmp) - zero_point_dw) * ratio_dw_pss_to_fms);
					if (scaled_val > 0)
					{
						result[o_o_d_offset + o_d][w] += scaled_val;
					}
				}
			}
		}
	}
}

void _5_layer_3_pw(
	fms_dt channels_buffer[layer_3_pw_depth][_5_stages_layer_3_pw_input_width],
	weights_dt weights[layer_3_pw_num_fils][layer_3_pw_depth],
	fms_dt result[layer_3_pw_num_fils][_5_stages_layer_3_pw_input_width])
{

#pragma HLS INLINE off

	// rows for next DW
	for (int o_o_d = 0;
		 o_o_d < layer_3_pw_num_fils / pw_layer_2_parallelism_out;
		 o_o_d++)
	{
		int o_o_d_offset = o_o_d * pw_layer_3_parallelism_out;
	// filters loop
	o_i_d_loop:
		for (int o_i_d = 0;
			 o_i_d < layer_3_pw_depth / pw_layer_3_parallelism_in;
			 o_i_d++)
		{
			// depth loop
			int o_i_d_offset = o_i_d * pw_layer_3_parallelism_in;
		layer_3_pw_pipeline:
			for (int w = 0; w < _5_stages_layer_3_pw_input_width; w++)
			{
#pragma HLS PIPELINE
			// FMs width loop
			layer_3_pw_loops:
				for (int o_d = 0;
					 o_d < pw_layer_3_parallelism_out; o_d++)
				{
#pragma HLS UNROLL
					// parallelized filters loop
					pss_dt tmp = 0;
					for (int d = 0; d < pw_layer_3_parallelism_in; d++)
					{
#pragma HLS UNROLL
						// parallelized depth loop
						tmp += ((fms_dt)channels_buffer[o_i_d_offset + d][w]) * weights[o_d][o_i_d_offset + d];
					}
					fms_dt scaled_val = (fms_dt)((((ap_fixed<17, 12>)tmp) - zero_point_dw) * ratio_dw_pss_to_fms);
					if (scaled_val > 0)
					{
						result[o_o_d_offset + o_d][w] += scaled_val;
					}
				}
			}
		}
	}
}

void _5_conv_pipeline(
	fms_dt channels[input_image_depth][input_image_height][input_image_width],
	fms_dt result[max_fms_size])
{

#pragma HLS ARRAY_PARTITION variable = channels type = complete dim = 1

	dw_weights_dt dw_weights_2[layer_1_dw_depth][3][3];

#pragma HLS ARRAY_PARTITION variable = dw_weights_2 type = complete dim = 1

	layer_0_weights_dt weights_1[first_conv_layer_num_fils][first_conv_layer_depth][3][3];

	weights_dt pw_weights_1[layer_1_pw_num_fils][layer_1_pw_depth];
	weights_dt pw_weights_3[layer_3_pw_specs.num_fils][layer_2_pw_depth];
	weights_dt pw_weights_3[layer_3_pw_num_fils][layer_3_pw_depth];
	_5_fill_layers_weights(weights_1, dw_weights_2, pw_weights_1, pw_weights_3,
						   pw_weights_3);

	//#########################even###############################
	fms_dt channels_buffer_0[input_image_depth][3 + _5_stages_layer_1_rows_at_once - 1][_5_stages_input_image_width];

	fms_dt _5_layer_0_3x3_conv_out_0[layer_1_pw_depth][_5_stages_layer_1_pw_input_width] =
		{0};

	fms_dt _5_layer_1_pw_out_0[layer_1_dw_depth][_5_stages_layer_1_dw_input_width] = {0};

	fms_dt _5_layer_1_dw_upper_0[layer_1_dw_depth][2][_5_stages_layer_1_dw_input_width];
	fms_dt _5_layer_1_dw_lower_0[layer_1_dw_depth][_5_stages_layer_1_dw_input_width];
	fms_dt _5_layer_1_dw_out_0[layer_2_pw_depth][_5_stages_layer_2_pw_input_width] = {0};

	fms_dt _5_layer_2_pw_out_0[layer_3_pw_depth][_5_stages_layer_3_pw_input_width] = {0};

	fms_dt _5_layer_3_pw_out_0[layer_3_pw_depth][_5_stages_layer_3_pw_input_width] = {0};

#pragma HLS ARRAY_PARTITION variable = channels_buffer_0 complete dim = 1
#pragma HLS ARRAY_PARTITION variable = channels_buffer_0 complete dim = 2
#pragma HLS ARRAY_PARTITION variable = channels_buffer_0 cyclic factor = 2 dim = 3

#pragma HLS ARRAY_PARTITION variable = _5_layer_0_3x3_conv_out_0 complete dim = 1

#pragma HLS ARRAY_PARTITION variable = _5_layer_1_dw_upper_0 cyclic factor = 3 dim = 3

#pragma HLS ARRAY_PARTITION variable = _5_layer_1_dw_out_0 cyclic factor = pw_layer_2_parallelism_in / 2 dim = 1

#pragma HLS ARRAY_PARTITION variable = _5_layer_2_pw_out_0 cyclic factor = pw_layer_3_parallelism_in / 2 dim = 1
	//###########################################################

	//#########################odd###############################
	fms_dt channels_buffer_1[input_image_depth][3 + _5_stages_layer_1_rows_at_once - 1][_5_stages_input_image_width];

	fms_dt _5_layer_0_3x3_conv_out_1[layer_1_pw_depth][_5_stages_layer_1_pw_input_width] =
		{0};

	fms_dt _5_layer_1_pw_out_1[layer_1_dw_depth][_5_stages_layer_1_dw_input_width] = {0};

	fms_dt _5_layer_1_dw_upper_1[layer_1_dw_depth][2][_5_stages_layer_1_dw_input_width];
	fms_dt _5_layer_1_dw_lower_1[layer_1_dw_depth][_5_stages_layer_1_dw_input_width];
	fms_dt _5_layer_1_dw_out_1[layer_2_pw_depth][_5_stages_layer_2_pw_input_width] = {0};

	fms_dt _5_layer_2_pw_out_1[layer_3_pw_depth][_5_stages_layer_3_pw_input_width] = {0};

	fms_dt _5_layer_3_pw_out_1[layer_3_pw_depth][_5_stages_layer_3_pw_input_width] = {0};

#pragma HLS ARRAY_PARTITION variable = channels_buffer_1 complete dim = 1
#pragma HLS ARRAY_PARTITION variable = channels_buffer_1 complete dim = 2
#pragma HLS ARRAY_PARTITION variable = channels_buffer_1 cyclic factor = 3 dim = 3

#pragma HLS ARRAY_PARTITION variable = _5_layer_0_3x3_conv_out_1 complete dim = 1

#pragma HLS ARRAY_PARTITION variable = _5_layer_1_dw_upper_1 cyclic factor = 2 dim = 3

#pragma HLS ARRAY_PARTITION variable = _5_layer_1_dw_out_1 cyclic factor = pw_layer_2_parallelism_in / 2 dim = 1

#pragma HLS ARRAY_PARTITION variable = _5_layer_2_pw_out_1 cyclic factor = pw_layer_3_parallelism_in / 2 dim = 1
	//###########################################################

	int odd_even = 0;
	int h = 0;

width_loop_division:
	for (int part_in_width = 0; part_in_width < 2; part_in_width++)
	{
	pre_pipeline_loop:
		for (; h < pipeline_depth; h++)
		{
			if (odd_even)
			{
				fill_channels_buffer(channels, channels_buffer_0, h, part_in_width * (_5_stages_input_image_width - (first_conv_layer_filter_dim - layer_0_strides)), 1- part_in_width);
				if (h >= 1)
				{
					_5_layer_0_3x3_conv_half(channels_buffer_1, weights_1,
										_5_layer_0_3x3_conv_out_1);
				}
				if (h >= 2)
				{
					_5_layer_1_pw(_5_layer_0_3x3_conv_out_0, pw_weights_1,
								  _5_layer_1_pw_out_0, part_in_width);
				}
				if (h >= 3)
				{
					_5_layer_1_dw(_5_layer_1_dw_upper_1, _5_layer_1_dw_lower_1,
								  dw_weights_2, _5_layer_1_dw_out_1);
				}
				if (h >= 4)
				{
					_5_layer_2_pw(_5_layer_1_dw_out_0, pw_weights_3,
								  _5_layer_2_pw_out_0);
				}
				if (h >= 5)
				{
					_5_layer_2_pw(_5_layer_1_dw_out_1, pw_weights_3,
								  _5_layer_2_pw_out_1);
				}
			}
			else
			{
				fill_channels_buffer(channels, channels_buffer_1, h, part_in_width * (_5_stages_input_image_width - (first_conv_layer_filter_dim - layer_0_strides)), 1- part_in_width );
				if (h >= 1)
				{
					_5_layer_0_3x3_conv_half(channels_buffer_0, weights_1,
										_5_layer_0_3x3_conv_out_0);
				}
				if (h >= 2)
				{
					_5_layer_1_pw(_5_layer_0_3x3_conv_out_1, pw_weights_1,
								  _5_layer_1_pw_out_1, part_in_width);
				}
				if (h >= 3)
				{
					_5_layer_1_dw(_5_layer_1_dw_upper_0, _5_layer_1_dw_lower_0,
								  dw_weights_2, _5_layer_1_dw_out_0);
				}
				if (h >= 4)
				{
					_5_layer_2_pw(_5_layer_1_dw_out_1, pw_weights_3,
								  _5_layer_2_pw_out_1);
				}
				if (h >= 5)
				{
					_5_layer_2_pw(_5_layer_1_dw_out_0, pw_weights_3,
								  _5_layer_2_pw_out_0);
				}
			}
			odd_even = 1 - odd_even;
		}
		if (h == pipeline_depth)
		{
		main_pipeline_loop:
			for (; h < switch_point_fms_height; h++)
			{
				if (odd_even)
				{
					fill_channels_buffer(channels, channels_buffer_0, h, part_in_width * (_5_stages_input_image_width - (first_conv_layer_filter_dim - layer_0_strides)), 1- part_in_width );
					_5_layer_0_3x3_conv_half(channels_buffer_1, weights_1,
										_5_layer_0_3x3_conv_out_1);
					_5_layer_1_pw(_5_layer_0_3x3_conv_out_0, pw_weights_1,
								  _5_layer_1_pw_out_0, part_in_width);
					_5_layer_1_dw(_5_layer_1_dw_upper_1, _5_layer_1_dw_lower_1,
								  dw_weights_2, _5_layer_1_dw_out_1);
					_5_layer_2_pw(_5_layer_1_dw_out_0, pw_weights_3,
								  _5_layer_2_pw_out_0);
					_5_layer_2_pw(_5_layer_1_dw_out_1, pw_weights_3,
								  _5_layer_2_pw_out_1);
					for (int i = 0; i < 7; i++)
					{
						result[i] = _5_layer_2_pw_out_0[h - pipeline_depth][i];
					}
				}
				else
				{
					fill_channels_buffer(channels, channels_buffer_1, h, part_in_width * (_5_stages_input_image_width - (first_conv_layer_filter_dim - layer_0_strides)), 1- part_in_width );
					_5_layer_0_3x3_conv_half(channels_buffer_0, weights_1,
										_5_layer_0_3x3_conv_out_0);
					_5_layer_1_pw(_5_layer_0_3x3_conv_out_1, pw_weights_1,
								  _5_layer_1_pw_out_1, part_in_width);
					_5_layer_1_dw(_5_layer_1_dw_upper_0, _5_layer_1_dw_lower_0,
								  dw_weights_2, _5_layer_1_dw_out_0);
					_5_layer_2_pw(_5_layer_1_dw_out_1, pw_weights_3,
								  _5_layer_2_pw_out_1);
					_5_layer_2_pw(_5_layer_1_dw_out_0, pw_weights_3,
								  _5_layer_2_pw_out_0);
					for (int i = 0; i < 7; i++)
					{
						result[i] = _5_layer_2_pw_out_1[h - pipeline_depth][i];
					}
				}
				odd_even = 1 - odd_even;
			}
		}
		if (h == switch_point_fms_height)
		{
		post_pipeline_loop:
			for (; h < switch_point_fms_height + pipeline_depth;
				 h++)
			{
				if (odd_even)
				{
					fill_channels_buffer(channels, channels_buffer_0, h, part_in_width * (_5_stages_input_image_width - (first_conv_layer_filter_dim - layer_0_strides)), 1- part_in_width );
					if (h >= 1)
					{
						_5_layer_0_3x3_conv_half(channels_buffer_1, weights_1,
											_5_layer_0_3x3_conv_out_1);
					}
					if (h >= 2)
					{
						_5_layer_1_pw(_5_layer_0_3x3_conv_out_0, pw_weights_1,
									  _5_layer_1_pw_out_0, part_in_width);
					}
					if (h >= 3)
					{
						_5_layer_1_dw(_5_layer_1_dw_upper_1, _5_layer_1_dw_lower_1,
									  dw_weights_2, _5_layer_1_dw_out_1);
					}
					if (h >= 4)
					{
						_5_layer_2_pw(_5_layer_1_dw_out_0, pw_weights_3,
									  _5_layer_2_pw_out_0);
					}
					if (h >= 5)
					{
						_5_layer_2_pw(_5_layer_1_dw_out_1, pw_weights_3,
									  _5_layer_2_pw_out_1);
					}
					if (h >= 6)
					{
						for (int i = 0; i < 7; i++)
						{
							result[i] = _5_layer_2_pw_out_0[h - pipeline_depth][i];
						}
					}
				}
				else
				{
					fill_channels_buffer(channels, channels_buffer_1, h, part_in_width * (_5_stages_input_image_width - (first_conv_layer_filter_dim - layer_0_strides)), 1- part_in_width );
					if (h >= 1)
					{
						_5_layer_0_3x3_conv_half(channels_buffer_0, weights_1,
											_5_layer_0_3x3_conv_out_0);
					}
					if (h >= 2)
					{
						_5_layer_1_pw(_5_layer_0_3x3_conv_out_1, pw_weights_1,
									  _5_layer_1_pw_out_1, part_in_width);
					}
					if (h >= 3)
					{
						_5_layer_1_dw(_5_layer_1_dw_upper_0, _5_layer_1_dw_lower_0,
									  dw_weights_2, _5_layer_1_dw_out_0);
					}
					if (h >= 4)
					{
						_5_layer_2_pw(_5_layer_1_dw_out_1, pw_weights_3,
									  _5_layer_2_pw_out_1);
					}
					if (h >= 5)
					{
						_5_layer_2_pw(_5_layer_1_dw_out_0, pw_weights_3,
									  _5_layer_2_pw_out_0);
					}
					if (h >= 6)
					{
						for (int i = 0; i < 7; i++)
						{
							result[i] = _5_layer_2_pw_out_1[h - pipeline_depth][i];
						}
					}
				}
				odd_even = 1 - odd_even;
			}
		}
	}
}
