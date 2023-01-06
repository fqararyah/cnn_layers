#include "bottleneck_kernels.h"

//*************************
pss_dt conv_kernel(fms_dt ifms_buffer[][layer_0_s_filter_dim * layer_0_s_filter_dim],
					  const static layer_0_weights_dt weights_0[layer_0_s_num_fils][layer_0_s_depth][layer_0_s_filter_dim][layer_0_s_filter_dim],
					  const int filter_dim, int conv_d)
{
#pragma HLS INLINE

	pss_dt pss = 0;
	for (int d = 0; d < input_image_depth; d++)
	{
		for (int c_h = 0; c_h < filter_dim; c_h++)
		{
			for (int c_w = 0; c_w < filter_dim; c_w++)
			{
				pss += ifms_buffer[c_h * filter_dim + c_w] * weights[conv_d][d][c_h][c_w];
			}
		}
	}
	return pss;
}

//*************************
void fill_expansion_kernel_input_buffer(fms_dt bottleneck_input[],
										fms_dt expansion_input_buffer[], const int bottleneck_ifms_depth,
										const int bottleneck_ifms_width, int filling_h, int filling_w)
{
#pragma HLS INLINE

	pss_dt pss = 0;
	const int filling_offset = filling_h * bottleneck_ifms_depth * bottleneck_ifms_width + filling_w * bottleneck_ifms_depth;
	for (int i = 0; i < bottleneck_ifms_depth; i++)
	{
#pragma HLS UNROLL
		expansion_input_buffer[i] += bottleneck_input[filling_offset + i];
	}
}

pss_dt expansion_kernel(fms_dt ifms_buffer[], const int ifms_depth,
						const weights_dt weights[][max_of_bottlenecks_expansion_layers_depths], int filter_index, int h, int w,
						const int parallelism_w)
{
#pragma HLS INLINE

	const int starting_index = (h * parallelism_w + w) * ifms_depth;
	pss_dt pss = 0;
	for (int i = 0; i < ifms_depth; i++)
	{
#pragma HLS UNROLL
		pss += weights[filter_index][i] * ifms_buffer[starting_index + i];
	}

	return pss;
}
//*************************

//*************************
void shift_dw_ifms_buffer_horizontally(fms_dt ifms_buffer[], const int strides,
									   const int filter_dim, int conv_d)
{
#pragma HLS INLINE

	for (int h = 0; h < filter_dim; h++)
	{
#pragma HLS UNROLL
		for (int w = 0; w < filter_dim - strides; w++)
		{
#pragma HLS UNROLL
			ifms_buffer[h * filter_dim + w] = ifms_buffer[h * filter_dim + w + strides];
		}
	}
}

void fill_dw_ifms_buffer_upper_part(fms_dt ifms_buffer[], fms_dt *filling_src,
									const int strides, const int filter_dim, int ifms_w_offset,
									const int ifms_width, const int ifms_depth, int filling_d,
									const int padding_left)
{
#pragma HLS INLINE

	const int ifms_dw = ifms_depth * ifms_width;
	const int end_filling_offset_h = filter_dim - strides;
	const int start_filling_offset_w = filter_dim - strides;
	for (int h = 0; h < end_filling_offset_h; h++)
	{
#pragma HLS UNROLL
		for (int w = start_filling_offset_w; w < filter_dim; w++)
		{
#pragma HLS UNROLL
			ifms_buffer[h * filter_dim + w] = filling_src[h * ifms_dw + (ifms_w_offset + w - start_filling_offset_w) * ifms_depth + filling_d];
		}
	}
}

void update_dw_ifms_buffer_upper_part(fms_dt *dw_ifms_buffer_upper_part,
									  fms_dt *filling_src, const int strides, const int filter_dim,
									  int ifms_w_offset, const int ifms_width, const int ifms_depth,
									  int filling_d, const int padding_left,
									  const int first_fill_from_left_offset)
{
#pragma HLS INLINE

	const int rows_to_shift =
		(filter_dim - strides) - strides < 0 ? 0 : (filter_dim - strides) - strides;
	const int ifms_dw = ifms_depth * ifms_width;
	const int end_filling_offset_h = filter_dim - strides;

	for (int h = 0; h < rows_to_shift; h++)
	{
#pragma HLS UNROLL
		for (int w = 0; w < strides; w++)
		{
#pragma HLS UNROLL
			dw_ifms_buffer_upper_part[h * ifms_dw + (ifms_w_offset + w) * ifms_depth + filling_d] =
				dw_ifms_buffer_upper_part[(h + strides) * ifms_dw + (ifms_w_offset + w) * ifms_depth + filling_d];
		}
	}

	for (int h = rows_to_shift; h < end_filling_offset_h; h++)
	{
#pragma HLS UNROLL
		for (int w = 0; w < strides; w++)
		{
#pragma HLS UNROLL
			if (w + first_fill_from_left_offset < strides)
			{
				dw_ifms_buffer_upper_part[h * ifms_dw + (ifms_w_offset + w) * ifms_depth + filling_d] =
					filling_src[h * strides + w + first_fill_from_left_offset];
			}
		}
	}
}

void fill_dw_ifms_buffer_lower_part(fms_dt ifms_buffer[], fms_dt *filling_src,
									const int strides, const int filter_dim, int filling_d)
{
#pragma HLS INLINE

	const int start_filling_offset = filter_dim - strides;
	for (int h = start_filling_offset; h < filter_dim; h++)
	{
#pragma HLS UNROLL
		for (int w = start_filling_offset; w < filter_dim; w++)
		{
#pragma HLS UNROLL
			ifms_buffer[h * filter_dim + w] = filling_src[(h - start_filling_offset) * strides + (w - start_filling_offset)];
		}
	}
}

dw_pss_dt dw_kernel(fms_dt ifms_buffer[],
					const dw_weights_dt weights[][max_dw_filter_area_in_a_chain],
					const int filter_dim, int conv_d)
{
#pragma HLS INLINE

	dw_pss_dt pss = 0;
	for (int c_h = 0; c_h < filter_dim; c_h++)
	{
		for (int c_w = 0; c_w < filter_dim; c_w++)
		{
			pss += ifms_buffer[c_h * filter_dim + c_w] * weights[conv_d][c_h * filter_dim + c_w];
		}
	}
	return pss;
}
//*************************

//*************************
void projection_kernel(fms_dt ifms_val, const int ofms_depth,
					   const weights_dt weights[][max_of_bottlenecks_layers_depths],
					   pss_dt pss_buffer[], int conv_d)
{
#pragma HLS INLINE

	pss_dt pss = 0;
	for (int i = 0; i < ofms_depth; i++)
	{
#pragma HLS UNROLL
		if (conv_d == 0)
		{
			pss_buffer[i] = 0;
		}
		pss_buffer[i] += weights[i][conv_d] * ifms_val;
	}
}
//*************************
void normalize_projection_kernel_output(pss_dt pss_buffer[],
										fms_dt normalized_buffer[],
										const fused_scales_dt projection_layer_fused_scales[],
										const fused_scales_log_2_shifts_dt projection_layer_fused_scales_log_2_shifts[],
										const relu_6_fused_scales_dt projection_layer_relu_6_fused_scales[],
										const biases_dt projection_layer_fused_zero_points[],
										const int ofms_depth, const int layer_relu,
										int bottleneck_projection_layer_index)
{

	const fms_dt projection_layer_ofms_zero_point =
		conv_fms_zero_points[bottleneck_projection_layer_index + 1];
	const rec_scales_dt projection_layer_ofms_scale_rec =
		conv_fms_scales_rec[bottleneck_projection_layer_index + 1];
	const rec_scales_dt projection_layer_ofms_scale =
		conv_fms_scales[bottleneck_projection_layer_index + 1];

#pragma HLS INLINE

	pss_dt pss = 0;
	for (int i = 0; i < ofms_depth; i++)
	{
#pragma HLS UNROLL
		fms_quantization_scheme projection_layer_normalization;
		projection_layer_normalization.ofm_zero_point =
			projection_layer_ofms_zero_point;
		projection_layer_normalization.ofm_scale_rec =
			projection_layer_ofms_scale_rec;
		projection_layer_normalization.ofm_scale = projection_layer_ofms_scale;

		projection_layer_normalization.fused_scales =
			projection_layer_fused_scales[i];
		projection_layer_normalization.fused_scales_log_2_shift =
			projection_layer_fused_scales_log_2_shifts[i];
		projection_layer_normalization.relu_6_fused_scale =
			projection_layer_relu_6_fused_scales[i];
		projection_layer_normalization.fused_zero_point =
			projection_layer_fused_zero_points[i];
		normalized_buffer[i] += pw_relu_norm(pss_buffer[i],
											 projection_layer_normalization, layer_relu);
	}
}
//*************************
