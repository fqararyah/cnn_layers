#include "bottleneck_kernels.h"

//*************************
pss_dt conv_kernel(fms_dt ifms_buffer[],
				   const layer_0_weights_dt weights_0[layer_0_s_num_fils][layer_0_s_depth][layer_0_s_filter_dim][layer_0_s_filter_dim],
				   const int filter_dim, int conv_d)
{
#pragma HLS INLINE
	const int ifms_buffer_hw = bottlenck_0_input_buffer_height * bottlenck_0_input_buffer_width;
	pss_dt pss = 0;
conv_kernel:
	for (int d = 0; d < input_image_depth; d++)
	{
#pragma HLS UNROLL
		for (int c_h = 0; c_h < filter_dim; c_h++)
		{
#pragma HLS UNROLL
			for (int c_w = 0; c_w < filter_dim; c_w++)
			{
#pragma HLS UNROLL
				pss += ifms_buffer[d * ifms_buffer_hw + c_h * bottlenck_0_input_buffer_width + c_w] * weights_0[conv_d][d][c_h][c_w];
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
fill_expansion_kernel_input_buffer:
	for (int i = 0; i < bottleneck_ifms_depth; i++)
	{
#pragma HLS UNROLL
		expansion_input_buffer[i] += bottleneck_input[filling_offset + i];
	}
}

pss_dt expansion_kernel(fms_dt ifms_buffer[],
						const weights_dt weights[],
						const int ifms_depth, int filter_index, int h, int w,
						const int parallelism_w)
{
#pragma HLS INLINE

	const int starting_index = (h * parallelism_w + w) * ifms_depth;
	pss_dt pss = 0;
expansion_kernel:
	for (int i = 0; i < ifms_depth; i++)
	{
#pragma HLS UNROLL
		pss += weights[i] * ifms_buffer[starting_index + i];
	}

	return pss;
}
//*************************

//*************************
void shift_dw_ifms_buffer_horizontally_3x3_s1(fms_dt ifms_buffer[][3], const int buffer_depth)
{
#pragma HLS INLINE

	const int strides = 1;
shift_dw_ifms_buffer_horizontally_3x3_s1:
	for (int d = 0; d < buffer_depth; d++)
	{
#pragma HLS UNROLL
		for (int w = 0; w < 3 - strides; w++)
		{
#pragma HLS UNROLL
			ifms_buffer[d][w] = ifms_buffer[d][w + strides];
		}
	}
}

void shift_dw_ifms_buffer_horizontally_3x3_s2(fms_dt ifms_buffer[][6], const int buffer_depth)
{
#pragma HLS INLINE

	const int strides = 2;
shift_dw_ifms_buffer_horizontally_3x3_s1:
	for (int d = 0; d < buffer_depth; d++)
	{
#pragma HLS UNROLL
		ifms_buffer[d][0] = ifms_buffer[d][strides];
		ifms_buffer[d][3] = ifms_buffer[d][3 + strides];
	}
}

// void fill_dw_ifms_buffer_upper_part(fms_dt ifms_buffer[], fms_dt *filling_src,
// 									const int strides, const int filter_dim, int ifms_w_offset,
// 									const int ifms_width, const int ifms_depth, int filling_d,
// 									const int padding_left)
// {
// 	#pragma HLS INLINE

// 	const int ifms_dw = ifms_depth * ifms_width;
// 	const int end_filling_offset_h = filter_dim - strides;
// 	const int start_filling_offset_w = filter_dim - strides;
// 	for (int h = 0; h < end_filling_offset_h; h++)
// 	{
// 		#pragma HLS UNROLL
// 		for (int w = start_filling_offset_w; w < filter_dim; w++)
// 		{
// 			#pragma HLS UNROLL
// 			ifms_buffer[h * filter_dim + w] = filling_src[h * ifms_dw + (ifms_w_offset + w - start_filling_offset_w) * ifms_depth + filling_d];
// 		}
// 	}
// }

// void update_dw_ifms_buffer_upper_part(fms_dt *dw_ifms_buffer_upper_part,
// 									  fms_dt *filling_src, const int strides, const int filter_dim,
// 									  int ifms_w_offset, const int ifms_width, const int ifms_depth,
// 									  int filling_d, const int padding_left,
// 									  const int first_fill_from_left_offset, bool do_shift)
// {
// 	#pragma HLS INLINE

// 	const int upper_part_heights = filter_dim - strides;
// 	const int rows_to_shift = upper_part_heights - strides;
// 	const int ifms_hw = upper_part_heights * ifms_width;

// 	if (do_shift)
// 	{
// 		for (int h = 0; h < rows_to_shift; h++)
// 		{
// 			#pragma HLS UNROLL
// 			for (int w = 0; w < strides; w++)
// 			{
// 				#pragma HLS UNROLL
// 				dw_ifms_buffer_upper_part[h * ifms_dw + ifms_w_offset + w] =
// 					dw_ifms_buffer_upper_part[filling_d * ifms_hw + (h + strides) * ifms_width + (ifms_w_offset + w) * ifms_depth + filling_d];
// 			}
// 		}
// 	}

// 	for (int h = rows_to_shift; h < upper_part_heights; h++)
// 	{
// 		#pragma HLS UNROLL
// 		for (int w = 0; w < strides; w++)
// 		{
// 			#pragma HLS UNROLL
// 			if (w + first_fill_from_left_offset < strides)
// 			{
// 				dw_ifms_buffer_upper_part[h * ifms_dw + (ifms_w_offset + w) * ifms_depth + filling_d] =
// 					filling_src[h * strides + w + first_fill_from_left_offset];
// 			}
// 		}
// 	}
// }

void fill_dw_ifms_buffer_lower_part(fms_dt ifms_buffer[], fms_dt *filling_src,
									const int strides, const int filter_dim, int filling_d,
									const int additional_offset_in_filling_src,
									const int negative_shift_in_filling_dst)
{
#pragma HLS INLINE

	const int start_filling_offset = filter_dim - strides;
fill_dw_ifms_buffer_lower_part:
	for (int h = start_filling_offset; h < filter_dim; h++)
	{
#pragma HLS UNROLL
		for (int w = start_filling_offset; w < filter_dim; w++)
		{
#pragma HLS UNROLL
			ifms_buffer[h * filter_dim + w - negative_shift_in_filling_dst] =
				filling_src[(h - start_filling_offset) * strides + (w - start_filling_offset) + additional_offset_in_filling_src];
		}
	}
}

dw_pss_dt dw_kernel(fms_dt ifms_buffer[],
					const dw_weights_dt weights[],
					const int filter_dim)
{
#pragma HLS INLINE

	dw_pss_dt pss = 0;
dw_kernel:
	for (int c_h = 0; c_h < filter_dim; c_h++)
	{
#pragma HLS UNROLL
		for (int c_w = 0; c_w < filter_dim; c_w++)
		{
#pragma HLS UNROLL
			pss += ifms_buffer[c_h * filter_dim + c_w] * weights[c_h * filter_dim + c_w];
		}
	}
	return pss;
}
//*************************

//*************************
void projection_kernel(fms_dt ifms_val, const int ofms_depth,
					   const weights_dt weights[],
					   pss_dt pss_buffer[], int conv_d)
{
#pragma HLS INLINE

	pss_dt pss = 0;
projection_kernel:
	for (int i = 0; i < ofms_depth; i++)
	{
#pragma HLS UNROLL
		pss_buffer[i] += weights[i] * ifms_val;
	}
}
//*************************
void copy_projection_kernel_output_buffer(pss_dt projection_kernel_output_buffer[],
										  pss_dt projection_kernel_output_buffer_prev[], const int ofms_depth)
{
#pragma HLS INLINE

copy_projection_kernel_output_buffer:
	for (int i = 0; i < ofms_depth; i++)
	{
#pragma HLS UNROLL
		projection_kernel_output_buffer_prev[i] = projection_kernel_output_buffer[i];
		projection_kernel_output_buffer[i] = 0;
	}
}
//*************************
fms_dt normalize_projection_kernel_output(pss_dt pss_buffer[],
										const fused_scales_dt projection_layer_fused_scales[],
										const fused_scales_log_2_shifts_dt projection_layer_fused_scales_log_2_shifts[],
										const relu_6_fused_scales_dt projection_layer_relu_6_fused_scales[],
										const biases_dt projection_layer_fused_zero_points[],
										const int offset_d,
										const int layer_relu,
										int bottleneck_projection_layer_index)
{
#pragma HLS INLINE

	const fms_dt projection_layer_ofms_zero_point =
		conv_fms_zero_points[bottleneck_projection_layer_index + 1];
	const rec_scales_dt projection_layer_ofms_scale_rec =
		conv_fms_scales_rec[bottleneck_projection_layer_index + 1];
	const rec_scales_dt projection_layer_ofms_scale =
		conv_fms_scales[bottleneck_projection_layer_index + 1];

	// normalize_projection_kernel_output: for (int i = 0; i < ofms_depth; i++)
	// {
	fms_quantization_scheme projection_layer_normalization;
	projection_layer_normalization.ofm_zero_point =
		projection_layer_ofms_zero_point;
	projection_layer_normalization.ofm_scale_rec =
		projection_layer_ofms_scale_rec;
	projection_layer_normalization.ofm_scale = projection_layer_ofms_scale;

	projection_layer_normalization.fused_scales =
		projection_layer_fused_scales[offset_d];
	projection_layer_normalization.fused_scales_log_2_shift =
		projection_layer_fused_scales_log_2_shifts[offset_d];
	projection_layer_normalization.relu_6_fused_scale =
		projection_layer_relu_6_fused_scales[offset_d];
	projection_layer_normalization.fused_zero_point =
		projection_layer_fused_zero_points[offset_d];

	return pw_relu_norm(pss_buffer[offset_d], projection_layer_normalization, layer_relu);
	//	}
}
//*************************
