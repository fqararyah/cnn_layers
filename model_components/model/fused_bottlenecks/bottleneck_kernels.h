#include "bottlenecks_parallelism.h"

pss_dt conv_kernel(fms_dt ifms_buffer[],
				   const layer_0_weights_dt weights_1[layer_1_s_num_fils][layer_1_s_depth][layer_1_s_filter_dim][layer_1_s_filter_dim],
				   const int filter_dim, int conv_d);

pss_dt expansion_kernel(fms_dt ifms_buffer[],
						const weights_dt weights[],
						const int ifms_depth, int filter_index, int h, int w,
						const int parallelism_w);

void fill_dw_ifms_buffer_upper_part(fms_dt ifms_buffer[], fms_dt *filling_src,
									const int strides, const int filter_dim, int ifms_w_offset,
									const int ifms_width, const int ifms_depth, int filling_d,
									const int padding_left);
void fill_dw_ifms_buffer_lower_part(fms_dt ifms_buffer[], fms_dt *filling_src,
									const int strides, const int filter_dim, int filling_d,
									const int additional_offset_in_filling_src,
									const int negative_shift_in_filling_dst);
void update_dw_ifms_buffer_upper_part(fms_dt *dw_ifms_buffer_upper_part,
									  fms_dt *filling_src, const int strides, const int filter_dim,
									  int ifms_w_offset, const int ifms_width, const int ifms_depth,
									  int filling_d, const int padding_left,
									  const int first_fill_from_left_offset, bool do_shift);
void shift_dw_ifms_buffer_horizontally_3x3_s1(fms_dt ifms_buffer[][3], const int buffer_depth);
void shift_dw_ifms_buffer_horizontally_3x3_s2(fms_dt ifms_buffer[][6], const int buffer_depth);
dw_pss_dt dw_kernel(fms_dt ifms_buffer[],
					const dw_weights_dt weights[],
					const int filter_dim);

void projection_kernel(fms_dt ifms_val, const int ofms_depth,
					   const weights_dt weights[],
					   pss_dt pss_buffer[], int conv_d);

void copy_projection_kernel_output_buffer(pss_dt projection_kernel_output_buffer[],
										  pss_dt projection_kernel_output_buffer_prev[], const int ofms_depth);

fms_dt normalize_projection_kernel_output(pss_dt pss_buffer[],
										const fused_scales_dt projection_layer_fused_scales[],
										const fused_scales_log_2_shifts_dt projection_layer_fused_scales_log_2_shifts[],
										const relu_6_fused_scales_dt projection_layer_relu_6_fused_scales[],
										const biases_dt projection_layer_fused_zero_points[],
										const int offset_d,
										const int layer_relu,
										const layer_specs layer_specs_struct);

pss_f_dt normalize_projection_kernel_output_no_q(pss_dt pss_buffer[],
										const fused_scales_dt projection_layer_fused_scales[],
										const fused_scales_log_2_shifts_dt projection_layer_fused_scales_log_2_shifts[],
										const biases_dt projection_layer_fused_zero_points[],
										const int offset_d,
										const int layer_relu,
										const layer_specs layer_specs_struct);
