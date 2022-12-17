#include "bottlenecks_parallelism.h"

pss_dt expansion_kernel(fms_dt ifms_buffer[], const int ifms_depth,
		const weights_dt weights[][16], int filter_index, int h, int w,
		const int parallelism_w);

void fill_dw_ifms_buffer_upper_part(fms_dt ifms_buffer[], fms_dt *filling_src,
		const int strides, const int filter_dim, int ifms_w_offset,
		const int ifms_width, const int ifms_depth, int filling_d,
		const int padding_left);
void fill_dw_ifms_buffer_lower_part(fms_dt ifms_buffer[], fms_dt *filling_src,
		const int strides, const int filter_dim, int filling_d);
void update_dw_ifms_buffer_upper_part(fms_dt *dw_ifms_buffer_upper_part,
		fms_dt *filling_src, const int strides, const int filter_dim,
		int ifms_w_offset, const int ifms_width, const int ifms_depth,
		int filling_d, const int padding_left);
void shift_dw_ifms_buffer_horizontally(fms_dt ifms_buffer[], const int strides,
		const int filter_dim, int conv_d);
dw_pss_dt dw_kernel(fms_dt ifms_buffer[],
		const dw_weights_dt weights[][max_dw_filter_area_in_a_chain],
		const int filter_dim, int conv_d);

void projection_kernel(fms_dt ifms_val, const int ofms_depth,
		const weights_dt weights[][max_of_bottlenecks_layers_depths],
		pss_dt pss_buffer[], int conv_d);

void normalize_projection_kernel_output(pss_dt pss_buffer[],
		fms_dt normalized_buffer[],
		const fused_scales_dt projection_layer_fused_scales[],
		const fused_scales_log_2_shifts_dt projection_layer_fused_scales_log_2_shifts[],
		const relu_6_fused_scales_dt projection_layer_relu_6_fused_scales[],
		const biases_dt projection_layer_fused_zero_points[],
		const int ofms_depth, const int layer_relu,
		int bottleneck_projection_layer_index);
