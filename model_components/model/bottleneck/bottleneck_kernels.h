#include "bottlenecks_parallelism.h"

pss_dt expansion_kernel(fms_dt ifms_buffer[], const int ifms_depth, weights_dt filter[]);

void projection_kernel(fms_dt ifms_val, const int ofms_depth, weights_dt filter[], pss_dt pss_buffer[]);

void fill_dw_ifms_buffer_upper_part(fms_dt ifms_buffer[], fms_dt *filling_src, const int strides, const int filter_dim, int ifms_w_offset, const int ifms_width);
void fill_dw_ifms_buffer_lower_part(fms_dt ifms_buffer[], fms_dt *filling_src, const int strides, const int filter_dim);
void shift_dw_ifms_buffer_horizontally(fms_dt ifms_buffer[], const int strides, const int filter_dim);

pss_dt dw_kernel(fms_dt ifms_buffer[], weights_dt filter[], const int filter_dim);