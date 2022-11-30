#include "bottlenecks_parallelism.h"

pss_dt expansion_kernel(fms_dt ifms_buffer[], const int ifms_depth, weights_dt filter[]);

void fill_dw_ifms_buffer_upper_part(fms_dt ifms_buffer[max_of_bottlenecks_layers_depths][],
                                    fms_dt *filling_src, const int strides, const int filter_dim,
                                    int ifms_w_offset, const int ifms_width, const int ifms_depth, int filling_d);
void fill_dw_ifms_buffer_lower_part(fms_dt ifms_buffer[max_of_bottlenecks_layers_depths][],
                                    fms_dt *filling_src, const int strides, const int filter_dim, int filling_d);
void shift_dw_ifms_buffer_horizontally(fms_dt ifms_buffer[max_of_bottlenecks_layers_depths][], const int strides, const int filter_dim,
                                       int conv_d);
dw_pss_dt dw_kernel(fms_dt ifms_buffer[max_of_bottlenecks_layers_depths][], weights_dt filter[], const int filter_dim, int conv_d);

void projection_kernel(fms_dt ifms_val, const int ofms_depth, const weights_dt weights[max_of_bottlenecks_projection_filters][], pss_dt pss_buffer[],
int conv_d);