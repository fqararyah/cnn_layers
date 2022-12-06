#include "bottlenecks_sesl_specs.h"

void mob_v2_bottleneck_block_1(fms_dt bottleneck_input[bottleneck_1_ifms_depth][bottleneck_1_input_buffer_height]
                                                    [bottleneck_1_ifms_width],
                             fms_dt bottleneck_output[bottleneck_1_ofms_depth][bottleneck_1_ofms_width],
                             fms_dt previous_pass_output[bottleneck_1_expanded_ifms_depth][bottleneck_1_ifms_width];// height=1
                             const weights_dt layer_4_pw_weights[layer_4_pw_num_fils][layer_4_pw_depth],
                             const dw_weights_dt layer_5_dw_weights[layer_5_dw_depth][layer_5_dw_filter_size*layer_5_dw_filter_size],
                             const weights_dt layer_6_pw_weights[layer_6_pw_num_fils][layer_6_pw_depth],
                             const int current_h); // bottleneck_1_ofms_height=1