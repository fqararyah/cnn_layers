#include "bottlenecks_chain.h"
#include "bottlenecks_parallelism.h"
// padding left and right
// padding top: just do not fill

void fill_first_bottleneck_input(fms_dt chain_input[chain_input_height][chain_input_width][chain_input_depth],
                                 fms_dt bottleneck_1_input[bottlenck_1_buffer_size],
                                 int starting_w)
{
    const int bottleneck_1_input_wd = bottleneck_1_expansion_parallelism_w * bottleneck_1_ifms_depth;
    starting_w *= bottleneck_1_dw_strides;
    for (int h = 0; h < bottleneck_1_expansion_parallelism_h; h++)
    {
        for (int w = 0; w < bottleneck_1_expansion_parallelism_w; w++)
        {
            for (int d = 0; d < bottleneck_1_ifms_depth; d++)
            {
                bottleneck_1_input[h * bottleneck_1_input_wd + w * bottleneck_1_ifms_depth + d] = chain_input[h][w + starting_w][d];
            }
        }
    }
}

void bottlenecks_chain(fms_dt chain_input[chain_input_height][chain_input_width][chain_input_depth], int starting_h)
{
    fms_dt bottleneck_1_input[bottlenck_1_buffer_size];
    fms_dt bottleneck_1_output[bottleneck_1_ofms_depth];
    fms_dt *chain_output = bottleneck_1_output;
    fms_dt previous_pass_dw_input[bottlenck_1_inter_pass_dw_input_size];

    for (int w = 0; w < chain_output_width; w++)
    {
        fill_first_bottleneck_input(chain_input, bottleneck_1_input,
                                    w);
        mob_v2_bottleneck(bottleneck_1_input,
                          bottleneck_output,
                          previous_pass_dw_input,
                          pw_weights_4,
                          dw_weights_5,
                          pw_weights_6,
                          layer_4_fused_scales,
                          layer_4_fused_scales_log_2_shifts,
                          layer_4_relu_6_fused_scales,
                          layer_4_fused_zero_points,
                          layer_5_fused_scales,
                          layer_5_fused_scales_log_2_shifts,
                          layer_5_relu_6_fused_scales,
                          layer_5_fused_zero_points,
                          layer_6_fused_scales,
                          layer_6_fused_scales_log_2_shifts,
                          layer_6_relu_6_fused_scales,
                          layer_6_fused_zero_points,
                          bottleneck_1_ifms_depth,
                          bottleneck_1_ifms_height,
                          bottleneck_1_ifms_width,
                          bottleneck_1_ofms_depth,
                          bottleneck_1_ofms_width,
                          bottleneck_1_expanded_ifms_depth,
                          bottleneck_1_dw_filter_dim,
                          bottleneck_1_dw_strides,
                          starting_h,
                          w,
                          bottleneck_1_expansion_parallelism_h,
                          bottleneck_1_expansion_parallelism_w,
                          bottleneck_1_expansion_layer_index,
                          bottleneck_1_dw_layer_index,
                          bottleneck_1_projection_layer_index,
                          bottleneck_1_expansion_layer_relu,
                          bottleneck_1_dw_layer_relu,
                          bottleneck_1_projection_layer_relu,
                          bottleneck_1_dw_padding_left, bottleneck_1_dw_padding_right,
                          bottleneck_1_dw_padding_top, bottleneck_1_dw_padding_bottom);
    }
    
}