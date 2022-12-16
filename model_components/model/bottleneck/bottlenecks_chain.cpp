#include "bottlenecks_chain.h"
#include "bottlenecks_parallelism.h"
#include "bottleneck_kernels.h"
// padding left and right
// padding top: just do not fill

void fill_first_bottleneck_input(fms_dt chain_input[],
                                 fms_dt bottleneck_1_input[bottlenck_1_buffer_size],
                                 const bottlenecks_chain_specs chain_specs;
                                 int starting_w)
{

    const int cols_filled_first_time = chain_specs.first_filter_dim;
    const int first_fill_offset = cols_filled_first_time * (starting_w != 0);
    const int bottleneck_1_input_wd = bottleneck_1_expansion_parallelism_w * bottleneck_1_ifms_depth;
    const int start_filling_index_in_chain_input =
        (starting_w * bottleneck_1_dw_strides + first_fill_offset - chain_specs.first_dw_padding_left) * bottleneck_1_ifms_depth;
    for (int h = 0; h < bottleneck_1_expansion_parallelism_h; h++)
    {
        for (int w = 0; w < bottleneck_1_expansion_parallelism_w; w++)
        {
            for (int d = 0; d < bottleneck_1_ifms_depth; d++)
            {
                if (starting_w * bottleneck_1_dw_strides + first_fill_offset >=
                        chain_specs.first_dw_padding_left &&
                    starting_w * bottleneck_1_dw_strides + first_fill_offset - chain_specs.first_dw_padding_left <
                        chain_specs.chain_input_width)
                {
                    bottleneck_1_input[h * bottleneck_1_input_wd + w * bottleneck_1_ifms_depth + d] =
                        chain_input[start_filling_index_in_chain_input + h * bottleneck_1_input_wd + w * bottleneck_1_ifms_depth + d];
                }
                else
                {
                    bottleneck_1_input[h * bottleneck_1_input_wd + w * bottleneck_1_ifms_depth + d] = zero_point;
                }
            }
        }
    }
}

void save_chain_output(fms_dw chain_output[], fms_dt result[max_fms_size],
                       const bottlenecks_chain_specs chain_specs, int h, int w)
{
#pragma HLS INLINE off

    const int num_of_tiles_w = chain_specs.chain_output_num_tiles_w;
    const int num_of_tiles_hw = chain_specs.chain_output_num_tiles_h * chain_specs.chain_output_num_tiles_w;
    const int tile_in_h = h / pw_tile_h;
    const int in_tile_h = h % pw_tile_h;
    const int tile_in_w = w / pw_tile_w;
    const int in_tile_w = w % pw_tile_w;

    for (int d = 0; d < chain_specs.chain_output_depth; d++)
    {
        const int tile_in_d = d / pw_tile_d;
        const int in_tile_d = d % pw_tile_d;
        const int tile_index = tile_in_d * num_of_tiles_hw + tile_in_h * num_of_tiles_w + tile_in_w;

        const int in_tile_index = in_tile_d * pw_tile_hw + in_tile_h * pw_tile_w + in_tile_w;

        const int index_in_result = tile_index * pw_tile_size + in_tile_index;

        result[index_in_result] = chain_output[d];
    }
}

void bottlenecks_chain(fms_dt chain_input[], // chain_input_height*chain_input_width*chain_input_depth
                       fms_dt result[max_fms_size],
                       const bottlenecks_chain_specs chain_specs, int starting_h)
{
    fms_dt bottleneck_1_input[bottlenck_1_buffer_size];
    fms_dt bottleneck_1_output[bottleneck_1_ofms_depth];
    fms_dt previous_pass_dw_input[bottlenck_1_inter_pass_dw_input_size];

    for (int w = 0; w < chain_specs.chain_output_width; w++)
    {
        fill_first_bottleneck_input(chain_input, bottleneck_1_input,
                                    w);
        mob_v2_bottleneck(bottleneck_1_input,
                          bottleneck_1_output,
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