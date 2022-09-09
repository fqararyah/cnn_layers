#include "./headers/utils.h"

void _7_layer_0_3x3_conv(
    fms_dt channels_buffer[input_image_depth][layer_0_filter_size + (_7_stages_layer_1_rows_at_once - 1) * layer_0_strides][input_image_width],
    layer_0_weights_dt weights[layer_0_num_fils][layer_0_depth][layer_0_filter_size][layer_0_filter_size],
    fms_dt result[layer_1_dw_depth][_7_stages_layer_1_rows_at_once][layer_1_dw_ifm_width])
{
#pragma HLS INLINE off

    fms_dt intermediate_channels_buffer[input_image_depth][layer_0_filter_size + (_7_stages_layer_1_rows_at_once - 1) * layer_0_strides][layer_1_dw_filter_size] = {0};
#pragma HLS ARRAY_PARTITION variable = intermediate_channels_buffer type = complete dim = 0

    // fill the intermediate_channels_buffer
    for (int d = 0; d < input_image_depth; d++)
    {
        for (int h = 0; h < layer_0_filter_size + (_7_stages_layer_1_rows_at_once - 1) * layer_0_strides; h++)
        {
            for (int w = 0; w < layer_0_filter_size - layer_0_padding_left; w++)
            {
                intermediate_channels_buffer[d][h][w + layer_0_padding_left] = channels_buffer[d][h][w];
            }
        }
    }
    // end fill the intermediate_channels_buffer

layer_0_ofms:
    for (int o_o_d = 0;
         o_o_d < layer_0_num_fils / layer_0_parallelism_ofms; o_o_d++)
    {
        // outer filters loop
        int o_o_d_offset = o_o_d * layer_0_parallelism_ofms; // for indexing in depth

    layer_0_pipeline:
        for (int w = 0; w < input_image_width; w +=
                                                    layer_0_strides)
        {
#pragma HLS PIPELINE
            // FMs width loop
            for (int row = 0; row <= _7_stages_layer_1_rows_at_once; row++)
            {
#pragma HLS UNROLL
            layer_0_parallelized_ofms:
                for (int o_d = 0;
                     o_d < layer_0_parallelism_ofms; o_d++)
                {
                    first_conv_pss_dt tmp = 0;
#pragma HLS UNROLL
                // parallelized filters loop
                layer_0_d_loops:
                    for (int d = 0; d < input_image_depth; d++)
                    {
#pragma HLS UNROLL
                    // parallelized depth loop
                    layer_0_ch:
                        for (int h = 0; h < layer_0_filter_size; h++)
                        {
#pragma HLS UNROLL
                        // conv height loop
                        layer_0_cw:
                            for (int c_w = 0; c_w < layer_0_filter_size; c_w++)
                            {
#pragma HLS UNROLL
                                // conv width loop
                                tmp += intermediate_channels_buffer[d][row + h][c_w] * weights[o_o_d_offset + o_d][d][h][c_w];
                            }
                        }
                    }
                    fms_dt scaled_val = (fms_dt)(((ap_fixed<17, 12>)tmp - zero_point_dw) * ratio_dw_pss_to_fms);
                    if (scaled_val > 0)
                    {
                        result[o_o_d_offset + o_d][row][w / layer_0_strides] =
                            scaled_val;
                    }
                }
            }
            // shift and fill the intermediate_channels_buffer
            if (w < input_image_width - layer_0_strides)
            {
                for (int d = 0; d < input_image_depth; d++)
                {
#pragma HLS UNROLL
                    for (int c_h = 0; c_h < layer_0_filter_size + (_7_stages_layer_1_rows_at_once - 1) * layer_0_strides; c_h++)
                    {
#pragma HLS UNROLL
                        for (int c_w = 0; c_w < layer_0_filter_size - layer_0_strides; c_w++)
                        {
#pragma HLS UNROLL
                            intermediate_channels_buffer[d][c_h][c_w] = intermediate_channels_buffer[d][c_h][c_w + layer_0_strides];
                        }
                        for (int c_w = layer_0_filter_size - layer_0_strides; c_w < layer_0_filter_size; c_w++)
                        {
#pragma HLS UNROLL
                            intermediate_channels_buffer[d][c_h][c_w] = channels_buffer[d][c_h][c_w - (layer_0_filter_size - layer_0_strides) +
                                                                                                (w + layer_0_filter_size - layer_0_padding_left)];
                        }
                    }
                }
            }
            // end shift and fill the intermediate_channels_buffer
        }
    }
}

void _7_layer_1_pw_dw(
    fms_dt channels_buffer[layer_1_pw_depth][_7_stages_layer_1_rows_at_once][layer_1_pw_ifm_width],
    weights_dt weights[layer_1_pw_num_fils][layer_1_pw_depth],
    dw_weights_dt dw_weights[layer_1_dw_depth][layer_1_dw_filter_size][layer_1_dw_filter_size],
    fms_dt upper[layer_1_dw_depth][layer_1_dw_filter_size - layer_1_dw_strides][layer_1_dw_ifm_width],
    fms_dt lower[layer_1_dw_depth][_7_stages_layer_1_rows_at_once][layer_1_dw_ifm_width],
    fms_dt result[layer_2_pw_depth][_7_stages_layer_1_rows_at_once][layer_2_pw_ifm_width], int active_row)
{

#pragma HLS INLINE off
layer_1_pw__dw_main_loop:
    for (int row = 0; row < _7_stages_layer_1_rows_at_once;
         row++)
    {
#pragma HLS UNROLL
        // rows for next DW
        fms_dt intermediate_pw_results[layer_1_pw_parallelism_out][layer_1_dw_filter_size][layer_1_dw_filter_size];
#pragma HLS ARRAY_PARTITION variable = intermediate_pw_results type = complete dim = 0
        for (int o_o_d = 0;
             o_o_d < layer_1_pw_num_fils / layer_1_pw_parallelism_out;
             o_o_d++)
        {
            int o_o_d_offset = o_o_d * layer_1_pw_parallelism_out;

            // fill upper and lower (except last) rows:
            for (int d = 0; d < layer_1_pw_parallelism_out; d++)
            {
#pragma HLS UNROLL
                for (int h = 0; h < layer_1_dw_filter_size - layer_1_dw_strides - row; h++)
                {
#pragma HLS UNROLL
                    // padding
                    intermediate_pw_results[d][h][0] = 0;
                    for (int w = 1; w < layer_1_dw_filter_size; w++)
                    {
#pragma HLS UNROLL
                        intermediate_pw_results[d][h][w] = upper[o_o_d_offset + d][h + row][w];
                    }
                }

                for (int h = 0; h < row; h++)
                {
#pragma HLS UNROLL
                    // padding
                    intermediate_pw_results[d][h][0] = 0;
                    for (int w = layer_1_dw_padding_left; w < layer_1_dw_filter_size; w++)
                    {
#pragma HLS UNROLL
                        intermediate_pw_results[d][h + (layer_1_dw_filter_size - layer_1_dw_strides - row)][w] = lower[o_o_d_offset + d][row][w - layer_1_dw_padding_left];
                    }
                }
                intermediate_pw_results[d][layer_1_dw_filter_size - 1][0] = 0;
            }

        layer_1_pw_pipeline:
            for (int w = 0; w < layer_1_dw_ifm_width + v1_layer_3_dw_filter_size - (v1_layer_3_dw_padding_left + layer_1_dw_padding_right);
				 w++)
            {
#pragma HLS PIPELINE
            //###################PW#######################
            layer_1_pw_loops:
                for (int o_d = 0;
                     o_d < layer_1_pw_parallelism_out; o_d++)
                {
#pragma HLS UNROLL
                    // parallelized filters loop
                    if (w < layer_1_dw_ifm_width)
                    {
                        // FMs width loop
                        pss_dt tmp = 0;
                        for (int d = 0; d < layer_1_pw_parallelism_in; d++)
                        {
#pragma HLS UNROLL
                            // parallelized depth loop
                            tmp +=
                                ((fms_dt)channels_buffer[d][row][w]) * weights[o_o_d_offset + o_d][d];
                        }
                        fms_dt scaled_val = (fms_dt)((((ap_fixed<17, 12>)tmp) - zero_point_dw) * ratio_dw_pss_to_fms);
                        if (scaled_val > 0)
                        {
                            if (w + layer_1_dw_padding_left < layer_1_dw_filter_size)
                            {
                                intermediate_pw_results[o_d][layer_1_dw_filter_size - 1][w + layer_1_dw_padding_left] = scaled_val;
                            }
                            lower[o_o_d_offset + o_d][row][w] = scaled_val;
                        }
                    }
                    //###############end PW####################
                    //###############DW########################
                    if (w + 1 >= layer_1_dw_filter_size - layer_1_dw_padding_left)
                    {
                        if (w < layer_1_dw_ifm_width)
                        {
                            intermediate_pw_results[o_d][layer_1_dw_filter_size - 1][layer_1_dw_filter_size - layer_1_dw_padding_left] =
                                lower[o_o_d_offset + o_d][row][w];
                        }
                        else
                        {
                            intermediate_pw_results[o_d][layer_1_dw_filter_size - 1][layer_1_dw_filter_size - layer_1_dw_padding_left] = 0;
                        }
                        dw_pss_dt tmp = 0;
                        // parallelized depth loop
                        for (int c_h = 0; c_h < layer_1_dw_filter_size; c_h++)
                        {
#pragma HLS UNROLL
                            for (int c_w = 0; c_w < layer_1_dw_filter_size; c_w++)
                            {
                                // conv width loop
#pragma HLS UNROLL
                                tmp += intermediate_pw_results[o_d][c_h][c_w] * dw_weights[o_o_d_offset + o_d][c_h][c_w];
                            }
                        }
                        fms_dt scaled_val = (fms_dt)((((ap_fixed<17, 12>)tmp) - zero_point_dw) * ratio_dw_pss_to_fms);
                        if (scaled_val > 0)
                        {
                            result[o_o_d_offset + o_d][row][(w + 1 - (layer_1_dw_filter_size - layer_1_dw_padding_left)) / layer_1_dw_strides] =
                                scaled_val;
                        }
                        //#####################end DW################
                        //#####################shift and fill intermediate#################
                        for (int c_h = 0; c_h < layer_1_dw_filter_size; c_h++)
                        {
#pragma HLS UNROLL
                            for (int c_w = 0; c_w < layer_1_dw_filter_size - layer_1_dw_strides; c_w++)
                            {
#pragma HLS UNROLL
                                intermediate_pw_results[o_d][c_h][c_w] = intermediate_pw_results[o_d][c_h][c_w + layer_1_dw_strides];
                            }
                            if (c_h + row < layer_1_dw_filter_size - layer_1_dw_strides)
                            {
                                for (int c_w = layer_1_dw_filter_size - layer_1_dw_strides; c_w < layer_1_dw_filter_size; c_w++)
                                {
#pragma HLS UNROLL
                                    if (w < layer_1_dw_ifm_width)
                                    {
                                        intermediate_pw_results[o_d][c_h][c_w] = upper[o_d][c_h + row][w + layer_1_dw_padding_left];
                                    }
                                    else
                                    {
                                        intermediate_pw_results[o_d][c_h][c_w] = 0;
                                    }
                                }
                            }
                            if (c_h < layer_1_dw_filter_size - layer_1_dw_strides && c_h + row >= layer_1_dw_filter_size - layer_1_dw_strides)
                            {
                                intermediate_pw_results[o_d][c_h][layer_1_dw_filter_size - 1] = lower[o_o_d_offset + o_d][0][w + layer_1_dw_padding_left];
                            }
                        }
                        //#####################end shift and fill intermediate#################
                    }
                }
            }
        }
    }

layer_1_pw_dw_shift_loop:
    for (int o_o_d = 0;
         o_o_d < layer_1_pw_num_fils / layer_1_pw_parallelism_out;
         o_o_d++)
    {
        int o_o_d_offset = o_o_d * layer_1_pw_parallelism_out;
    layer_1_shift_pipeline:
        for (int w = 0; w < layer_1_dw_ifm_width;
             w++)
        {
#pragma HLS UNROLL
        //###################PW#######################
        layer_1_shift_loops:
            for (int o_d = 0;
                 o_d < layer_1_pw_parallelism_out; o_d++)
            {
#pragma HLS UNROLL
                if (active_row == 0)
                {
                    upper[o_o_d_offset + o_d][0][w] = 0;
                    upper[o_o_d_offset + o_d][1][w] = lower[o_o_d_offset + o_d][0][w];
                }
                else
                {
                    upper[o_o_d_offset + o_d][0][w] = lower[o_o_d_offset + o_d][0][w];
                    upper[o_o_d_offset + o_d][1][w] = lower[o_o_d_offset + o_d][1][w];
                }
            }
        }
    }
}

void _7_layer_2_pw(
    fms_dt channels_buffer[layer_2_pw_depth][_7_stages_layer_1_rows_at_once][layer_2_pw_ifm_width],
    weights_dt weights[layer_2_pw_num_fils][layer_2_pw_depth],
    fms_dt result[layer_3_pw_depth][_7_stages_layer_1_rows_at_once][layer_3_dw_ifm_width])
{

#pragma HLS INLINE off

    // rows for next DW
    for (int o_o_d = 0;
         o_o_d < layer_2_pw_num_fils / layer_2_pw_parallelism_out;
         o_o_d++)
    {
        int o_o_d_offset = o_o_d * layer_2_pw_parallelism_out;
        // filters loop
    layer_2_pw_pipeline:
        for (int w = 0; w < layer_2_pw_ifm_width; w++)
        {
#pragma HLS PIPELINE
            for (int row = 0; row < _7_stages_layer_1_rows_at_once;
                 row++)
            {
            // FMs width loop
            layer_2_pw_loops:
                for (int o_d = 0;
                     o_d < layer_2_pw_parallelism_out; o_d++)
                {
#pragma HLS UNROLL
                    // parallelized filters loop
                    pss_dt tmp = 0;
                    for (int d = 0; d < layer_2_pw_parallelism_in; d++)
                    {
#pragma HLS UNROLL
                        // parallelized depth loop
                        tmp += ((fms_dt)channels_buffer[d][row][w]) * weights[o_o_d_offset + o_d][d];
                    }
                    fms_dt scaled_val = (fms_dt)((((ap_fixed<17, 12>)tmp) - zero_point_dw) * ratio_dw_pss_to_fms);
                    if (scaled_val > 0)
                    {
                        result[o_o_d_offset + o_d][row][w] += scaled_val;
                    }
                }
            }
        }
    }
}

void _7_layer_3_pw_dw(
    fms_dt channels_buffer[layer_3_pw_depth][layer_3_dw_strides][layer_3_dw_ifm_width],
    weights_dt weights[layer_3_pw_num_fils][layer_3_pw_depth],
    dw_weights_dt dw_weights[layer_3_dw_depth][layer_3_dw_filter_size][layer_3_dw_filter_size],
    fms_dt upper[layer_3_dw_depth][layer_3_dw_ifm_width],
    fms_dt lower[layer_3_dw_depth][layer_3_dw_strides][layer_3_dw_ifm_width],
    fms_dt result[layer_4_pw_depth][layer_4_pw_ifm_width], int active_row) // active_row is depricated, just pass strides of that layer - 1
{

    fms_dt intermediate_pw_results[layer_3_pw_parallelism_out][layer_3_dw_filter_size][layer_3_dw_filter_size];
#pragma HLS ARRAY_PARTITION variable = intermediate_pw_results type = complete dim = 0

#pragma HLS INLINE off
layer_3_pw__dw_main_loop:
    for (int o_o_d = 0;
         o_o_d < layer_3_pw_num_fils / layer_3_pw_parallelism_out;
         o_o_d++)
    {
        int o_o_d_offset = o_o_d * layer_3_pw_parallelism_out;

        // fill upper and lower (except last) rows:
        for (int d = 0; d < layer_3_pw_parallelism_out; d++)
        {
#pragma HLS UNROLL
            for (int h = 0; h < layer_3_dw_filter_size - layer_3_dw_strides; h++)
            {
#pragma HLS UNROLL
                // padding
                intermediate_pw_results[d][h][0] = 0;
                for (int w = 1; w < layer_3_dw_filter_size; w++)
                {
#pragma HLS UNROLL
                    intermediate_pw_results[d][h][w] = upper[d][h][w];
                }
            }
            intermediate_pw_results[d][1][0] = 0;
            intermediate_pw_results[d][2][0] = 0;
        }

    layer_3_pw_pipeline:
        for (int w = 0; w < layer_3_dw_ifm_width + layer_3_dw_padding_left + layer_3_dw_padding_right;
             w++)
        {
#pragma HLS PIPELINE
        //###################PW#######################
        layer_1_pw_loops:
            for (int o_d = 0;
                 o_d < layer_3_pw_parallelism_out; o_d++)
            {
#pragma HLS UNROLL
                // parallelized filters loop
                if (w < layer_3_dw_ifm_width)
                {
                    for (int row = 0; row < _7_stages_layer_3_rows_at_once;
                         row++)
                    {
#pragma HLS UNROLL
                        // FMs width loop
                        pss_dt tmp = 0;
                        for (int d = 0; d < layer_3_pw_parallelism_in; d++)
                        {
#pragma HLS UNROLL
                            // parallelized depth loop
                            tmp +=
                                ((fms_dt)channels_buffer[d][row][w]) * weights[o_o_d_offset + o_d][d];
                        }
                        fms_dt scaled_val = (fms_dt)((((ap_fixed<17, 12>)tmp) - zero_point_dw) * ratio_dw_pss_to_fms);
                        if (scaled_val > 0)
                        {
                            if (w + 1 < layer_1_dw_filter_size)
                            {
                                intermediate_pw_results[o_d][row + layer_3_dw_filter_size - layer_3_dw_strides][w + layer_3_dw_padding_left] = scaled_val;
                            }
                            lower[o_o_d_offset + o_d][row][w] = scaled_val;
                        }
                    }
                }
                //###############end PW####################
                //###############DW########################
                if (w + 1 >= layer_3_dw_filter_size - layer_3_dw_padding_left && active_row == layer_3_dw_strides - 1 &&
                 (w + 1 - (layer_3_dw_filter_size - layer_3_dw_padding_left)) % layer_3_dw_strides == 0)
                {
                    for (int row = 0; row < layer_3_dw_strides; row++)
                    {
                        for (int c_w = layer_3_dw_filter_size - layer_3_dw_strides; c_w < layer_3_dw_filter_size; c_w++)
                        {
                            // conv width loop
#pragma HLS UNROLL
                            intermediate_pw_results[o_d][layer_3_dw_filter_size - layer_3_dw_strides + row][c_w] =
                                lower[o_o_d_offset + o_d][row][w + 1 - (layer_3_dw_filter_size - layer_3_dw_padding_left) + 
                                (c_w - ( layer_3_dw_filter_size - layer_3_dw_strides) )];
                        }
                    }
                    dw_pss_dt tmp = 0;
                    // parallelized depth loop
                    for (int c_h = 0; c_h < layer_3_dw_filter_size; c_h++)
                    {
#pragma HLS UNROLL
                        for (int c_w = 0; c_w < layer_3_dw_filter_size; c_w++)
                        {
                            // conv width loop
#pragma HLS UNROLL
                            tmp += intermediate_pw_results[o_d][c_h][w + c_w] * dw_weights[o_o_d_offset + o_d][c_h][c_w];
                        }
                    }
                    fms_dt scaled_val = (fms_dt)((((ap_fixed<17, 12>)tmp) - zero_point_dw) * ratio_dw_pss_to_fms);
                    if (scaled_val > 0)
                    {
                        result[o_o_d_offset + o_d][(w - layer_3_dw_padding_left) / layer_3_dw_strides] =
                            scaled_val;
                    }
                    //#####################end DW################
                    //#####################shift and fill intermediate#################
                    for (int c_h = 0; c_h < layer_3_dw_filter_size; c_h++)
                    {
#pragma HLS UNROLL
                        for (int c_w = 0; c_w < layer_3_dw_filter_size - layer_3_dw_strides; c_w++)
                        {
#pragma HLS UNROLL
                            intermediate_pw_results[o_d][c_h][c_w] = intermediate_pw_results[o_d][c_h][c_w + layer_3_dw_strides];
                        }
                        if (c_h < layer_3_dw_filter_size - layer_3_dw_strides)
                        {
                            for (int c_w = layer_3_dw_filter_size - layer_3_dw_strides; c_w < layer_3_dw_filter_size; c_w++)
                            {
#pragma HLS UNROLL
                                intermediate_pw_results[o_d][c_h][c_w] = upper[o_o_d_offset + o_d][w + (c_w - (layer_3_dw_filter_size - layer_3_dw_strides) )
                                 + layer_3_dw_padding_left];
                            }
                        }
                    }
                    //#####################end shift and fill intermediate#################
                }
            }
        }
    }

layer_3_pw_dw_shift_loop:
    for (int o_o_d = 0;
         o_o_d < layer_3_pw_num_fils / layer_3_pw_parallelism_out;
         o_o_d++)
    {
        int o_o_d_offset = o_o_d * layer_3_pw_parallelism_out;
    layer_3_shift_pipeline:
        for (int w = 0; w < layer_3_dw_ifm_width;
             w++)
        {
#pragma HLS UNROLL factor = 4
        //###################PW#######################
        layer_1_shift_loops:
            for (int o_d = 0;
                 o_d < layer_3_pw_parallelism_out; o_d++)
            {
#pragma HLS UNROLL
                upper[o_o_d_offset + o_d][w] =
                    lower[o_o_d_offset + o_d][1][w];
            }
        }
    }
}

void _7_layer_4_pw(
    fms_dt channels_buffer[layer_4_pw_depth][layer_4_pw_ifm_width],
    weights_dt weights[layer_4_pw_num_fils][layer_4_pw_depth],
    fms_dt result[max_fms_size], int starting_h)
{

#pragma HLS INLINE off

    // rows for next DW
    for (int o_o_d = 0;
         o_o_d < layer_4_pw_num_fils / layer_4_pw_parallelism_out;
         o_o_d++)
    {
        int o_o_d_offset = o_o_d * layer_4_pw_parallelism_out;
        // filters loop
    layer_4_pw_pipeline:
        for (int w = 0; w < layer_4_pw_ifm_width; w++)
        {
#pragma HLS PIPELINE
        // FMs width loop
        layer_4_pw_loops:
            for (int o_d = 0;
                 o_d < layer_4_pw_parallelism_out; o_d++)
            {
#pragma HLS UNROLL
                // parallelized filters loop
                pss_dt tmp = 0;
                int offset_in_result = (o_o_d * layer_4_pw_parallelism_out + o_d) * switch_point_fms_height * switch_point_fms_width + starting_h * switch_point_fms_width + w;
                for (int d = 0; d < layer_4_pw_parallelism_in; d++)
                {
#pragma HLS UNROLL
                    // parallelized depth loop
                    tmp += ((fms_dt)channels_buffer[d][w]) * weights[o_d][d];
                }
                fms_dt scaled_val = (fms_dt)((((ap_fixed<17, 12>)tmp) - zero_point_dw) * ratio_dw_pss_to_fms);
                if (scaled_val > 0)
                {
                    result[offset_in_result] = scaled_val; // must be fully unrolled in the IFMs depth dimension
                }
            }
        }
    }
}

void _7_stages_fill_channels_buffer(
    fms_dt channels[input_image_depth][input_image_height][input_image_width],
    fms_dt channels_buffer_0[input_image_depth][layer_0_filter_size + (_7_stages_layer_1_rows_at_once - 1) * layer_0_strides][input_image_width],
    int starting_h)
{

    int h_offset = starting_h < _7_stages_layer_1_rows_at_once * layer_0_strides ? layer_0_strides : layer_0_filter_size + (_7_stages_layer_1_rows_at_once - 1) * layer_0_strides - (layer_0_filter_size - layer_0_strides);

    for (int w = 0; w < input_image_width; w++)
    {
#pragma HLS PIPELINE
        for (int d = 0; d < input_image_depth; d++)
        {
#pragma HLS UNROLL
            for (int h = 0; h < layer_0_filter_size - layer_0_strides; h++)
            {
#pragma HLS UNROLL
                channels_buffer_0[d][h][w] = channels_buffer_0[d][h + h_offset][w];
            }
        }
    }

    for (int w = 0; w < input_image_width; w++)
    {
#pragma HLS PIPELINE
        for (int d = 0; d < input_image_depth; d++)
        {
#pragma HLS UNROLL
            for (int h = layer_0_filter_size - layer_0_strides; h < layer_0_filter_size + (_7_stages_layer_1_rows_at_once - 1) * layer_0_strides; h++)
            {
#pragma HLS UNROLL
                if (starting_h + h - (layer_0_filter_size - layer_0_strides) < input_image_height)
                {
                    channels_buffer_0[d][h][w] = channels[d][starting_h + h - (layer_0_filter_size - layer_0_strides)][w];
                }
            }
        }
    }
}

void mobilenet_v2_pipeline_7(
    fms_dt channels[input_image_depth][input_image_height][input_image_width],
    fms_dt result[max_fms_size])
{
#pragma HLS INLINE off

#pragma HLS ARRAY_PARTITION variable = channels type = complete dim = 1

    dw_weights_dt dw_weights_1[layer_1_dw_depth][layer_1_dw_filter_size][layer_1_dw_filter_size];
    dw_weights_dt dw_weights_3[layer_3_dw_depth][layer_3_dw_filter_size][layer_3_dw_filter_size];

#pragma HLS ARRAY_PARTITION variable = dw_weights_1 type = complete dim = 1

    layer_0_weights_dt weights_0[layer_0_num_fils][layer_0_depth][layer_0_filter_size][layer_0_filter_size];

    weights_dt pw_weights_1[layer_1_pw_num_fils][layer_1_pw_depth];
    weights_dt pw_weights_2[layer_2_pw_num_fils][layer_2_pw_depth];
    weights_dt pw_weights_3[layer_3_pw_num_fils][layer_3_pw_depth];
    weights_dt pw_weights_4[layer_4_pw_num_fils][layer_4_pw_depth];

    _7_fill_layers_weights(weights_0,
                           dw_weights_1,
                           dw_weights_3,
                           pw_weights_1,
                           pw_weights_2,
                           pw_weights_3,
                           pw_weights_4);

    //#########################even###############################
    fms_dt channels_buffer_0[input_image_depth][layer_0_filter_size + (_7_stages_layer_1_rows_at_once - 1) * layer_0_strides][input_image_width];
#pragma HLS ARRAY_PARTITION variable = channels_buffer_0 complete dim = 1
#pragma HLS ARRAY_PARTITION variable = channels_buffer_0 complete dim = 2
    fms_dt _7_layer_0_3x3_conv_out_0[layer_1_pw_depth][_7_stages_layer_1_rows_at_once][layer_1_pw_ifm_width] =
        {0};

    fms_dt _7_layer_1_pw_out_0[layer_1_dw_depth][layer_1_dw_ifm_width] = {0};

    //##############
    fms_dt _7_layer_1_dw_upper[layer_1_dw_depth][layer_1_dw_filter_size - layer_1_dw_strides][layer_1_dw_ifm_width];
#pragma HLS ARRAY_PARTITION variable = _7_layer_1_dw_upper complete dim = 3

    fms_dt _7_layer_1_dw_lower[layer_1_dw_depth][_7_stages_layer_1_rows_at_once][layer_1_dw_ifm_width];
#pragma HLS ARRAY_PARTITION variable = _7_layer_1_dw_lower complete dim = 3

    fms_dt _7_layer_1_dw_out_0[layer_2_pw_depth][_7_stages_layer_1_rows_at_once][layer_2_pw_ifm_width] = {0};
#pragma HLS ARRAY_PARTITION variable = _7_layer_1_dw_out_0 complete dim = 1
    //##############

    fms_dt _7_layer_2_pw_out_0[layer_3_pw_depth][_7_stages_layer_1_rows_at_once][layer_3_dw_ifm_width] = {0};

    fms_dt _7_layer_3_dw_upper[layer_3_dw_depth][layer_3_dw_ifm_width];
#pragma HLS ARRAY_PARTITION variable = _7_layer_3_dw_upper cyclic factor = layer_3_pw_parallelism_out dim = 1
#pragma HLS ARRAY_PARTITION variable = _7_layer_3_dw_upper cyclic factor = 6 dim = 2

    fms_dt _7_layer_3_dw_lower[layer_3_dw_depth][layer_3_dw_strides][layer_3_dw_ifm_width];
#pragma HLS ARRAY_PARTITION variable = _7_layer_3_dw_lower cyclic factor = 2 dim = 1
#pragma HLS ARRAY_PARTITION variable = _7_layer_3_dw_lower complete dim = 2
#pragma HLS ARRAY_PARTITION variable = _7_layer_3_dw_lower cyclic factor = 12 dim = 3

    fms_dt _7_layer_3_pw_dw_out_0[layer_3_pw_depth][layer_4_pw_ifm_width] = {0};

#pragma HLS ARRAY_PARTITION variable = _7_layer_0_3x3_conv_out_0 complete dim = 1
#pragma HLS ARRAY_PARTITION variable = _7_layer_0_3x3_conv_out_0 complete dim = 2

#pragma HLS ARRAY_PARTITION variable = _7_layer_2_pw_out_0 complete dim = 1

#pragma HLS ARRAY_PARTITION variable = _7_layer_3_pw_dw_out_0 cyclic factor = layer_4_pw_parallelism_in/2 dim = 1
    //###########################################################

    //#########################odd###############################
    fms_dt channels_buffer_1[input_image_depth][layer_0_filter_size + (_7_stages_layer_1_rows_at_once - 1) * layer_0_strides][input_image_width];
#pragma HLS ARRAY_PARTITION variable = channels_buffer_1 complete dim = 1
#pragma HLS ARRAY_PARTITION variable = channels_buffer_1 complete dim = 2

    fms_dt _7_layer_0_3x3_conv_out_1[layer_1_pw_depth][_7_stages_layer_1_rows_at_once][layer_1_pw_ifm_width] =
        {0};

    fms_dt _7_layer_1_pw_out_1[layer_1_dw_depth][layer_1_dw_ifm_width] = {0};

    //##############

    fms_dt _7_layer_1_dw_out_1[layer_2_pw_depth][_7_stages_layer_1_rows_at_once][layer_2_pw_ifm_width] = {0};
    //##############
    fms_dt _7_layer_2_pw_out_1[layer_3_pw_depth][_7_stages_layer_1_rows_at_once][layer_3_dw_ifm_width] = {0};

    fms_dt _7_layer_3_pw_dw_out_1[layer_3_pw_depth][layer_4_pw_ifm_width] = {0};

#pragma HLS ARRAY_PARTITION variable = _7_layer_0_3x3_conv_out_1 complete dim = 1
#pragma HLS ARRAY_PARTITION variable = _7_layer_0_3x3_conv_out_1 complete dim = 2

#pragma HLS ARRAY_PARTITION variable = _7_layer_1_dw_out_1 complete dim = 1

#pragma HLS ARRAY_PARTITION variable = _7_layer_2_pw_out_1 complete dim = 1

#pragma HLS ARRAY_PARTITION variable = _7_layer_3_pw_dw_out_1 cyclic factor = layer_4_pw_parallelism_in/2 dim = 1

    //###########################################################
    // pipeline filling##########################################
    _7_stages_fill_channels_buffer(channels, channels_buffer_0, 0);
    //##########
    _7_stages_fill_channels_buffer(channels, channels_buffer_1, 2);
    _7_layer_0_3x3_conv(channels_buffer_0, weights_0,
                        _7_layer_0_3x3_conv_out_0);
    //##########
    _7_stages_fill_channels_buffer(channels, channels_buffer_0, 6);
    _7_layer_0_3x3_conv(channels_buffer_1, weights_0,
                        _7_layer_0_3x3_conv_out_1);
    _7_layer_1_pw_dw(_7_layer_0_3x3_conv_out_0, pw_weights_1, dw_weights_1,
                     _7_layer_1_dw_upper, _7_layer_1_dw_lower, _7_layer_1_dw_out_0, 0);
    //##########
    _7_stages_fill_channels_buffer(channels, channels_buffer_1, 10);
    _7_layer_0_3x3_conv(channels_buffer_0, weights_0,
                        _7_layer_0_3x3_conv_out_0);
    _7_layer_1_pw_dw(_7_layer_0_3x3_conv_out_1, pw_weights_1, dw_weights_1,
                     _7_layer_1_dw_upper, _7_layer_1_dw_lower, _7_layer_1_dw_out_1, 1);
    //##########
    _7_stages_fill_channels_buffer(channels, channels_buffer_0, 14);
    _7_layer_0_3x3_conv(channels_buffer_1, weights_0,
                        _7_layer_0_3x3_conv_out_1);
    _7_layer_1_pw_dw(_7_layer_0_3x3_conv_out_0, pw_weights_1, dw_weights_1,
                     _7_layer_1_dw_upper, _7_layer_1_dw_lower, _7_layer_1_dw_out_0, 1);
    _7_layer_2_pw(_7_layer_1_dw_out_1, pw_weights_2,
                  _7_layer_2_pw_out_1);
    //##########
    _7_stages_fill_channels_buffer(channels, channels_buffer_1, 18);
    _7_layer_0_3x3_conv(channels_buffer_0, weights_0,
                        _7_layer_0_3x3_conv_out_0);
    _7_layer_1_pw_dw(_7_layer_0_3x3_conv_out_1, pw_weights_1, dw_weights_1,
                     _7_layer_1_dw_upper, _7_layer_1_dw_lower, _7_layer_1_dw_out_1, 1);
    _7_layer_2_pw(_7_layer_1_dw_out_0, pw_weights_2,
                  _7_layer_2_pw_out_0);
    _7_layer_3_pw_dw(
        _7_layer_2_pw_out_1,
        pw_weights_3,
        dw_weights_3,
        _7_layer_3_dw_upper,
        _7_layer_3_dw_lower,
        _7_layer_3_pw_dw_out_1, 1);
    //##########
    int even_odd = 1;
    int h = 6;
main_pipeline_loop:
    for (; h < switch_point_fms_height; h++)
    {
        if (even_odd)
        {
            _7_stages_fill_channels_buffer(channels, channels_buffer_0, (h * _7_stages_layer_1_rows_at_once - 1) * layer_0_strides);
            _7_layer_0_3x3_conv(channels_buffer_1, weights_0,
                                _7_layer_0_3x3_conv_out_1);
            _7_layer_1_pw_dw(
                _7_layer_0_3x3_conv_out_0,
                pw_weights_1,
                dw_weights_1,
                _7_layer_1_dw_upper,
                _7_layer_1_dw_lower,
                _7_layer_1_dw_out_0,
                0);
            _7_layer_2_pw(_7_layer_1_dw_out_1, pw_weights_2,
                          _7_layer_2_pw_out_1);
            _7_layer_3_pw_dw(
                _7_layer_2_pw_out_0,
                pw_weights_3,
                dw_weights_3,
                _7_layer_3_dw_upper,
                _7_layer_3_dw_lower,
                _7_layer_3_pw_dw_out_0, 1);
            _7_layer_4_pw(_7_layer_3_pw_dw_out_1, pw_weights_4,
                          result, h - 6);
        }
        else
        {
            _7_stages_fill_channels_buffer(channels, channels_buffer_1, (h * _7_stages_layer_1_rows_at_once - 1) * layer_0_strides);
            _7_layer_0_3x3_conv(channels_buffer_0, weights_0,
                                _7_layer_0_3x3_conv_out_0);
            _7_layer_1_pw_dw(
                _7_layer_0_3x3_conv_out_1,
                pw_weights_1,
                dw_weights_1,
                _7_layer_1_dw_upper,
                _7_layer_1_dw_lower,
                _7_layer_1_dw_out_1,
                0);
            _7_layer_2_pw(_7_layer_1_dw_out_0, pw_weights_2,
                          _7_layer_2_pw_out_0);
            _7_layer_3_pw_dw(
                _7_layer_2_pw_out_1,
                pw_weights_3,
                dw_weights_3,
                _7_layer_3_dw_upper,
                _7_layer_3_dw_lower,
                _7_layer_3_pw_dw_out_1, 1);
            _7_layer_4_pw(_7_layer_3_pw_dw_out_0, pw_weights_4,
                          result, h - 6);
        }
        even_odd = 1 - even_odd;
    }
    //###########################################################
    // pipeline flushing##########################################
    _7_layer_4_pw(_7_layer_3_pw_dw_out_1, pw_weights_4,
                  result, switch_point_fms_height - 6);
    //##########
    _7_layer_3_pw_dw(
        _7_layer_2_pw_out_0,
        pw_weights_3,
        dw_weights_3,
        _7_layer_3_dw_upper,
        _7_layer_3_dw_lower,
        _7_layer_3_pw_dw_out_0, 1);
    _7_layer_4_pw(_7_layer_3_pw_dw_out_0, pw_weights_4,
                  result, switch_point_fms_height - 5);
    //##########
    _7_layer_2_pw(_7_layer_1_dw_out_1, pw_weights_2,
                  _7_layer_2_pw_out_1);
    _7_layer_3_pw_dw(
        _7_layer_2_pw_out_1,
        pw_weights_3,
        dw_weights_3,
        _7_layer_3_dw_upper,
        _7_layer_3_dw_lower,
        _7_layer_3_pw_dw_out_1, 1);
    _7_layer_4_pw(_7_layer_3_pw_dw_out_1, pw_weights_4,
                  result, switch_point_fms_height - 4);
    //##########
    _7_layer_1_pw_dw(
        _7_layer_0_3x3_conv_out_0,
        pw_weights_1,
        dw_weights_1,
        _7_layer_1_dw_upper,
        _7_layer_1_dw_lower,
        _7_layer_1_dw_out_0,
        0);
    _7_layer_2_pw(_7_layer_1_dw_out_0, pw_weights_2,
                  _7_layer_2_pw_out_0);
    _7_layer_3_pw_dw(
        _7_layer_2_pw_out_0,
        pw_weights_3,
        dw_weights_3,
        _7_layer_3_dw_upper,
        _7_layer_3_dw_lower,
        _7_layer_3_pw_dw_out_0, 1);
    _7_layer_4_pw(_7_layer_3_pw_dw_out_0, pw_weights_4,
                  result, switch_point_fms_height - 3);
    //##########
    _7_layer_0_3x3_conv(channels_buffer_1, weights_0,
                        _7_layer_0_3x3_conv_out_1);
    _7_layer_1_pw_dw(
        _7_layer_0_3x3_conv_out_1,
        pw_weights_1,
        dw_weights_1,
        _7_layer_1_dw_upper,
        _7_layer_1_dw_lower,
        _7_layer_1_dw_out_1,
        0);
    _7_layer_2_pw(_7_layer_1_dw_out_1, pw_weights_2,
                  _7_layer_2_pw_out_1);
    _7_layer_3_pw_dw(
        _7_layer_2_pw_out_1,
        pw_weights_3,
        dw_weights_3,
        _7_layer_3_dw_upper,
        _7_layer_3_dw_lower,
        _7_layer_3_pw_dw_out_1, 1);
    _7_layer_4_pw(_7_layer_3_pw_dw_out_1, pw_weights_4,
                  result, switch_point_fms_height - 2);
    //#########
    _7_stages_fill_channels_buffer(channels, channels_buffer_0, (switch_point_fms_height * _7_stages_layer_1_rows_at_once - 1) * layer_0_strides);
    _7_layer_0_3x3_conv(channels_buffer_0, weights_0,
                        _7_layer_0_3x3_conv_out_0);
    _7_layer_1_pw_dw(
        _7_layer_0_3x3_conv_out_0,
        pw_weights_1,
        dw_weights_1,
        _7_layer_1_dw_upper,
        _7_layer_1_dw_lower,
        _7_layer_1_dw_out_0,
        0);
    // padding bottom
    for (int d = 0; d < layer_2_pw_depth; d++)
    {
        for (int w = 0; w < layer_2_pw_ifm_width; w++)
        {
            _7_layer_1_dw_out_0[d][_7_stages_layer_1_rows_at_once - 1][w] = 0;
        }
    }
    _7_layer_2_pw(_7_layer_1_dw_out_0, pw_weights_2,
                  _7_layer_2_pw_out_0);
    _7_layer_3_pw_dw(
        _7_layer_2_pw_out_0,
        pw_weights_3,
        dw_weights_3,
        _7_layer_3_dw_upper,
        _7_layer_3_dw_lower,
        _7_layer_3_pw_dw_out_0, 1);
    _7_layer_4_pw(_7_layer_3_pw_dw_out_0, pw_weights_4,
                  result, switch_point_fms_height - 1);
}
