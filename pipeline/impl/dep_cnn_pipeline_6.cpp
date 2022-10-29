#include "./headers/utils.h"


void _6_layer_3_pw_dw(
    fms_dt channels_buffer[layer_3_pw_depth][layer_3_dw_strides][layer_3_dw_ifm_width],
    weights_dt weights[layer_3_pw_num_fils][layer_3_pw_depth],
    dw_weights_dt dw_weights[layer_3_dw_depth][layer_3_dw_filter_size][layer_3_dw_filter_size],
    fms_dt upper[layer_3_dw_depth][layer_3_dw_ifm_width],
    fms_dt lower[layer_3_dw_depth][layer_3_dw_strides][layer_3_dw_ifm_width],
	fms_dt result[max_fms_size], int starting_h, int active_row) // active_row is depricated, just pass strides of that layer - 1
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
                intermediate_pw_results[o_o_d_offset + d][h][0] = 0;
                for (int w = 1; w < layer_3_dw_filter_size; w++)
                {
#pragma HLS UNROLL
                    intermediate_pw_results[o_o_d_offset + d][h][w] = upper[o_o_d_offset + d][h][w];
                }
            }
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
                                intermediate_pw_results[o_d][row + layer_3_dw_filter_size - layer_3_dw_strides][w + 1] = scaled_val;
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
					int offset_in_result = (o_o_d * layer_3_pw_parallelism_out + o_d) * switch_point_fms_height * switch_point_fms_width + 
                    starting_h * switch_point_fms_width + w - layer_3_dw_padding_left;
                    if (scaled_val > 0)
                    {
                       result[offset_in_result] = scaled_val;
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

void mobilenet_v2_pipeline_6(
    fms_dt channels[input_image_depth][input_image_height][input_image_width],
    fms_dt result[max_fms_size])
{
#pragma HLS INLINE off

#pragma HLS ARRAY_PARTITION variable = channels type = complete dim = 1

    dw_weights_dt dw_weights_1[layer_1_dw_depth][layer_1_dw_filter_size][layer_1_dw_filter_size];
    dw_weights_dt dw_weights_3[layer_3_dw_depth][layer_3_dw_filter_size][layer_3_dw_filter_size];

#pragma HLS ARRAY_PARTITION variable = dw_weights_1 type = complete dim = 1

    layer_0_weights_dt weights_0[layer_0_num_fils][layer_0_depth][3][3];

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
    fms_dt channels_buffer_0[input_image_depth][layer_0_filter_dim + (_7_stages_layer_1_rows_at_once - 1) * layer_0_strides][input_image_width];
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


#pragma HLS ARRAY_PARTITION variable = _7_layer_0_3x3_conv_out_0 complete dim = 1
#pragma HLS ARRAY_PARTITION variable = _7_layer_0_3x3_conv_out_0 complete dim = 2

#pragma HLS ARRAY_PARTITION variable = _7_layer_2_pw_out_0 complete dim = 1

    //###########################################################

    //#########################odd###############################
    fms_dt channels_buffer_1[input_image_depth][layer_0_filter_dim + (_7_stages_layer_1_rows_at_once - 1) * layer_0_strides][input_image_width];
#pragma HLS ARRAY_PARTITION variable = channels_buffer_1 complete dim = 1
#pragma HLS ARRAY_PARTITION variable = channels_buffer_1 complete dim = 2

    fms_dt _7_layer_0_3x3_conv_out_1[layer_1_pw_depth][_7_stages_layer_1_rows_at_once][layer_1_pw_ifm_width] =
        {0};

    fms_dt _7_layer_1_pw_out_1[layer_1_dw_depth][layer_1_dw_ifm_width] = {0};

    //##############

    fms_dt _7_layer_1_dw_out_1[layer_2_pw_depth][_7_stages_layer_1_rows_at_once][layer_2_pw_ifm_width] = {0};
    //##############
    fms_dt _7_layer_2_pw_out_1[layer_3_pw_depth][_7_stages_layer_1_rows_at_once][layer_3_dw_ifm_width] = {0};


#pragma HLS ARRAY_PARTITION variable = _7_layer_0_3x3_conv_out_1 complete dim = 1
#pragma HLS ARRAY_PARTITION variable = _7_layer_0_3x3_conv_out_1 complete dim = 2

#pragma HLS ARRAY_PARTITION variable = _7_layer_1_dw_out_1 complete dim = 1

#pragma HLS ARRAY_PARTITION variable = _7_layer_2_pw_out_1 complete dim = 1


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

    int even_odd = 0;
    int h = 5;
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
            _6_layer_3_pw_dw(
                _7_layer_2_pw_out_0,
                pw_weights_3,
                dw_weights_3,
                _7_layer_3_dw_upper,
                _7_layer_3_dw_lower,
                result, h-5, 1);
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
            _6_layer_3_pw_dw(
                _7_layer_2_pw_out_1,
                pw_weights_3,
                dw_weights_3,
                _7_layer_3_dw_upper,
                _7_layer_3_dw_lower,
                result, h-5, 1);
        }
        even_odd = 1 - even_odd;
    }
    //###########################################################
    // pipeline flushing##########################################
    _6_layer_3_pw_dw(
        _7_layer_2_pw_out_0,
        pw_weights_3,
        dw_weights_3,
        _7_layer_3_dw_upper,
        _7_layer_3_dw_lower,
        result, switch_point_fms_height-5, 1);
    //##########
    _7_layer_2_pw(_7_layer_1_dw_out_1, pw_weights_2,
                  _7_layer_2_pw_out_1);
    _6_layer_3_pw_dw(
        _7_layer_2_pw_out_1,
        pw_weights_3,
        dw_weights_3,
        _7_layer_3_dw_upper,
        _7_layer_3_dw_lower,
        result, switch_point_fms_height-4, 1);
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
    _6_layer_3_pw_dw(
        _7_layer_2_pw_out_0,
        pw_weights_3,
        dw_weights_3,
        _7_layer_3_dw_upper,
        _7_layer_3_dw_lower,
        result, switch_point_fms_height-3, 1);
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
    _6_layer_3_pw_dw(
        _7_layer_2_pw_out_1,
        pw_weights_3,
        dw_weights_3,
        _7_layer_3_dw_upper,
        _7_layer_3_dw_lower,
        result, switch_point_fms_height-2, 1);
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
    //padding bottom
    for (int d = 0; d < layer_2_pw_depth; d++)
    {
        for (int w = 0; w < layer_2_pw_ifm_width; w++)
        {
            _7_layer_1_dw_out_0[d][_7_stages_layer_1_rows_at_once - 1][w] = 0;
        }
    }
    _7_layer_2_pw(_7_layer_1_dw_out_0, pw_weights_2,
                  _7_layer_2_pw_out_0);
    _6_layer_3_pw_dw(
        _7_layer_2_pw_out_0,
        pw_weights_3,
        dw_weights_3,
        _7_layer_3_dw_upper,
        _7_layer_3_dw_lower,
        result, switch_point_fms_height-1, 1);
}
