#include "quantization_and_biases.h"
#include "../utils/utils.h"

#include "../layers/headers/layers_glue.h"

using namespace std;

#include "../model/model_glue.h"

#include "../pipeline/headers/pipeline_glue.h"

#include "../cnn_functions_v1.h"

#include "dw_weights.h"
#include "quantization_and_biases.h"
#include <iostream>
#include <math.h>


void _7_layer_6_pw(
    fms_dt channels_buffer[layer_6_pw_depth][layer_6_pw_ifm_width],
    weights_dt weights[layer_6_pw_num_fils][layer_6_pw_depth],
    fms_dt result[layer_7_pw_depth][layer_7_pw_ifm_width])
{

#pragma HLS INLINE off

    // rows for next DW
    for (int o_o_d = 0;
         o_o_d < layer_6_pw_num_fils / layer_6_pw_parallelism_out;
         o_o_d++)
    {
        int o_o_d_offset = o_o_d * layer_6_pw_parallelism_out;
        // filters loop
    layer_6_pw_pipeline:
        for (int w = 0; w < layer_6_pw_ifm_width; w++)
        {
#pragma HLS PIPELINE
        // FMs width loop
        layer_6_pw_loops:
            for (int o_d = 0;
                 o_d < layer_6_pw_parallelism_out; o_d++)
            {
#pragma HLS UNROLL
                // parallelized filters loop
                pss_dt tmp = 0;
                for (int d = 0; d < layer_6_pw_parallelism_in; d++)
                {
#pragma HLS UNROLL
                    // parallelized depth loop
                    tmp += ((fms_dt)channels_buffer[d][w]) * weights[o_d][d];
                }
                fms_dt scaled_val = (fms_dt)((((ap_fixed<17, 12>)tmp) - zero_point_dw) * ratio_dw_pss_to_fms);
                if (scaled_val > 0)
                {
                    result[o_o_d_offset + o_d] = scaled_val; // must be fully unrolled in the IFMs depth dimension
                }
            }
        }
    }
}

void _7_layer_7_pw(
    fms_dt channels_buffer[layer_7_pw_depth][layer_7_pw_ifm_width],
    weights_dt weights[layer_7_pw_num_fils][layer_7_pw_depth],
    fms_dt result[max_fms_size], int starting_h)
{

#pragma HLS INLINE off

    // rows for next DW
    for (int o_o_d = 0;
         o_o_d < layer_6_pw_num_fils / layer_7_pw_parallelism_out;
         o_o_d++)
    {
        int o_o_d_offset = o_o_d * layer_7_pw_parallelism_out;
        // filters loop
    layer_7_pw_pipeline:
        for (int w = 0; w < layer_7_pw_ifm_width; w++)
        {
#pragma HLS PIPELINE
        // FMs width loop
        layer_7_pw_loops:
            for (int o_d = 0;
                 o_d < layer_7_pw_parallelism_out; o_d++)
            {
#pragma HLS UNROLL
                // parallelized filters loop
                pss_dt tmp = 0;
                int offset_in_result = (o_o_d * layer_7_pw_parallelism_out + o_d) * switch_point_fms_height * switch_point_fms_width + starting_h * switch_point_fms_width + w;
                for (int d = 0; d < layer_7_pw_parallelism_in; d++)
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

void cnn_pipeline_7_mob_v2(
    fms_dt channels[input_image_depth][input_image_height][input_image_width],
    fms_dt result[max_fms_size])
{
#pragma HLS INLINE off

#pragma HLS ARRAY_PARTITION variable = channels type = complete dim = 1

    layer_0_weights_dt weights_0[layer_0_num_fils][layer_0_depth][layer_0_filter_size][layer_0_filter_size];

    fill_layer_0_weights(weights_0);

    //#########################even###############################
    fms_dt channels_buffer_0[input_image_depth][layer_0_filter_size + (_7_stages_layer_0_rows_at_once - 1) * layer_0_strides][input_image_width];
#pragma HLS ARRAY_PARTITION variable = channels_buffer_0 complete dim = 1
#pragma HLS ARRAY_PARTITION variable = channels_buffer_0 complete dim = 2
    fms_dt _6_layer_0_3x3_conv_out_0[layer_1_pw_depth][_7_stages_layer_0_rows_at_once][layer_1_pw_ifm_width] =
        {0};

    fms_dt _6_layer_1_pw_out_0[layer_2_dw_depth][layer_2_dw_ifm_width] = {0};

    //##############
    fms_dt _6_layer_2_dw_upper[layer_2_dw_depth][layer_2_dw_filter_size - layer_2_dw_strides][layer_2_dw_ifm_width];
#pragma HLS ARRAY_PARTITION variable = _6_layer_2_dw_upper complete dim = 3

    fms_dt _6_layer_2_dw_lower[layer_2_dw_depth][_7_stages_layer_2_rows_at_once][layer_2_dw_ifm_width];
#pragma HLS ARRAY_PARTITION variable = _6_layer_2_dw_lower complete dim = 3

    fms_dt _6_layer_2_dw_out_0[layer_3_pw_depth][_7_stages_layer_2_rows_at_once][layer_3_pw_ifm_width] = {0};
#pragma HLS ARRAY_PARTITION variable = _6_layer_2_dw_out_0 complete dim = 1
    //##############

    fms_dt _6_layer_3_pw_out_0[layer_4_pw_depth][_7_stages_layer_3_rows_at_once][layer_5_dw_ifm_width] = {0};

    fms_dt _6_layer_5_dw_upper[layer_5_dw_depth][layer_5_dw_ifm_width];
#pragma HLS ARRAY_PARTITION variable = _6_layer_5_dw_upper cyclic factor = layer_4_pw_parallelism_out dim = 1
#pragma HLS ARRAY_PARTITION variable = _6_layer_5_dw_upper cyclic factor = 6 dim = 2

    fms_dt _6_layer_5_dw_lower[layer_5_dw_depth][layer_5_dw_strides][layer_5_dw_ifm_width];
#pragma HLS ARRAY_PARTITION variable = _6_layer_5_dw_lower cyclic factor = 2 dim = 1
#pragma HLS ARRAY_PARTITION variable = _6_layer_5_dw_lower complete dim = 2
#pragma HLS ARRAY_PARTITION variable = _6_layer_5_dw_lower cyclic factor = 12 dim = 3

    fms_dt _6_layer_4_5_pw_dw_out_0[layer_4_pw_depth][layer_4_pw_ifm_width] = {0};

#pragma HLS ARRAY_PARTITION variable = _6_layer_0_3x3_conv_out_0 complete dim = 1
#pragma HLS ARRAY_PARTITION variable = _6_layer_0_3x3_conv_out_0 complete dim = 2

#pragma HLS ARRAY_PARTITION variable = _6_layer_3_pw_out_0 complete dim = 1

#pragma HLS ARRAY_PARTITION variable = _6_layer_4_5_pw_dw_out_0 cyclic factor = layer_4_pw_parallelism_in/2 dim = 1

    fms_dt _6_layer_6_pw_out_0[layer_7_pw_depth][layer_7_pw_ifm_width] = {0};
#pragma HLS ARRAY_PARTITION variable = _6_layer_6_pw_out_0 complete dim = 1
    //###########################################################

    //#########################odd###############################
    fms_dt channels_buffer_1[input_image_depth][layer_0_filter_size + (_7_stages_layer_2_rows_at_once - 1) * layer_0_strides][input_image_width];
#pragma HLS ARRAY_PARTITION variable = channels_buffer_1 complete dim = 1
#pragma HLS ARRAY_PARTITION variable = channels_buffer_1 complete dim = 2

    fms_dt _6_layer_0_3x3_conv_out_1[layer_1_pw_depth][_7_stages_layer_0_rows_at_once][layer_1_pw_ifm_width] =
        {0};

    fms_dt _6_layer_1_pw_out_1[layer_2_dw_depth][layer_2_dw_ifm_width] = {0};

    //##############

    fms_dt _6_layer_2_dw_out_1[layer_3_pw_depth][_7_stages_layer_2_rows_at_once][layer_3_pw_ifm_width] = {0};
    //##############
    fms_dt _6_layer_3_pw_out_1[layer_4_pw_depth][_7_stages_layer_3_rows_at_once][layer_5_dw_ifm_width] = {0};

    fms_dt _6_layer_4_5_pw_dw_out_1[layer_4_pw_depth][layer_4_pw_ifm_width] = {0};

#pragma HLS ARRAY_PARTITION variable = _6_layer_0_3x3_conv_out_1 complete dim = 1
#pragma HLS ARRAY_PARTITION variable = _6_layer_0_3x3_conv_out_1 complete dim = 2

#pragma HLS ARRAY_PARTITION variable = _6_layer_2_dw_out_1 complete dim = 1

#pragma HLS ARRAY_PARTITION variable = _6_layer_3_pw_out_1 complete dim = 1

#pragma HLS ARRAY_PARTITION variable = _6_layer_4_5_pw_dw_out_1 cyclic factor = layer_4_pw_parallelism_in/2 dim = 1

    fms_dt _6_layer_6_pw_out_1[layer_7_pw_depth][layer_7_pw_ifm_width] = {0};
#pragma HLS ARRAY_PARTITION variable = _6_layer_6_pw_out_1 complete dim = 1

    //###########################################################
    // pipeline filling##########################################
    _7_stages_fill_channels_buffer(channels, channels_buffer_0, 0);
    //##########
    _7_stages_fill_channels_buffer(channels, channels_buffer_1, 2);
    _6_layer_0_3x3_conv(channels_buffer_0, weights_0,
                        _6_layer_0_3x3_conv_out_0);
    //##########
    _7_stages_fill_channels_buffer(channels, channels_buffer_0, 6);
    _6_layer_0_3x3_conv(channels_buffer_1, weights_0,
                        _6_layer_0_3x3_conv_out_1);
    _6_layer_2_dw(_6_layer_0_3x3_conv_out_0, dw_weights_2,
                     _6_layer_2_dw_upper, _6_layer_2_dw_lower, _6_layer_2_dw_out_0, 0);
    //##########
    _7_stages_fill_channels_buffer(channels, channels_buffer_1, 10);
    _6_layer_0_3x3_conv(channels_buffer_0, weights_0,
                        _6_layer_0_3x3_conv_out_0);
    _6_layer_2_dw(_6_layer_0_3x3_conv_out_1, dw_weights_2,
                     _6_layer_2_dw_upper, _6_layer_2_dw_lower, _6_layer_2_dw_out_1, 1);
    //##########
    _7_stages_fill_channels_buffer(channels, channels_buffer_0, 14);
    _6_layer_0_3x3_conv(channels_buffer_1, weights_0,
                        _6_layer_0_3x3_conv_out_1);
    _6_layer_2_dw(_6_layer_0_3x3_conv_out_0, dw_weights_2,
                     _6_layer_2_dw_upper, _6_layer_2_dw_lower, _6_layer_2_dw_out_0, 1);
    _6_layer_3_pw(_6_layer_2_dw_out_1, pw_weights_3,
                  _6_layer_3_pw_out_1);
    //##########
    _7_stages_fill_channels_buffer(channels, channels_buffer_1, 18);
    _6_layer_0_3x3_conv(channels_buffer_0, weights_0,
                        _6_layer_0_3x3_conv_out_0);
    _6_layer_2_dw(_6_layer_0_3x3_conv_out_1, dw_weights_2,
                     _6_layer_2_dw_upper, _6_layer_2_dw_lower, _6_layer_2_dw_out_1, 1);
    _6_layer_3_pw(_6_layer_2_dw_out_0, pw_weights_3,
                  _6_layer_3_pw_out_0);
    _6_layer_4_pw_5_dw(
        _6_layer_3_pw_out_1,
        pw_weights_4,
        dw_weights_5,
        _6_layer_5_dw_upper,
        _6_layer_5_dw_lower,
        _6_layer_4_5_pw_dw_out_1, 1);
    //##########
    _7_stages_fill_channels_buffer(channels, channels_buffer_0, 22);
    _6_layer_0_3x3_conv(channels_buffer_1, weights_0,
                        _6_layer_0_3x3_conv_out_1);
    _6_layer_2_dw(_6_layer_0_3x3_conv_out_0, dw_weights_2,
                     _6_layer_2_dw_upper, _6_layer_2_dw_lower, _6_layer_2_dw_out_0, 1);
    _6_layer_3_pw(_6_layer_2_dw_out_1, pw_weights_3,
                  _6_layer_3_pw_out_1);
    _6_layer_4_pw_5_dw(
        _6_layer_3_pw_out_0,
        pw_weights_4,
        dw_weights_5,
        _6_layer_5_dw_upper,
        _6_layer_5_dw_lower,
        _6_layer_4_5_pw_dw_out_0, 1);
     _7_layer_6_pw(_6_layer_4_5_pw_dw_out_1, pw_weights_6,
                          _6_layer_6_pw_out_1);
    int even_odd = 0;
    int h = 7;
main_pipeline_loop:
    for (; h < switch_point_fms_height; h++)
    {
        if (even_odd)
        {
            _7_stages_fill_channels_buffer(channels, channels_buffer_0, (h * _7_stages_layer_0_rows_at_once - 1) * layer_0_strides);
            _6_layer_0_3x3_conv(channels_buffer_1, weights_0,
                                _6_layer_0_3x3_conv_out_1);
            _6_layer_2_dw(
                _6_layer_0_3x3_conv_out_0,
                dw_weights_2,
                _6_layer_2_dw_upper,
                _6_layer_2_dw_lower,
                _6_layer_2_dw_out_0,
                0);
            _6_layer_3_pw(_6_layer_2_dw_out_1, pw_weights_3,
                          _6_layer_3_pw_out_1);
            _6_layer_4_pw_5_dw(
                _6_layer_3_pw_out_0,
                pw_weights_4,
                dw_weights_5,
                _6_layer_5_dw_upper,
                _6_layer_5_dw_lower,
                _6_layer_4_5_pw_dw_out_0, 1);
            _7_layer_6_pw(_6_layer_4_5_pw_dw_out_1, pw_weights_6,
                          _6_layer_6_pw_out_1);
            _7_layer_7_pw(_6_layer_6_pw_out_0, pw_weights_7,
                          result, h - 7);
        }
        else
        {
            _7_stages_fill_channels_buffer(channels, channels_buffer_1, (h * _7_stages_layer_0_rows_at_once - 1) * layer_0_strides);
            _6_layer_0_3x3_conv(channels_buffer_0, weights_0,
                                _6_layer_0_3x3_conv_out_0);
            _6_layer_2_dw(
                _6_layer_0_3x3_conv_out_1,
                dw_weights_1,
                _6_layer_2_dw_upper,
                _6_layer_2_dw_lower,
                _6_layer_2_dw_out_1,
                0);
            _6_layer_3_pw(_6_layer_2_dw_out_0, pw_weights_3,
                          _6_layer_3_pw_out_0);
            _6_layer_4_pw_5_dw(
                _6_layer_3_pw_out_1,
                pw_weights_4,
                dw_weights_5,
                _6_layer_5_dw_upper,
                _6_layer_5_dw_lower,
                _6_layer_4_5_pw_dw_out_1, 1);
            _7_layer_6_pw(_6_layer_4_5_pw_dw_out_0, pw_weights_6,
                          _6_layer_6_pw_out_0);
            _7_layer_7_pw(_6_layer_6_pw_out_1, pw_weights_7,
                          result, h - 7);
        }
        even_odd = 1 - even_odd;
    }
    //###########################################################
    // pipeline flushing##########################################
    _7_layer_7_pw(_6_layer_6_pw_out_1, pw_weights_7,
                          result, switch_point_fms_height - 7);
    //##########
    _7_layer_6_pw(_6_layer_4_5_pw_dw_out_0, pw_weights_6,
                  _6_layer_6_pw_out_0);
    _7_layer_7_pw(_6_layer_6_pw_out_0, pw_weights_7,
                          result, switch_point_fms_height - 6);
    //##########
    _6_layer_4_pw_5_dw(
        _6_layer_3_pw_out_1,
        pw_weights_4,
        dw_weights_5,
        _6_layer_5_dw_upper,
        _6_layer_5_dw_lower,
        _6_layer_4_5_pw_dw_out_1, 1);
    _7_layer_6_pw(_6_layer_4_5_pw_dw_out_1, pw_weights_6,
                  _6_layer_6_pw_out_1);
    _7_layer_7_pw(_6_layer_6_pw_out_1, pw_weights_7,
                          result, switch_point_fms_height - 5);
    //##########
    _6_layer_3_pw(_6_layer_2_dw_out_0, pw_weights_3,
                  _6_layer_3_pw_out_0);
    _6_layer_4_pw_5_dw(
        _6_layer_3_pw_out_0,
        pw_weights_4,
        dw_weights_5,
        _6_layer_5_dw_upper,
        _6_layer_5_dw_lower,
        _6_layer_4_5_pw_dw_out_0, 1);
    _7_layer_6_pw(_6_layer_4_5_pw_dw_out_0, pw_weights_6,
                  _6_layer_6_pw_out_0);
    _7_layer_7_pw(_6_layer_6_pw_out_0, pw_weights_7,
                          result, switch_point_fms_height - 4);
    //##########
    _6_layer_2_dw(
        _6_layer_0_3x3_conv_out_1,
        dw_weights_2,
        _6_layer_2_dw_upper,
        _6_layer_2_dw_lower,
        _6_layer_2_dw_out_1,
        0);
    __6_layer_3_pw(_6_layer_2_dw_out_1, pw_weights_3,
                  _6_layer_3_pw_out_1);
    _6_layer_4_pw_5_dw(
        _6_layer_3_pw_out_1,
        pw_weights_4,
        dw_weights_5,
        _6_layer_5_dw_upper,
        _6_layer_5_dw_lower,
        _6_layer_4_5_pw_dw_out_1, 1);
    _7_layer_6_pw(_6_layer_4_5_pw_dw_out_1, pw_weights_6,
                  _6_layer_6_pw_out_1);
    _7_layer_7_pw(_6_layer_6_pw_out_1, pw_weights_4
                          result, switch_point_fms_height - 3);
    //##########
    _6_layer_0_3x3_conv(channels_buffer_0, weights_0,
                        _6_layer_0_3x3_conv_out_0);
    _6_layer_2_dw(
        _6_layer_0_3x3_conv_out_0,
        dw_weights_2,
        _6_layer_2_dw_upper,
        _6_layer_2_dw_lower,
        _6_layer_2_dw_out_0,
        0);
    __6_layer_3_pw(_6_layer_2_dw_out_0, pw_weights_3,
                  _6_layer_3_pw_out_0);
    _6_layer_4_pw_5_dw(
        _6_layer_3_pw_out_0,
        pw_weights_4,
        dw_weights_5,
        _6_layer_5_dw_upper,
        _6_layer_5_dw_lower,
        _6_layer_4_5_pw_dw_out_0, 1);
    _7_layer_6_pw(_6_layer_4_5_pw_dw_out_0, pw_weights_6,
                  _6_layer_6_pw_out_0);
    _7_layer_7_pw(_6_layer_6_pw_out_0, pw_weights_7,
                          result, switch_point_fms_height - 2);
    //#########
    _7_stages_fill_channels_buffer(channels, channels_buffer_0, (switch_point_fms_height * _7_stages_layer_0_rows_at_once - 1) * layer_0_strides);
    _6_layer_0_3x3_conv(channels_buffer_0, weights_0,
                        _6_layer_0_3x3_conv_out_0);
    _6_layer_2_dw(
        _6_layer_0_3x3_conv_out_0,
        dw_weights_2,
        _6_layer_2_dw_upper,
        _6_layer_2_dw_lower,
        _6_layer_2_dw_out_0,
        0);
    // padding bottom
    for (int d = 0; d < layer_3_pw_depth; d++)
    {
        for (int w = 0; w < layer_3_pw_ifm_width; w++)
        {
            _6_layer_2_dw_out_0[d][_7_stages_layer_0_rows_at_once - 1][w] = 0;
        }
    }
    _6_layer_3_pw(_6_layer_2_dw_out_0, pw_weights_3,
                  _6_layer_3_pw_out_0);
    _6_layer_4_pw_5_dw(
        _6_layer_3_pw_out_0,
        pw_weights_4,
        dw_weights_5,
        _6_layer_5_dw_upper,
        _6_layer_5_dw_lower,
        _6_layer_4_5_pw_dw_out_0, 1);
    _7_layer_6_pw(_6_layer_4_5_pw_dw_out_0, pw_weights_6,
                  _6_layer_6_pw_out_0);
    _7_layer_7_pw(_6_layer_6_pw_out_0, pw_weights_7,
                          result, switch_point_fms_height - 1);
}
