#include "pipeline_main.h"

#if FIRST_PART_IMPLEMENTATION == PIPELINED_ENGINES_MODE && FIBHA_VERSION == 2 && !ONLY_SEML

using namespace pipelined_engines;

void padd_lr_dw_channels_tile(fms_dt dw_channels_tile[DW_TILE_DEPTH][DW_BUFFER_HEIGHT][DW_BUFFER_WIDTH],
                              fms_dt dw_channels_tile_copy[DW_TILE_DEPTH][DW_BUFFER_HEIGHT][DW_BUFFER_WIDTH],
                              layer_specs layer_specs_struct)
{
    const fms_dt current_layer_ifms_zero_point = layer_specs_struct.layer_ifms_zero_point;
    const int padding_left = layer_specs_struct.padding_left;
    const int layer_ifms_width = layer_specs_struct.layer_ifm_width;

    for (int d = 0; d < DW_TILE_DEPTH; d++)
    {
        for (int h = 0; h < DW_BUFFER_HEIGHT; h++)
        {
            for (int w = 0; w < MAX_DW_PADDING_IN_PIPE; w++)
            {
                if (w < padding_left)
                {
                    dw_channels_tile[d][h][w] = current_layer_ifms_zero_point;
                    dw_channels_tile_copy[d][h][w] = current_layer_ifms_zero_point;
                }
                dw_channels_tile[d][h][layer_ifms_width + padding_left + w] = current_layer_ifms_zero_point;
                dw_channels_tile_copy[d][h][layer_ifms_width + padding_left + w] = current_layer_ifms_zero_point;
            }
        }
    }
}

void padd_left_dw_channels_tile(fms_dt dw_channels_tile[DW_TILE_DEPTH][DW_BUFFER_HEIGHT][DW_BUFFER_WIDTH],
                                fms_dt dw_channels_tile_copy[DW_TILE_DEPTH][DW_BUFFER_HEIGHT][DW_BUFFER_WIDTH],
                                layer_specs layer_specs_struct)
{
    const fms_dt current_layer_ifms_zero_point = layer_specs_struct.layer_ifms_zero_point;
    const int padding_left = layer_specs_struct.padding_left;
    const int layer_ifms_width = layer_specs_struct.layer_ifm_width;

    for (int d = 0; d < DW_TILE_DEPTH; d++)
    {
        for (int h = 0; h < DW_BUFFER_HEIGHT; h++)
        {
            for (int w = 0; w < MAX_DW_PADDING_IN_PIPE; w++)
            {
                if (w < padding_left)
                {
                    dw_channels_tile[d][h][w] = current_layer_ifms_zero_point;
                    dw_channels_tile_copy[d][h][w] = current_layer_ifms_zero_point;
                }
            }
        }
    }
}

void padd_top_dw_channels_tile(fms_dt dw_channels_tile[DW_TILE_DEPTH][DW_BUFFER_HEIGHT][DW_BUFFER_WIDTH],
                               fms_dt dw_channels_tile_copy[DW_TILE_DEPTH][DW_BUFFER_HEIGHT][DW_BUFFER_WIDTH],
                               layer_specs layer_specs_struct)
{
    const fms_dt current_layer_ifms_zero_point = layer_specs_struct.layer_ifms_zero_point;
    const int layer_ifms_width = layer_specs_struct.layer_ifm_width;
    const int padding_top = layer_specs_struct.padding_top;

    for (int d = 0; d < DW_TILE_DEPTH; d++)
    {
        for (int h = 0; h < padding_top; h++)
        {
            for (int w = 0; w < DW_BUFFER_WIDTH; w++)
            {
                dw_channels_tile[d][h][w] = current_layer_ifms_zero_point;
                dw_channels_tile_copy[d][h][w] = current_layer_ifms_zero_point;
            }
        }
    }
}

void padd_bottom_dw_channels_tile(fms_dt dw_channels_tile[DW_TILE_DEPTH][DW_BUFFER_HEIGHT][DW_BUFFER_WIDTH],
                                  fms_dt dw_channels_tile_copy[DW_TILE_DEPTH][DW_BUFFER_HEIGHT][DW_BUFFER_WIDTH],
                                  layer_specs layer_specs_struct)
{
    const fms_dt current_layer_ifms_zero_point = layer_specs_struct.layer_ifms_zero_point;
    const int layer_ifms_width = layer_specs_struct.layer_ifm_width;
    const int padding_bottom = layer_specs_struct.padding_bottom;

    for (int d = 0; d < DW_TILE_DEPTH; d++)
    {
        for (int h = 0; h < MAX_DW_PADDING_IN_PIPE; h++)
        {
            for (int w = 0; w < DW_BUFFER_WIDTH; w++)
            {
                dw_channels_tile[d][DW_BUFFER_HEIGHT - padding_bottom + h][w] = current_layer_ifms_zero_point;
                dw_channels_tile_copy[d][DW_BUFFER_HEIGHT - padding_bottom + h][w] = current_layer_ifms_zero_point;
            }
        }
    }
}

void write_pipe_seml_communication_buffer(
    fms_dt pipe_seml_communication_buffer[MAX_PW_BUFFER_DEPTH][PW_BUFFER_HEIGHT][MAX_PW_BUFFER_WIDTH],
    fms_dt result[MAX_FMS_BUFFER_DEPTH][MIN_FMS_HEIGHT][MIN_FMS_WIDTH],
    const int starting_h,
    const int offset_h_in_communication_buffer,
    const layer_specs layer_specs_struct)
{
#pragma HLS INLINE off

    if (starting_h >= 0)
    {
        const int layer_depth = layer_specs_struct.layer_depth;
        const int layer_ifms_width = layer_specs_struct.layer_ifm_width;
        const int layer_ifms_height = layer_specs_struct.layer_ifm_height;
        const int num_of_tiles_h = layer_specs_struct.layer_num_of_ifm_tiles_h;
        const int num_of_tiles_w = layer_specs_struct.layer_num_of_ifm_tiles_w;
        const int num_of_tiles_hw = num_of_tiles_h * num_of_tiles_w;

        const int initial_h_offset = starting_h % CHANNELS_TILE_HEIGHT;

        for (int h = 0; h < PW_BUFFER_HEIGHT; h++)
        {
            if (offset_h_in_communication_buffer + h >= PW_BUFFER_HEIGHT || h + starting_h >= layer_ifms_height)
            {
                break;
            }
            int h_in_tile = initial_h_offset + h >= CHANNELS_TILE_HEIGHT ? initial_h_offset + h - CHANNELS_TILE_HEIGHT : initial_h_offset + h;
            int tile_offset_h = ((h + starting_h) / CHANNELS_TILE_HEIGHT) * num_of_tiles_w;
            for (int w = 0; w < layer_ifms_width; w++)
            {
                int w_in_tile = w % CHANNELS_TILE_WIDTH;
                int h_w_offset = tile_offset_h + w / CHANNELS_TILE_WIDTH;
                for (int d = 0; d < layer_depth; d++)
                {
#pragma HLS PIPELINE
                    int tile_index = d * num_of_tiles_hw + h_w_offset;
                    result[tile_index][h_in_tile + offset_h_in_communication_buffer][w_in_tile] =
                        pipe_seml_communication_buffer[d][h + offset_h_in_communication_buffer][w];
                }
            }
        }
    }
}

void fill_first_dw_layer_weights(weights_dt dw_layer_weights[layer_2_dw_num_fils][layer_2_dw_filter_dim * layer_2_dw_filter_dim])
{
    for (int f = 0; f < layer_2_dw_num_fils; f++)
    {
        for (int hw = 0; hw < layer_2_dw_filter_dim * layer_2_dw_filter_dim; hw++)
        {
            dw_layer_weights[f][hw] = pipe_dw_weights_3x3[f][hw];
        }
    }
}

void fill_first_conv_layers_quantization_params(fms_quantization_scheme first_layer_quantization_params[], const int starting_offset)
{
    for (int f = 0; f < first_conv_layer_num_fils; f++)
    {
        fms_quantization_scheme normalization;
        if (starting_offset == 0)
        {
            normalization.ofm_scale = first_conv_layer_specs.layer_ofms_scale;
            normalization.ofm_zero_point = first_conv_layer_specs.layer_ofms_zero_point;
        }
        else
        {
            normalization.ofm_scale = layer_2_dw_specs.layer_ofms_scale;
            normalization.ofm_zero_point = layer_2_dw_specs.layer_ofms_zero_point;
        }

        normalization.fused_scales =
            pipe_fused_scales[f + starting_offset];
        normalization.fused_scales_log_2_shift =
            pipe_fused_scales_log_2_shifts[f + +starting_offset];
        normalization.relu_6_fused_scale =
            pipe_fused_scales_log_2_shifts[f + +starting_offset];
        normalization.fused_zero_point =
            pipe_fused_zero_points[f + starting_offset];
        normalization.layer_0_relu_6_fused_scale =
            pipe_relu_6_fused_scales[f + starting_offset];

        first_layer_quantization_params[f] = normalization;
    }
}

void padd_top_conv_dw_communication_buffer_inter(fms_dt conv_dw_communication_buffer_inter[first_conv_layer_num_fils][layer_2_dw_filter_dim]
                                                                                          [layer_2_dw_ifm_width])
{
    for (int d = 0; d < first_conv_layer_num_fils; d++)
    {
        for (int h = 0; h < layer_2_dw_specs.padding_top; h++)
        {
            for (int w = 0; w < layer_2_dw_ifm_width; w++)
            {
                conv_dw_communication_buffer_inter[d][h][w] = layer_2_dw_specs.layer_ifms_zero_point;
            }
        }
    }
}

void main_pipeline_engine_calls_loop(weights_dt on_chip_weights[][ON_CHIP_WEIGHTS_PORTS],
                                     fms_dt result[MAX_FMS_BUFFER_DEPTH][MIN_FMS_HEIGHT][MIN_FMS_WIDTH],
                                     fms_dt pre_first_pipeline_layers_output[PRE_FIRST_PIPELINE_OUTPUT_DEPTH]
                                                                            [PRE_FIRST_PIPELINE_OUTPUT_HEIGHT]
                                                                            [PRE_FIRST_PIPELINE_OUTPUT_WIDTH],
                                     fms_dt channels_buffer[MAX_PW_BUFFER_DEPTH][PW_BUFFER_HEIGHT][MAX_PW_BUFFER_WIDTH],
                                     fms_dt result_buffer[MAX_PW_BUFFER_DEPTH][PW_BUFFER_HEIGHT][MAX_PW_BUFFER_WIDTH],
                                     fms_dt tmp_channels[MAX_PW_BUFFER_DEPTH][PIPELINE_TMP_CHANNELS_HEIGHT][MAX_PW_BUFFER_WIDTH],
                                     fms_dt dw_pipe_overlap_buffer[DW_PIPE_OVERLAP_BUFFER_DEPTH][2][2][DW_PIPE_OVERLAP_BUFFER_WIDTH],
                                     fms_dt dw_channels_tile[DW_TILE_DEPTH][DW_BUFFER_HEIGHT][DW_BUFFER_WIDTH],
                                     fms_dt dw_channels_tile_copy[DW_TILE_DEPTH][DW_BUFFER_HEIGHT][DW_BUFFER_WIDTH],
                                     const int start_filling_offset_in_buffer_non_first,
                                     const int rows_produced_in_pipeline_filling_phase,
                                     const int pipe_rows_produced_in_a_pass,
                                     layer_specs first_layer_in_second_part,
                                     int &dw_6_odd_even,
                                     int &dw_9_odd_even,
                                     int &dw_14_odd_even,
                                     const int h,
                                     const int start_filling_offset_in_buffer_first_time,
                                     const int rows_to_fill_first_time,
                                     const bool before_pipeline_main_loop)
{
#pragma HLS INLINE off

    if (before_pipeline_main_loop)
    {

        pw_dw_conv(on_chip_weights,
                   pipe_dw_weights_3x3,
                   pre_first_pipeline_layers_output,
                   channels_buffer,
                   result_buffer,
                   tmp_channels,
                   dw_pipe_overlap_buffer,
                   dw_channels_tile,
                   dw_channels_tile_copy,
                   0, // starting_h
                   0, // h_offset_in_result,
                   start_filling_offset_in_buffer_first_time,
                   0,
                   0,
                   layer_3_pw_specs,
                   layer_6_dw_specs,
                   pipe_fused_scales,
                   pipe_fused_scales_log_2_shifts,
                   pipe_relu_6_fused_scales,
                   pipe_fused_zero_points,
                   0,
                   true);
        pw_dw_conv(on_chip_weights,
                   pipe_dw_weights_3x3,
                   pre_first_pipeline_layers_output,
                   result_buffer,
                   channels_buffer,
                   tmp_channels,
                   dw_pipe_overlap_buffer,
                   dw_channels_tile,
                   dw_channels_tile_copy,
                   0, // starting_h
                   0, // h_offset_in_result
                   0,
                   0,
                   1, // fused
                   layer_4_pw_specs,
                   layer_6_dw_specs,
                   pipe_fused_scales,
                   pipe_fused_scales_log_2_shifts,
                   pipe_relu_6_fused_scales,
                   pipe_fused_zero_points,
                   dw_6_odd_even,
                   false);
        dw_6_odd_even = 1 - dw_6_odd_even;
        //######################################################

        pw_dw_conv(on_chip_weights,
                   pipe_dw_weights_3x3,
                   pre_first_pipeline_layers_output,
                   channels_buffer,
                   result_buffer,
                   tmp_channels,
                   dw_pipe_overlap_buffer,
                   dw_channels_tile,
                   dw_channels_tile_copy,
                   rows_to_fill_first_time, // starting_h
                   0,                       // h_offset_in_result,
                   start_filling_offset_in_buffer_non_first,
                   rows_to_fill_first_time,
                   0,
                   layer_3_pw_specs,
                   layer_6_dw_specs,
                   pipe_fused_scales,
                   pipe_fused_scales_log_2_shifts,
                   pipe_relu_6_fused_scales,
                   pipe_fused_zero_points,
                   0,
                   true);

        pw_dw_conv(on_chip_weights,
                   pipe_dw_weights_3x3,
                   pre_first_pipeline_layers_output,
                   result_buffer,
                   channels_buffer,
                   tmp_channels,
                   dw_pipe_overlap_buffer,
                   dw_channels_tile,
                   dw_channels_tile_copy,
                   rows_to_fill_first_time, // starting_h
                   0,                       // h_offset_in_result,
                   0,
                   0,
                   1,
                   layer_4_pw_specs,
                   layer_6_dw_specs,
                   pipe_fused_scales,
                   pipe_fused_scales_log_2_shifts,
                   pipe_relu_6_fused_scales,
                   pipe_fused_zero_points,
                   dw_6_odd_even,
                   false);
        dw_6_odd_even = 1 - dw_6_odd_even;

        for (int d = 0; d < MAX_PW_BUFFER_DEPTH; d++)
        {
            for (int w = 0; w < MAX_PW_BUFFER_WIDTH; w++)
            {
                channels_buffer[d][PW_BUFFER_HEIGHT - 2][w] = channels_buffer[d][0][w];
                channels_buffer[d][PW_BUFFER_HEIGHT - 1][w] = channels_buffer[d][1][w];
            }
        }

        pw_dw_conv(on_chip_weights,
                   pipe_dw_weights_3x3,
                   pre_first_pipeline_layers_output,
                   channels_buffer,
                   result_buffer,
                   tmp_channels,
                   dw_pipe_overlap_buffer,
                   dw_channels_tile,
                   dw_channels_tile_copy,
                   0, // starting_h
                   0, // h_offset_in_result,
                   0,
                   0,
                   0,
                   layer_7_pw_specs,
                   layer_6_dw_specs,
                   pipe_fused_scales,
                   pipe_fused_scales_log_2_shifts,
                   pipe_relu_6_fused_scales,
                   pipe_fused_zero_points,
                   0,
                   false);

        for (int d = 0; d < MAX_PW_BUFFER_DEPTH; d++)
        {
            for (int w = 0; w < PW_BUFFER_WIDTH; w++)
            {
                tmp_channels[d][0][w] = tmp_channels[d][PIPELINE_TMP_CHANNELS_HEIGHT - 2][w]; // two rows were produced
            }
        }

        pw_dw_conv(on_chip_weights,
                   pipe_dw_weights_3x3,
                   pre_first_pipeline_layers_output,
                   result_buffer,
                   channels_buffer,
                   tmp_channels,
                   dw_pipe_overlap_buffer,
                   dw_channels_tile,
                   dw_channels_tile_copy,
                   0, // starting_h
                   0, // h_offset_in_result,
                   0,
                   0, // h_offset_in_result,
                   1,
                   layer_8_pw_specs,
                   layer_9_dw_specs,
                   pipe_fused_scales,
                   pipe_fused_scales_log_2_shifts,
                   pipe_relu_6_fused_scales,
                   pipe_fused_zero_points,
                   dw_9_odd_even,
                   false);
        dw_9_odd_even = 1 - dw_9_odd_even;

        pw_dw_conv(on_chip_weights,
                   pipe_dw_weights_3x3,
                   pre_first_pipeline_layers_output,
                   channels_buffer,
                   result_buffer,
                   tmp_channels,
                   dw_pipe_overlap_buffer,
                   dw_channels_tile,
                   dw_channels_tile_copy,
                   0, // starting_h
                   0, // h_offset_in_result,
                   0,
                   0,
                   0,
                   layer_10_pw_specs,
                   layer_6_dw_specs,
                   pipe_fused_scales,
                   pipe_fused_scales_log_2_shifts,
                   pipe_relu_6_fused_scales,
                   pipe_fused_zero_points,
                   0,
                   false);

        pw_dw_conv(on_chip_weights,
                   pipe_dw_weights_3x3,
                   pre_first_pipeline_layers_output,
                   result_buffer,
                   channels_buffer,
                   tmp_channels,
                   dw_pipe_overlap_buffer,
                   dw_channels_tile,
                   dw_channels_tile_copy,
                   0, // starting_h
                   0, // h_offset_in_result,
                   0,
                   0,
                   1,
                   layer_12_pw_specs,
                   layer_14_dw_specs,
                   pipe_fused_scales,
                   pipe_fused_scales_log_2_shifts,
                   pipe_relu_6_fused_scales,
                   pipe_fused_zero_points,
                   dw_14_odd_even,
                   false);
        dw_14_odd_even = 1 - dw_14_odd_even;
    }
    else
    {
        if (h >= 0)
        {

            for (int i = 0; i < 2; i++)
            { // todo change 2

                pw_dw_conv(on_chip_weights,
                           pipe_dw_weights_3x3,
                           pre_first_pipeline_layers_output,
                           channels_buffer,
                           result_buffer,
                           tmp_channels,
                           dw_pipe_overlap_buffer,
                           dw_channels_tile,
                           dw_channels_tile_copy,
                           h * 4 + (i + 1) * pipe_rows_produced_in_a_pass + rows_produced_in_pipeline_filling_phase, // starting_h
                           0,
                           start_filling_offset_in_buffer_non_first,
                           i * 2 * pipe_rows_produced_in_a_pass,
                           0,
                           layer_3_pw_specs,
                           layer_6_dw_specs,
                           pipe_fused_scales,
                           pipe_fused_scales_log_2_shifts,
                           pipe_relu_6_fused_scales,
                           pipe_fused_zero_points,
                           0,
                           true);
                pw_dw_conv(on_chip_weights,
                           pipe_dw_weights_3x3,
                           pre_first_pipeline_layers_output,
                           result_buffer,
                           channels_buffer,
                           tmp_channels,
                           dw_pipe_overlap_buffer,
                           dw_channels_tile,
                           dw_channels_tile_copy,
                           h * 4 + (i + 1) * pipe_rows_produced_in_a_pass + rows_produced_in_pipeline_filling_phase, // starting_h
                           i * 2,
                           0,
                           0, // h_offset_in_result,
                           1,
                           layer_4_pw_specs,
                           layer_6_dw_specs,
                           pipe_fused_scales,
                           pipe_fused_scales_log_2_shifts,
                           pipe_relu_6_fused_scales,
                           pipe_fused_zero_points,
                           dw_6_odd_even,
                           false);
                dw_6_odd_even = 1 - dw_6_odd_even;
            }

            pw_dw_conv(on_chip_weights,
                       pipe_dw_weights_3x3,
                       pre_first_pipeline_layers_output,
                       channels_buffer,
                       result_buffer,
                       tmp_channels,
                       dw_pipe_overlap_buffer,
                       dw_channels_tile,
                       dw_channels_tile_copy,
                       h * 2 +
                           rows_produced_in_pipeline_filling_phase + 1, // starting_h
                       0,
                       0,
                       0, // h_offset_in_result,
                       0,
                       layer_7_pw_specs,
                       layer_6_dw_specs,
                       pipe_fused_scales,
                       pipe_fused_scales_log_2_shifts,
                       pipe_relu_6_fused_scales,
                       pipe_fused_zero_points,
                       0,
                       false);

            pw_dw_conv(on_chip_weights,
                       pipe_dw_weights_3x3,
                       pre_first_pipeline_layers_output,
                       result_buffer,
                       channels_buffer,
                       tmp_channels,
                       dw_pipe_overlap_buffer,
                       dw_channels_tile,
                       dw_channels_tile_copy,
                       h * 2 +
                           rows_produced_in_pipeline_filling_phase + 1, // starting_h
                       0,                                               // h_offset_in_result,
                       0,
                       0,
                       1,
                       layer_8_pw_specs,
                       layer_9_dw_specs,
                       pipe_fused_scales,
                       pipe_fused_scales_log_2_shifts,
                       pipe_relu_6_fused_scales,
                       pipe_fused_zero_points,
                       dw_9_odd_even,
                       false);

            dw_9_odd_even = 1 - dw_9_odd_even;

            pw_dw_conv(on_chip_weights,
                       pipe_dw_weights_3x3,
                       pre_first_pipeline_layers_output,
                       channels_buffer,
                       result_buffer,
                       tmp_channels,
                       dw_pipe_overlap_buffer,
                       dw_channels_tile,
                       dw_channels_tile_copy,
                       h * 2 +
                           rows_produced_in_pipeline_filling_phase + 1, // starting_h
                       0,                                               // h_offset_in_result,
                       0,
                       0,
                       0,
                       layer_10_pw_specs,
                       layer_6_dw_specs,
                       pipe_fused_scales,
                       pipe_fused_scales_log_2_shifts,
                       pipe_relu_6_fused_scales,
                       pipe_fused_zero_points,
                       0,
                       false);

            pw_dw_conv(on_chip_weights,
                       pipe_dw_weights_3x3,
                       pre_first_pipeline_layers_output,
                       result_buffer,
                       channels_buffer,
                       tmp_channels,
                       dw_pipe_overlap_buffer,
                       dw_channels_tile,
                       dw_channels_tile_copy,
                       h * 2 + rows_produced_in_pipeline_filling_phase, // starting_h
                       (h & 1) * OFFSET_H_IN_PIPELINE_RESULTS,          // h_offset_in_result,
                       0,
                       0,
                       1,
                       layer_12_pw_specs,
                       layer_14_dw_specs,
                       pipe_fused_scales,
                       pipe_fused_scales_log_2_shifts,
                       pipe_relu_6_fused_scales,
                       pipe_fused_zero_points,
                       dw_14_odd_even,
                       false);
            dw_14_odd_even = 1 - dw_14_odd_even;
        }
    }
}

void pipelined_engines_caller(fms_grp_dt input_image[input_image_depth * input_image_num_fms_groups_in_a_channel],
                              weights_dt on_chip_weights[][ON_CHIP_WEIGHTS_PORTS],
                              fms_dt result[MAX_FMS_BUFFER_DEPTH][MIN_FMS_HEIGHT][MIN_FMS_WIDTH])
{

    fms_dt channels_buffer[MAX_PW_BUFFER_DEPTH][PW_BUFFER_HEIGHT][MAX_PW_BUFFER_WIDTH];

#pragma HLS ARRAY_PARTITION variable = channels_buffer type = complete dim = 2
#pragma HLS ARRAY_PARTITION variable = channels_buffer type = cyclic factor = PW_BUFFER_WIDTH dim = 3

    fms_dt channels_buffer_copy[MAX_PW_BUFFER_DEPTH][PW_BUFFER_HEIGHT][MAX_PW_BUFFER_WIDTH];

#pragma HLS ARRAY_PARTITION variable = channels_buffer_copy type = complete dim = 2
#pragma HLS ARRAY_PARTITION variable = channels_buffer_copy type = cyclic factor = PW_BUFFER_WIDTH dim = 3

    fms_dt result_buffer[MAX_PW_BUFFER_DEPTH][PW_BUFFER_HEIGHT][MAX_PW_BUFFER_WIDTH];

#pragma HLS ARRAY_PARTITION variable = result_buffer type = complete dim = 2
#pragma HLS ARRAY_PARTITION variable = result_buffer type = cyclic factor = PW_BUFFER_WIDTH dim = 3

    fms_dt tmp_channels[MAX_PW_BUFFER_DEPTH][PIPELINE_TMP_CHANNELS_HEIGHT][MAX_PW_BUFFER_WIDTH];

#pragma HLS ARRAY_PARTITION variable = tmp_channels type = complete dim = 2

    fms_dt dw_pipe_overlap_buffer[DW_PIPE_OVERLAP_BUFFER_DEPTH][2][2][DW_PIPE_OVERLAP_BUFFER_WIDTH];

#pragma HLS ARRAY_PARTITION variable = dw_pipe_overlap_buffer type = cyclic factor = 2 dim = 1
#pragma HLS ARRAY_PARTITION variable = dw_pipe_overlap_buffer type = complete dim = 2
#pragma HLS ARRAY_PARTITION variable = dw_pipe_overlap_buffer type = complete dim = 3
#pragma HLS ARRAY_PARTITION variable = dw_pipe_overlap_buffer type = cyclic factor = PW_BUFFER_WIDTH dim = 4

    fms_dt dw_channels_tile[DW_TILE_DEPTH][DW_BUFFER_HEIGHT][DW_BUFFER_WIDTH];
    fms_dt dw_channels_tile_copy[DW_TILE_DEPTH][DW_BUFFER_HEIGHT][DW_BUFFER_WIDTH];

#pragma HLS ARRAY_PARTITION variable = dw_channels_tile type = complete dim = 2
#pragma HLS ARRAY_PARTITION variable = dw_channels_tile type = complete dim = 3
#pragma HLS ARRAY_PARTITION variable = dw_channels_tile_copy type = complete dim = 2
#pragma HLS ARRAY_PARTITION variable = dw_channels_tile_copy type = complete dim = 3

    layer_specs first_layer_in_second_part = layer_15_pw_specs;

    weights_dt dw_layer_weights[layer_2_dw_num_fils][layer_2_dw_filter_dim * layer_2_dw_filter_dim];
    fill_first_dw_layer_weights(dw_layer_weights);

    fms_quantization_scheme first_layer_quantization_params[layer_2_dw_num_fils];
    fms_quantization_scheme first_dw_layer_quantization_params[layer_2_dw_num_fils];
    fill_first_conv_layers_quantization_params(first_layer_quantization_params, 0);
    fill_first_conv_layers_quantization_params(first_dw_layer_quantization_params, first_conv_layer_num_fils);

    fms_dt conv_dw_communication_buffer_inter[first_conv_layer_num_fils][layer_2_dw_filter_dim]
                                             [layer_2_dw_ifm_width];

#pragma HLS ARRAY_PARTITION variable = conv_dw_communication_buffer_inter type = complete dim = 2

    padd_top_conv_dw_communication_buffer_inter(conv_dw_communication_buffer_inter);

    fms_dt first_layers_input[input_image_depth][PRE_FIRST_PIPELINE_INPUT_HEIGHT][input_image_width];

    const int rows_to_fill_first_time = 1;
    const int start_filling_offset_in_buffer_first_time = PW_BUFFER_HEIGHT - rows_to_fill_first_time;
    const int start_filling_offset_in_buffer_non_first = 0;

    const int switching_layer_strides = layer_9_dw_specs.strides;
    const int pipe_rows_produced_in_a_pass = 2;

    int dw_6_odd_even = 0;
    int dw_9_odd_even = 0;
    int dw_14_odd_even = 0;
    //######################################################
    fms_dt pre_first_pipeline_layers_output[PRE_FIRST_PIPELINE_OUTPUT_DEPTH]
                                           [PRE_FIRST_PIPELINE_OUTPUT_HEIGHT]
                                           [PRE_FIRST_PIPELINE_OUTPUT_WIDTH];
    fms_dt pre_first_pipeline_layers_output_copy[PRE_FIRST_PIPELINE_OUTPUT_DEPTH]
                                                [PRE_FIRST_PIPELINE_OUTPUT_HEIGHT]
                                                [PRE_FIRST_PIPELINE_OUTPUT_WIDTH];

#pragma HLS ARRAY_PARTITION variable = pre_first_pipeline_layers_output type = complete dim = 2
#pragma HLS ARRAY_PARTITION variable = pre_first_pipeline_layers_output_copy type = complete dim = 2

    pre_first_pipeline_layers_mob_v2(input_image,
                                     pre_first_pipeline_layers_output,
                                     dw_layer_weights,
                                     first_layer_quantization_params,
                                     first_dw_layer_quantization_params,
                                     conv_dw_communication_buffer_inter,
                                     first_layers_input,
                                     0,
                                     5 + 4 * first_conv_layer_strides);

    const int rows_produced_in_pipeline_filling_phase = 1; // todo
    int even_odd = 1;
    int prev_h = -pipe_rows_produced_in_a_pass;
    int prev_prev_h = -2 * pipe_rows_produced_in_a_pass;

    main_pipeline_engine_calls_loop(on_chip_weights,
                                    result,
                                    pre_first_pipeline_layers_output,
                                    channels_buffer,
                                    result_buffer,
                                    tmp_channels,
                                    dw_pipe_overlap_buffer,
                                    dw_channels_tile,
                                    dw_channels_tile_copy,
                                    start_filling_offset_in_buffer_non_first,
                                    rows_produced_in_pipeline_filling_phase,
                                    pipe_rows_produced_in_a_pass,
                                    first_layer_in_second_part,
                                    dw_6_odd_even,
                                    dw_9_odd_even,
                                    dw_14_odd_even,
                                    prev_h,
                                    start_filling_offset_in_buffer_first_time,
                                    rows_to_fill_first_time, true);

    const int first_layer_in_second_part_height = first_layer_in_second_part.layer_ifm_height;

    int indices_to_solve_timing_issue[first_layer_in_second_part_height / pipe_rows_produced_in_a_pass][2];

    for (int h = 0; h < first_layer_in_second_part_height / pipe_rows_produced_in_a_pass; h++)
    {
        indices_to_solve_timing_issue[h][0] = 5 + 4 * first_conv_layer_strides +
                                              h * PRE_FIRST_PIPELINE_OUTPUT_HEIGHT * first_conv_layer_strides;
        indices_to_solve_timing_issue[h][1] = 5 + 4 * first_conv_layer_strides +
                                              (h + 1) * PRE_FIRST_PIPELINE_OUTPUT_HEIGHT * first_conv_layer_strides;
    }
    for (int h = 0; h < first_layer_in_second_part_height / pipe_rows_produced_in_a_pass; h++)
    {
#pragma HLS PIPELINE off

        if (even_odd)
        {
            write_pipe_seml_communication_buffer(
                channels_buffer_copy,
                result,
                prev_prev_h, // starting_h
                0,
                first_layer_in_second_part);
            pre_first_pipeline_layers_mob_v2(input_image,
                                             pre_first_pipeline_layers_output,
                                             dw_layer_weights,
                                             first_layer_quantization_params,
                                             first_dw_layer_quantization_params,
                                             conv_dw_communication_buffer_inter,
                                             first_layers_input,
                                             indices_to_solve_timing_issue[h][0],
                                             indices_to_solve_timing_issue[h][1]);

            main_pipeline_engine_calls_loop(on_chip_weights,
                                            result,
                                            pre_first_pipeline_layers_output_copy,
                                            channels_buffer,
                                            result_buffer,
                                            tmp_channels,
                                            dw_pipe_overlap_buffer,
                                            dw_channels_tile,
                                            dw_channels_tile_copy,
                                            start_filling_offset_in_buffer_non_first,
                                            rows_produced_in_pipeline_filling_phase,
                                            pipe_rows_produced_in_a_pass,
                                            first_layer_in_second_part,
                                            dw_6_odd_even,
                                            dw_9_odd_even,
                                            dw_14_odd_even,
                                            prev_h,
                                            start_filling_offset_in_buffer_first_time,
                                            rows_to_fill_first_time, false);
        }
        else
        {
            write_pipe_seml_communication_buffer(
                channels_buffer,
                result,
                prev_prev_h, // starting_h
                0,
                first_layer_in_second_part);
            pre_first_pipeline_layers_mob_v2(input_image,
                                             pre_first_pipeline_layers_output_copy,
                                             dw_layer_weights,
                                             first_layer_quantization_params,
                                             first_dw_layer_quantization_params,
                                             conv_dw_communication_buffer_inter,
                                             first_layers_input,
                                             indices_to_solve_timing_issue[h][0],
                                             indices_to_solve_timing_issue[h][1]);

            main_pipeline_engine_calls_loop(on_chip_weights,
                                            result,
                                            pre_first_pipeline_layers_output,
                                            channels_buffer_copy,
                                            result_buffer,
                                            tmp_channels,
                                            dw_pipe_overlap_buffer,
                                            dw_channels_tile,
                                            dw_channels_tile_copy,
                                            start_filling_offset_in_buffer_non_first,
                                            rows_produced_in_pipeline_filling_phase,
                                            pipe_rows_produced_in_a_pass,
                                            first_layer_in_second_part,
                                            dw_6_odd_even,
                                            dw_9_odd_even,
                                            dw_14_odd_even,
                                            prev_h,
                                            start_filling_offset_in_buffer_first_time,
                                            rows_to_fill_first_time, false);
        }
        prev_h += pipe_rows_produced_in_a_pass;
        prev_prev_h += pipe_rows_produced_in_a_pass;
        even_odd = 1 - even_odd;
    }

    if (even_odd)
    {
        write_pipe_seml_communication_buffer(
            channels_buffer_copy,
            result,
            prev_prev_h, // starting_h
            0,
            first_layer_in_second_part);
        prev_prev_h += pipe_rows_produced_in_a_pass;
        //************************************
        main_pipeline_engine_calls_loop(on_chip_weights,
                                        result,
                                        pre_first_pipeline_layers_output_copy,
                                        channels_buffer,
                                        result_buffer,
                                        tmp_channels,
                                        dw_pipe_overlap_buffer,
                                        dw_channels_tile,
                                        dw_channels_tile_copy,
                                        start_filling_offset_in_buffer_non_first,
                                        rows_produced_in_pipeline_filling_phase,
                                        pipe_rows_produced_in_a_pass,
                                        first_layer_in_second_part,
                                        dw_6_odd_even,
                                        dw_9_odd_even,
                                        dw_14_odd_even,
                                        prev_h,
                                        start_filling_offset_in_buffer_first_time,
                                        rows_to_fill_first_time, false);
        write_pipe_seml_communication_buffer(
            channels_buffer,
            result,
            prev_h, // starting_h
            0,
            first_layer_in_second_part);
    }
    else
    {
        write_pipe_seml_communication_buffer(
            channels_buffer,
            result,
            prev_prev_h, // starting_h
            0,
            first_layer_in_second_part);
        prev_prev_h += pipe_rows_produced_in_a_pass;
        //************************************
        main_pipeline_engine_calls_loop(on_chip_weights,
                                        result,
                                        pre_first_pipeline_layers_output,
                                        channels_buffer,
                                        result_buffer,
                                        tmp_channels,
                                        dw_pipe_overlap_buffer,
                                        dw_channels_tile,
                                        dw_channels_tile_copy,
                                        start_filling_offset_in_buffer_non_first,
                                        rows_produced_in_pipeline_filling_phase,
                                        pipe_rows_produced_in_a_pass,
                                        first_layer_in_second_part,
                                        dw_6_odd_even,
                                        dw_9_odd_even,
                                        dw_14_odd_even,
                                        prev_h,
                                        start_filling_offset_in_buffer_first_time,
                                        rows_to_fill_first_time, false);
        write_pipe_seml_communication_buffer(
            channels_buffer,
            result,
            prev_h, // starting_h
            0,
            first_layer_in_second_part);
    }
}

#endif
