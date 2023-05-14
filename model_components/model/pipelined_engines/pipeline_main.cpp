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

void pipelined_engines_caller(weights_dt on_chip_weights[][ON_CHIP_WEIGHTS_PORTS],
                              fms_dt result[MAX_FMS_BUFFER_DEPTH][MIN_FMS_HEIGHT][MIN_FMS_WIDTH])
{
    const int tmp_channels_height = PW_BUFFER_HEIGHT + 1;

    fms_dt channels_buffer_0[MAX_PW_BUFFER_DEPTH][PW_BUFFER_HEIGHT][MAX_PW_BUFFER_WIDTH];

#pragma HLS ARRAY_PARTITION variable = channels_buffer_0 type = complete dim = 2
#pragma HLS ARRAY_PARTITION variable = channels_buffer_0 type = cyclic factor = PW_BUFFER_WIDTH dim = 3

    fms_dt channels_buffer[MAX_PW_BUFFER_DEPTH][PW_BUFFER_HEIGHT][MAX_PW_BUFFER_WIDTH];

#pragma HLS ARRAY_PARTITION variable = channels_buffer type = complete dim = 2
#pragma HLS ARRAY_PARTITION variable = channels_buffer type = cyclic factor = PW_BUFFER_WIDTH dim = 3

    fms_dt result_buffer[MAX_PW_BUFFER_DEPTH][PW_BUFFER_HEIGHT][MAX_PW_BUFFER_WIDTH];

#pragma HLS ARRAY_PARTITION variable = result_buffer type = complete dim = 2
#pragma HLS ARRAY_PARTITION variable = result_buffer type = cyclic factor = PW_BUFFER_WIDTH dim = 3

    fms_dt tmp_channels[MAX_PW_BUFFER_DEPTH][tmp_channels_height][MAX_PW_BUFFER_WIDTH];

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

    // padd_top_dw_channels_tile(dw_channels_tile, dw_channels_tile_copy,
    //                           layer_6_dw_specs);

    const int rows_to_fill_first_time = 1;
    const int start_filling_offset_in_buffer_first_time = PW_BUFFER_HEIGHT - rows_to_fill_first_time;
    const int start_filling_offset_in_buffer_non_first = 0;

    const int switching_layer_strides = layer_9_dw_specs.strides;
    const int pipe_rows_produced_in_a_pass = PARALLELISM_PW_H;

    int dw_6_odd_even = 0;
    int dw_9_odd_even = 0;
    int dw_14_odd_even = 0;
    //######################################################
#if HW == CPU
    fill_pipe_layer_input_buffer(
        "/media/SSD2TB/wd/my_repos/DL_Benchmarking/tflite_scripts_imgnt_accuracy_and_weight_extraction/mob_v2/fms/ifms_3.txt",
        channels_buffer_0, 0, start_filling_offset_in_buffer_first_time, layer_3_pw_specs);
#endif
    pw_dw_conv(on_chip_weights,
               pipe_dw_weights_3x3,
               result_buffer,
               channels_buffer_0,
               channels_buffer,
               tmp_channels,
               dw_pipe_overlap_buffer,
               dw_channels_tile,
               dw_channels_tile_copy,
               0, // starting_h
               0, // h_offset_in_result,
               0,
               layer_3_pw_specs,
               layer_6_dw_specs,
               pipe_fused_scales,
               pipe_fused_scales_log_2_shifts,
               pipe_relu_6_fused_scales,
               pipe_fused_zero_points,
               0);
    pw_dw_conv(on_chip_weights,
               pipe_dw_weights_3x3,
               channels_buffer_0,
               channels_buffer,
               result_buffer,
               tmp_channels,
               dw_pipe_overlap_buffer,
               dw_channels_tile,
               dw_channels_tile_copy,
               0, // starting_h
               0, // h_offset_in_result
               1, // fused
               layer_4_pw_specs,
               layer_6_dw_specs,
               pipe_fused_scales,
               pipe_fused_scales_log_2_shifts,
               pipe_relu_6_fused_scales,
               pipe_fused_zero_points,
               dw_6_odd_even);
    dw_6_odd_even = 1 - dw_6_odd_even;
    //######################################################
#if HW == CPU
    fill_pipe_layer_input_buffer(
        "/media/SSD2TB/wd/my_repos/DL_Benchmarking/tflite_scripts_imgnt_accuracy_and_weight_extraction/mob_v2/fms/ifms_3.txt",
        channels_buffer_0, rows_to_fill_first_time,
        start_filling_offset_in_buffer_non_first, layer_3_pw_specs);
#endif

    pw_dw_conv(on_chip_weights,
               pipe_dw_weights_3x3,
               result_buffer,
               channels_buffer_0,
               channels_buffer,
               tmp_channels,
               dw_pipe_overlap_buffer,
               dw_channels_tile,
               dw_channels_tile_copy,
               rows_to_fill_first_time, // starting_h
               0,                       // h_offset_in_result,
               0,
               layer_3_pw_specs,
               layer_6_dw_specs,
               pipe_fused_scales,
               pipe_fused_scales_log_2_shifts,
               pipe_relu_6_fused_scales,
               pipe_fused_zero_points,
               0);

    pw_dw_conv(on_chip_weights,
               pipe_dw_weights_3x3,
               channels_buffer_0,
               channels_buffer,
               result_buffer,
               tmp_channels,
               dw_pipe_overlap_buffer,
               dw_channels_tile,
               dw_channels_tile_copy,
               rows_to_fill_first_time, // starting_h
               0,                       // h_offset_in_result,
               1,
               layer_4_pw_specs,
               layer_6_dw_specs,
               pipe_fused_scales,
               pipe_fused_scales_log_2_shifts,
               pipe_relu_6_fused_scales,
               pipe_fused_zero_points,
               dw_6_odd_even);
    dw_6_odd_even = 1 - dw_6_odd_even;

    for (int d = 0; d < MAX_PW_BUFFER_DEPTH; d++)
    {
        for (int w = 0; w < MAX_PW_BUFFER_WIDTH; w++)
        {
            result_buffer[d][PW_BUFFER_HEIGHT - 2][w] = result_buffer[d][0][w];
            result_buffer[d][PW_BUFFER_HEIGHT - 1][w] = result_buffer[d][1][w];
        }
    }
    pw_dw_conv(on_chip_weights,
               pipe_dw_weights_3x3,
               channels_buffer_0,
               result_buffer,
               channels_buffer,
               tmp_channels,
               dw_pipe_overlap_buffer,
               dw_channels_tile,
               dw_channels_tile_copy,
               0, // starting_h
               0, // h_offset_in_result,
               0,
               layer_7_pw_specs,
               layer_6_dw_specs,
               pipe_fused_scales,
               pipe_fused_scales_log_2_shifts,
               pipe_relu_6_fused_scales,
               pipe_fused_zero_points,
               0);

    for (int d = 0; d < MAX_PW_BUFFER_DEPTH; d++)
    {
        for (int w = 0; w < PW_BUFFER_WIDTH; w++)
        {
            tmp_channels[d][0][w] = tmp_channels[d][tmp_channels_height - 2][w]; // two rows were produced
        }
    }

    pw_dw_conv(on_chip_weights,
               pipe_dw_weights_3x3,
               channels_buffer_0,
               channels_buffer,
               result_buffer,
               tmp_channels,
               dw_pipe_overlap_buffer,
               dw_channels_tile,
               dw_channels_tile_copy,
               0, // starting_h
               0, // h_offset_in_result,
               1,
               layer_8_pw_specs,
               layer_9_dw_specs,
               pipe_fused_scales,
               pipe_fused_scales_log_2_shifts,
               pipe_relu_6_fused_scales,
               pipe_fused_zero_points,
               dw_9_odd_even);
    dw_9_odd_even = 1 - dw_9_odd_even;

    pw_dw_conv(on_chip_weights,
               pipe_dw_weights_3x3,
               channels_buffer_0,
               result_buffer,
               channels_buffer,
               tmp_channels,
               dw_pipe_overlap_buffer,
               dw_channels_tile,
               dw_channels_tile_copy,
               0, // starting_h
               0, // h_offset_in_result,
               0,
               layer_10_pw_specs,
               layer_6_dw_specs,
               pipe_fused_scales,
               pipe_fused_scales_log_2_shifts,
               pipe_relu_6_fused_scales,
               pipe_fused_zero_points,
               0);

    pw_dw_conv(on_chip_weights,
               pipe_dw_weights_3x3,
               channels_buffer_0,
               channels_buffer,
               result_buffer,
               tmp_channels,
               dw_pipe_overlap_buffer,
               dw_channels_tile,
               dw_channels_tile_copy,
               0, // starting_h
               0, // h_offset_in_result,
               1,
               layer_12_pw_specs,
               layer_14_dw_specs,
               pipe_fused_scales,
               pipe_fused_scales_log_2_shifts,
               pipe_relu_6_fused_scales,
               pipe_fused_zero_points,
               dw_14_odd_even);
    dw_14_odd_even = 1 - dw_14_odd_even;
    // write_pipe_seml_communication_buffer(
    //     channels_buffer,
    //     result,
    //     0, // starting_h
    //     3, // todo
    //     first_layer_in_second_part);

    const int rows_produced_in_pipeline_filling_phase = 1; // todo

    for (int h = 0; h < first_layer_in_second_part.layer_ifm_height; h += pipe_rows_produced_in_a_pass)
    {
        for (int o_i = 0; o_i < 2; o_i++)
        { // todo change 2
            for (int i = 0; i < 2; i++)
            { // todo change 2
#if HW == CPU
                fill_pipe_layer_input_buffer(
                    "/media/SSD2TB/wd/my_repos/DL_Benchmarking/tflite_scripts_imgnt_accuracy_and_weight_extraction/mob_v2/fms/ifms_3.txt",
                    channels_buffer_0, h * 4 + (o_i * 2 + i + 1) * pipe_rows_produced_in_a_pass + rows_produced_in_pipeline_filling_phase,
                    start_filling_offset_in_buffer_non_first, layer_3_pw_specs);
#endif
                pw_dw_conv(on_chip_weights,
                           pipe_dw_weights_3x3,
                           result_buffer,
                           channels_buffer_0,
                           channels_buffer,
                           tmp_channels,
                           dw_pipe_overlap_buffer,
                           dw_channels_tile,
                           dw_channels_tile_copy,
                           h * 4 + (o_i * 2 + i + 1) * pipe_rows_produced_in_a_pass + rows_produced_in_pipeline_filling_phase, // starting_h
                           0,
                           0,
                           layer_3_pw_specs,
                           layer_6_dw_specs,
                           pipe_fused_scales,
                           pipe_fused_scales_log_2_shifts,
                           pipe_relu_6_fused_scales,
                           pipe_fused_zero_points,
                           0);
                pw_dw_conv(on_chip_weights,
                           pipe_dw_weights_3x3,
                           channels_buffer_0,
                           channels_buffer,
                           result_buffer,
                           tmp_channels,
                           dw_pipe_overlap_buffer,
                           dw_channels_tile,
                           dw_channels_tile_copy,
                           h * 4 + (o_i * 2 + i + 1) * pipe_rows_produced_in_a_pass + rows_produced_in_pipeline_filling_phase, // starting_h
                           i * 2,                                                                                              // h_offset_in_result,
                           1,
                           layer_4_pw_specs,
                           layer_6_dw_specs,
                           pipe_fused_scales,
                           pipe_fused_scales_log_2_shifts,
                           pipe_relu_6_fused_scales,
                           pipe_fused_zero_points,
                           dw_6_odd_even);
                dw_6_odd_even = 1 - dw_6_odd_even;
            }

            pw_dw_conv(on_chip_weights,
                       pipe_dw_weights_3x3,
                       channels_buffer_0,
                       result_buffer,
                       channels_buffer,
                       tmp_channels,
                       dw_pipe_overlap_buffer,
                       dw_channels_tile,
                       dw_channels_tile_copy,
                       h * 2 + o_i * pipe_rows_produced_in_a_pass +
                           rows_produced_in_pipeline_filling_phase + 1, // starting_h
                       0,                                               // h_offset_in_result,
                       0,
                       layer_7_pw_specs,
                       layer_6_dw_specs,
                       pipe_fused_scales,
                       pipe_fused_scales_log_2_shifts,
                       pipe_relu_6_fused_scales,
                       pipe_fused_zero_points,
                       0);

            pw_dw_conv(on_chip_weights,
                       pipe_dw_weights_3x3,
                       channels_buffer_0,
                       channels_buffer,
                       result_buffer,
                       tmp_channels,
                       dw_pipe_overlap_buffer,
                       dw_channels_tile,
                       dw_channels_tile_copy,
                       h * 2 + o_i * pipe_rows_produced_in_a_pass +
                           rows_produced_in_pipeline_filling_phase + 1, // starting_h
                       0,                                               // h_offset_in_result,
                       1,
                       layer_8_pw_specs,
                       layer_9_dw_specs,
                       pipe_fused_scales,
                       pipe_fused_scales_log_2_shifts,
                       pipe_relu_6_fused_scales,
                       pipe_fused_zero_points,
                       dw_9_odd_even);

            dw_9_odd_even = 1 - dw_9_odd_even;

            pw_dw_conv(on_chip_weights,
                       pipe_dw_weights_3x3,
                       channels_buffer_0,
                       result_buffer,
                       channels_buffer,
                       tmp_channels,
                       dw_pipe_overlap_buffer,
                       dw_channels_tile,
                       dw_channels_tile_copy,
                       h * 2 + o_i * pipe_rows_produced_in_a_pass +
                           rows_produced_in_pipeline_filling_phase + 1, // starting_h
                       0,                                               // h_offset_in_result,
                       0,
                       layer_10_pw_specs,
                       layer_6_dw_specs,
                       pipe_fused_scales,
                       pipe_fused_scales_log_2_shifts,
                       pipe_relu_6_fused_scales,
                       pipe_fused_zero_points,
                       0);

            pw_dw_conv(on_chip_weights,
                       pipe_dw_weights_3x3,
                       channels_buffer_0,
                       channels_buffer,
                       result_buffer,
                       tmp_channels,
                       dw_pipe_overlap_buffer,
                       dw_channels_tile,
                       dw_channels_tile_copy,
                       h * 2 + o_i * pipe_rows_produced_in_a_pass + rows_produced_in_pipeline_filling_phase, // starting_h
                       o_i * 2,                                                                              // h_offset_in_result,
                       1,
                       layer_12_pw_specs,
                       layer_14_dw_specs,
                       pipe_fused_scales,
                       pipe_fused_scales_log_2_shifts,
                       pipe_relu_6_fused_scales,
                       pipe_fused_zero_points,
                       dw_14_odd_even);
            dw_14_odd_even = 1 - dw_14_odd_even;

            write_pipe_seml_communication_buffer(
                result_buffer,
                result,
                h, // starting_h
                o_i * 2,
                first_layer_in_second_part);
        }
    }
}

#endif
