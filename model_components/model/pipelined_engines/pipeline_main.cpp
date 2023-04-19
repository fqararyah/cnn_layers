#include "pipeline_main.h"

#if FIRST_PART_IMPLEMENTATION == PIPELINED_ENGINES_MODE

using namespace pipelined_engines;

void padd_lr_dw_channels_tile(fms_dt dw_channels_tile[DW_TILE_DEPTH][MAX_DW_BUFFER_HEIGHT][MAX_DW_BUFFER_WIDTH],
                              layer_specs layer_specs_struct)
{
    const fms_dt current_layer_ifms_zero_point = layer_specs_struct.layer_ifms_zero_point;
    const int layer_ifms_width = layer_specs_struct.layer_ifm_width;

    for (int d = 0; d < DW_TILE_DEPTH; d++)
    {
        for (int h = 0; h < MAX_DW_BUFFER_HEIGHT; h++)
        {
            for (int w = 0; w < MAX_DW_PADDING_IN_PIPE; w++)
            {
                dw_channels_tile[d][h][w] = current_layer_ifms_zero_point;
                dw_channels_tile[d][h][layer_ifms_width + MAX_DW_PADDING_IN_PIPE + w] = current_layer_ifms_zero_point;
            }
        }
    }
}

void padd_top_dw_channels_tile(fms_dt dw_channels_tile[DW_TILE_DEPTH][MAX_DW_BUFFER_HEIGHT][MAX_DW_BUFFER_WIDTH],
                               layer_specs layer_specs_struct)
{
    const fms_dt current_layer_ifms_zero_point = layer_specs_struct.layer_ifms_zero_point;
    const int layer_ifms_width = layer_specs_struct.layer_ifm_width;
    const int padding_top = layer_specs_struct.padding_top;

    for (int d = 0; d < DW_TILE_DEPTH; d++)
    {
        for (int h = 0; h < padding_top; h++)
        {
            for (int w = 0; w < MAX_DW_BUFFER_WIDTH; w++)
            {
                dw_channels_tile[d][h][w] = current_layer_ifms_zero_point;
            }
        }
    }
}

void padd_bottom_dw_channels_tile(fms_dt dw_channels_tile[DW_TILE_DEPTH][MAX_DW_BUFFER_HEIGHT][MAX_DW_BUFFER_WIDTH],
                                  layer_specs layer_specs_struct)
{
    const fms_dt current_layer_ifms_zero_point = layer_specs_struct.layer_ifms_zero_point;
    const int layer_ifms_width = layer_specs_struct.layer_ifm_width;
    const int padding_bottom = layer_specs_struct.padding_bottom;

    for (int d = 0; d < DW_TILE_DEPTH; d++)
    {
        for (int h = 0; h < MAX_DW_PADDING_IN_PIPE; h++)
        {
            for (int w = 0; w < MAX_DW_BUFFER_WIDTH; w++)
            {
                dw_channels_tile[d][MAX_DW_BUFFER_HEIGHT - padding_bottom + h][w] = current_layer_ifms_zero_point;
            }
        }
    }
}

void write_pipe_seml_communication_buffer(
    fms_dt pipe_seml_communication_buffer[MAX_PW_BUFFER_DEPTH][MAX_PW_BUFFER_HEIGHT][MAX_PW_BUFFER_WIDTH],
    fms_dt result[MAX_FMS_BUFFER_DEPTH][MIN_FMS_HEIGHT][MIN_FMS_WIDTH],
    const int starting_h,
    const layer_specs layer_specs_struct)
{
#pragma HLS INLINE off

    const int layer_depth = layer_specs_struct.layer_depth;
    const int layer_ifms_width = layer_specs_struct.layer_ifm_width;
    const int num_of_tiles_h = layer_specs_struct.layer_num_of_ofm_tiles_h;
    const int num_of_tiles_w = layer_specs_struct.layer_num_of_ofm_tiles_w;
    const int num_of_tiles_hw = num_of_tiles_h * num_of_tiles_w;

    for (int d = 0; d < layer_depth; d++)
    {
        for (int h = 0; h < PIPE_TO_SEML_NUM_ROWS_TO_COPY; h++)
        {
            for (int w = 0; w < layer_ifms_width; w++)
            {
                int tile_index = d * num_of_tiles_hw + ((h + starting_h) / CHANNELS_TILE_HEIGHT) * num_of_tiles_w +
                                 w / CHANNELS_TILE_WIDTH;
                int h_in_tile = (h + starting_h) % CHANNELS_TILE_HEIGHT;
                int w_in_tile = w % CHANNELS_TILE_WIDTH;
                result[tile_index][h_in_tile][w_in_tile] = pipe_seml_communication_buffer[d][h][w];
            }
        }
    }
}

void pipelined_engines_caller(fms_dt result[MAX_FMS_BUFFER_DEPTH][MIN_FMS_HEIGHT][MIN_FMS_WIDTH])
{
    fms_dt channels_buffer[MAX_PW_BUFFER_DEPTH][MAX_PW_BUFFER_HEIGHT][MAX_PW_BUFFER_WIDTH];
    fms_dt result_buffer[MAX_PW_BUFFER_DEPTH][MAX_PW_BUFFER_HEIGHT][MAX_PW_BUFFER_WIDTH];
    fms_dt tmp_channels[PARALLELISM_PW_OFMS][PARALLELISM_PW_H][MAX_PW_BUFFER_WIDTH];
    fms_dt dw_pipe_overlap_buffer[DW_PIPE_OVERLAP_BUFFER_DEPTH][DW_PIPE_OVERLAP_BUFFER_WIDTH];
    fms_dt dw_channels_tile[DW_TILE_DEPTH][MAX_DW_BUFFER_HEIGHT][MAX_DW_BUFFER_WIDTH];
    fms_dt dw_channels_tile_copy[DW_TILE_DEPTH][MAX_DW_BUFFER_HEIGHT][MAX_DW_BUFFER_WIDTH];

    padd_top_dw_channels_tile(dw_channels_tile,
                              layer_9_dw_specs);

    padd_lr_dw_channels_tile(dw_channels_tile,
                             layer_9_dw_specs);

    fill_pipe_layer_input_buffer("/media/SSD2TB/wd/my_repos/DL_Benchmarking/tflite_scripts_imgnt_accuracy_and_weight_extraction/mob_v2/fms/ifms_12.txt",
                                 channels_buffer, 0, layer_12_pw_specs);

    pw_dw_conv(on_chip_pw_weights,
               pipe_dw_weights_3x3,
               channels_buffer,
               result_buffer,
               tmp_channels,
               dw_pipe_overlap_buffer,
               dw_channels_tile,
               dw_channels_tile_copy,
               0, // starting_h
               layer_12_pw_specs,
               layer_14_dw_specs,
               pipe_fused_scales,
               pipe_fused_scales_log_2_shifts,
               pipe_relu_6_fused_scales,
               pipe_fused_zero_points);

    write_pipe_seml_communication_buffer(
        result_buffer,
        result,
        0, // starting_h
        layer_12_pw_specs);

    const int switching_layer_strides = layer_14_dw_specs.strides;
    const int pipe_rows_produced_in_a_pass = PARALLELISM_PW_H / switching_layer_strides;
    const int rows_filled_to_produce_one_row = 2;
    for (int h = 1; h < layer_51_pw_specs.layer_ifm_height; h += pipe_rows_produced_in_a_pass)
    {
        fill_pipe_layer_input_buffer("/media/SSD2TB/wd/my_repos/DL_Benchmarking/tflite_scripts_imgnt_accuracy_and_weight_extraction/mob_v2/fms/ifms_12.txt",
                                     channels_buffer, h * rows_filled_to_produce_one_row, layer_12_pw_specs);

        pw_dw_conv(on_chip_pw_weights,
                   pipe_dw_weights_3x3,
                   channels_buffer,
                   result_buffer,
                   tmp_channels,
                   dw_pipe_overlap_buffer,
                   dw_channels_tile,
                   dw_channels_tile_copy,
                   h, // starting_h
                   layer_12_pw_specs,
                   layer_14_dw_specs,
                   pipe_fused_scales,
                   pipe_fused_scales_log_2_shifts,
                   pipe_relu_6_fused_scales,
                   pipe_fused_zero_points);

        write_pipe_seml_communication_buffer(
            result_buffer,
            result,
            h, // starting_h
            layer_12_pw_specs);
    }
}

#endif