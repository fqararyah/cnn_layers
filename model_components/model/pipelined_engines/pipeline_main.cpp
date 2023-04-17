#include "pipelined_engines.h"

using namespace pipelined_engines;

void func()
{
    fms_dt channels[MAX_PW_BUFFER_DEPTH][MAX_PW_BUFFER_HEIGHT][MAX_PW_BUFFER_WIDTH];
    fms_dt result[MAX_PW_BUFFER_DEPTH][MAX_PW_BUFFER_HEIGHT][MAX_PW_BUFFER_WIDTH];
    fms_dt tmp_channels[PARALLELISM_PW_OFMS][PARALLELISM_PW_H][MAX_PW_BUFFER_WIDTH];
    fms_dt dw_pipe_overlap_buffer[DW_PIPE_OVERLAP_BUFFER_DEPTH][DW_PIPE_OVERLAP_BUFFER_WIDTH];
    fms_dt dw_channels_tile[DW_TILE_DEPTH][MAX_DW_BUFFER_HEIGHT][DW_TILE_WIDTH_PADDED];
    fms_dt dw_channels_tile_copy[DW_TILE_DEPTH][MAX_DW_BUFFER_HEIGHT][DW_TILE_WIDTH_PADDED];

    pipelined_engines::pw_dw_conv(on_chip_pw_weights,
                                  pipe_dw_weights_3x3,
                                  channels,
                                  result,
                                  tmp_channels,
                                  dw_pipe_overlap_buffer,
                                  dw_channels_tile,
                                  dw_channels_tile_copy,
                                  layer_4_pw_specs,
                                  layer_6_dw_specs,
                                  pipe_fused_scales,
                                  pipe_fused_scales_log_2_shifts,
                                  pipe_relu_6_fused_scales,
                                  pipe_fused_zero_points);
}