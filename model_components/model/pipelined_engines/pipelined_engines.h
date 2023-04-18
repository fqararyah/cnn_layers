#ifndef PIPELINED_ENGINES
#define PIPELINED_ENGINES

#include "../../basic_defs/basic_defs_glue.h"
#include "../headers/quantization_and_biases.h"
#include "../headers/model_glue.h"
#include "../headers/quantization_and_biases_v2.h"
#include "../../layers/headers/norm_act.h"
#include "pipelined_engines_specs.h"

namespace pipelined_engines
{

    void fill_fused_scales_and_zps_buffer(const fused_scales_dt fused_scales[],
                                          const fused_scales_log_2_shifts_dt fused_scales_log_2_shifts[],
                                          const relu_6_fused_scales_dt relu_6_fused_scales[],
                                          const biases_dt fused_zero_points[],
                                          fms_quantization_scheme normalization_buffer[],
                                          int starting_d,
                                          const int current_layer_fused_parameters_offset,
                                          const layer_specs layer_specs_struct);

    void load_pw_weights(const weights_dt pw_weights[],
                         weights_dt weights_tile[PARALLELISM_PW_OFMS][MAX_PW_BUFFER_DEPTH],
                         const int starting_filter,
                         layer_specs layer_specs_struct);

    void pw_conv_engine(weights_dt weights_tile[PARALLELISM_PW_OFMS][MAX_PW_BUFFER_DEPTH],
                        fms_dt channels[MAX_PW_BUFFER_DEPTH][MAX_PW_BUFFER_HEIGHT][MAX_PW_BUFFER_WIDTH],
                        pss_dt engine_result[PARALLELISM_PW_OFMS][PARALLELISM_PW_H][PARALLELISM_PW_W],
                        const int starting_filter,
                        const int starting_w,
                        const layer_specs layer_specs_struct);

    void pw_conv(const weights_dt pw_weights[],
                 fms_dt channels[MAX_PW_BUFFER_DEPTH][MAX_PW_BUFFER_HEIGHT][MAX_PW_BUFFER_WIDTH],
                 fms_dt result[PARALLELISM_PW_OFMS][PARALLELISM_PW_H][MAX_PW_BUFFER_WIDTH],
                 fms_dt tmp_channels[PARALLELISM_PW_OFMS][PARALLELISM_PW_H][MAX_PW_BUFFER_WIDTH],
                 int layer, const layer_specs layer_specs_struct,
                 const fused_scales_dt fused_scales[],
                 const fused_scales_log_2_shifts_dt fused_scales_log_2_shifts[],
                 const relu_6_fused_scales_dt relu_6_fused_scales[],
                 const biases_dt fused_zero_points[]);

    void pw_normalize_engine_result(pss_dt engine_result_tile[PARALLELISM_PW_OFMS][PARALLELISM_PW_H][PARALLELISM_PW_W],
                                    fms_dt normalized_tile[PARALLELISM_PW_OFMS][PARALLELISM_PW_H][PARALLELISM_PW_W],
                                    const fms_quantization_scheme normalization_buffer[],
                                    const layer_specs layer_specs_struct);

    void write_next_overlap_and_read_current(fms_dt dw_pipe_overlap_buffer[][DW_PIPE_OVERLAP_BUFFER_WIDTH],
                                             fms_dt normalized_tile[PARALLELISM_PW_OFMS][PARALLELISM_PW_H][PARALLELISM_PW_W],
                                             fms_dt dw_channels_tile[DW_TILE_DEPTH][MAX_DW_BUFFER_HEIGHT][DW_TILE_WIDTH_PADDED],
                                             const int starting_d_read,
                                             const int starting_d_write,
                                             const int starting_w,
                                             layer_specs layer_specs_struct);

    void fill_dw_weights_tile(const dw_weights_dt weights[][MAX_DW_FILTER_AREA_IN_PIPE],
                              dw_weights_dt weights_tile[][MAX_DW_FILTER_AREA_IN_PIPE],
                              int starting_d, const int current_dw_layer_weights_offset);

    void dw_conv_engine(
        dw_weights_dt weights[][MAX_DW_FILTER_AREA_IN_PIPE],
        fms_dt channels_tile[DW_TILE_DEPTH][MAX_DW_BUFFER_HEIGHT][DW_TILE_WIDTH_PADDED],
        dw_pss_dt result_tile[DW_TILE_DEPTH][MAX_PW_BUFFER_HEIGHT][DW_TILE_WIDTH],
        layer_specs layer_specs_struct);

    void dw_normalize_and_write_back_result_tile(dw_pss_dt result_tile[DW_TILE_DEPTH][MAX_PW_BUFFER_HEIGHT][DW_TILE_WIDTH],
                                                 fms_dt result[MAX_PW_BUFFER_DEPTH][MAX_PW_BUFFER_HEIGHT][MAX_PW_BUFFER_WIDTH],
                                                 const fms_quantization_scheme normalization_buffer[],
                                                 const int starting_d,
                                                 const int starting_w,
                                                 layer_specs layer_specs_struct);

    void pw_dw_conv(const weights_dt pw_weights[],
                    const dw_weights_dt weights[][3 * 3],
                    fms_dt channels[MAX_PW_BUFFER_DEPTH][MAX_PW_BUFFER_HEIGHT][MAX_PW_BUFFER_WIDTH],
                    fms_dt result[PARALLELISM_PW_OFMS][PARALLELISM_PW_H][MAX_PW_BUFFER_WIDTH],
                    fms_dt tmp_channels[PARALLELISM_PW_OFMS][PARALLELISM_PW_H][MAX_PW_BUFFER_WIDTH],
                    fms_dt dw_pipe_overlap_buffer[][DW_PIPE_OVERLAP_BUFFER_WIDTH],
                    fms_dt dw_channels_tile[DW_TILE_DEPTH][MAX_DW_BUFFER_HEIGHT][DW_TILE_WIDTH_PADDED],
                    fms_dt dw_channels_tile_copy[DW_TILE_DEPTH][MAX_DW_BUFFER_HEIGHT][DW_TILE_WIDTH_PADDED],
                    const layer_specs pw_layer_specs_struct,
                    const layer_specs dw_layer_specs_struct,
                    const fused_scales_dt fused_scales[],
                    const fused_scales_log_2_shifts_dt fused_scales_log_2_shifts[],
                    const relu_6_fused_scales_dt relu_6_fused_scales[],
                    const biases_dt fused_zero_points[]);

}
#endif