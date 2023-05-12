#ifndef PIPELINED_ENGINES
#define PIPELINED_ENGINES

#include "../../basic_defs/basic_defs_glue.h"
#include "../headers/model_glue.h"
#if MODEL_ID == MOB_V2
#include "../../model/headers/quantization_and_biases.h"
#include "../../model/headers/mob_v2_on_chip_weights_v2.h"
#include "../../model/headers/mob_v2_quantization_and_biases_v2.h"
#elif MODEL_ID == RESNET50
#include "../../model/headers/resnet50_quantization_and_biases_v2.h"
#endif
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
                                          const int buffer_size,
                                          const layer_specs layer_specs_struct);

    void load_pw_weights(weights_dt on_chip_weights[][ON_CHIP_WEIGHTS_PORTS],
                         weights_dt weights_tile[PARALLELISM_PW_OFMS][MAX_PW_BUFFER_DEPTH],
                         const int starting_filter,
                         layer_specs layer_specs_struct);

    void pw_conv_engine(weights_dt weights_tile[PARALLELISM_PW_OFMS][MAX_PW_BUFFER_DEPTH],
                        fms_dt channels[MAX_PW_BUFFER_DEPTH][PW_BUFFER_HEIGHT][MAX_PW_BUFFER_WIDTH],
                        pss_dt engine_result[PARALLELISM_PW_OFMS][PW_BUFFER_HEIGHT][PW_BUFFER_WIDTH],
                        const int starting_filter,
                        const int starting_w,
                        const layer_specs layer_specs_struct);

    void pw_normalize_engine_result(
        pss_dt engine_result_tile[PARALLELISM_PW_OFMS][PW_BUFFER_HEIGHT][PW_BUFFER_WIDTH],
        fms_dt normalized_tile[DW_TILE_DEPTH][DW_BUFFER_HEIGHT][DW_BUFFER_WIDTH],
        fms_dt dw_vertical_overlap_buffer[DW_TILE_DEPTH][DW_BUFFER_HEIGHT][MAX_FILTER_MINUS_STRIDES],
        const fms_quantization_scheme normalization_buffer[],
        const int starting_d, const int starting_h, const int starting_w,
        const layer_specs layer_specs_struct,
        const layer_specs dw_layer_specs_struct);

    void pw_only_normalize_engine_result(
        pss_dt engine_result_tile[PARALLELISM_PW_OFMS][PW_BUFFER_HEIGHT][PW_BUFFER_WIDTH],
        fms_dt normalized_tile[DW_TILE_DEPTH][DW_BUFFER_HEIGHT][DW_BUFFER_WIDTH],
        fms_dt tmp_channels[MAX_PW_BUFFER_DEPTH][PW_BUFFER_HEIGHT + 1][MAX_PW_BUFFER_WIDTH],
        const fms_quantization_scheme normalization_buffer[],
        const int starting_d, const int starting_h, const int starting_w,
        const layer_specs layer_specs_struct,
        const layer_specs dw_layer_specs_struct);

    void write_next_overlap_and_read_current(fms_dt dw_pipe_overlap_buffer[][DW_PIPE_OVERLAP_BUFFER_WIDTH],
                                             fms_dt dw_vertical_overlap_buffer[DW_TILE_DEPTH]
                                                                              [DW_BUFFER_HEIGHT][MAX_FILTER_MINUS_STRIDES],
                                             fms_dt dw_channels_tile[DW_TILE_DEPTH][DW_BUFFER_HEIGHT][DW_BUFFER_WIDTH],
                                             const int starting_d,
                                             const int starting_h,
                                             const int starting_w,
                                             layer_specs layer_specs_struct);

    void write_next_overlap_and_read_current_only_p2(
        fms_dt dw_pipe_overlap_buffer[][DW_PIPE_OVERLAP_BUFFER_WIDTH],
        fms_dt dw_vertical_overlap_buffer[DW_TILE_DEPTH][DW_BUFFER_HEIGHT][MAX_FILTER_MINUS_STRIDES],
        fms_dt dw_channels_tile[DW_TILE_DEPTH][DW_BUFFER_HEIGHT][DW_BUFFER_WIDTH],
        const int starting_d, const int starting_h, const int starting_w,
        layer_specs layer_specs_struct);

    void padd_left_dw_channels_tile(fms_dt dw_channels_tile[DW_TILE_DEPTH][DW_BUFFER_HEIGHT][DW_BUFFER_WIDTH],
                                    fms_dt dw_channels_tile_copy[DW_TILE_DEPTH][DW_BUFFER_HEIGHT][DW_BUFFER_WIDTH],
                                    layer_specs layer_specs_struct);

    void fill_dw_weights_tile(const dw_weights_dt weights[][MAX_DW_FILTER_AREA_IN_PIPE],
                              dw_weights_dt weights_tile[][MAX_DW_FILTER_AREA_IN_PIPE],
                              int starting_d, const int current_dw_layer_weights_offset);

    void dw_normalize_and_write_back_result_tile(dw_pss_dt result_tile[DW_TILE_DEPTH][PW_BUFFER_HEIGHT][PW_BUFFER_WIDTH],
                                                 fms_dt result[MAX_PW_BUFFER_DEPTH][PW_BUFFER_HEIGHT][MAX_PW_BUFFER_WIDTH],
                                                 const fms_quantization_scheme normalization_buffer[],
                                                 const int starting_d,
                                                 const int h_offset_in_result,
                                                 const int starting_w,
                                                 layer_specs layer_specs_struct);

    void dw_conv_engine(
        dw_weights_dt weights[DW_TILE_DEPTH][MAX_DW_FILTER_AREA_IN_PIPE],
        fms_dt channels_tile[DW_TILE_DEPTH][DW_BUFFER_HEIGHT][DW_BUFFER_WIDTH],
        dw_pss_dt result_tile[DW_TILE_DEPTH][PW_BUFFER_HEIGHT][PW_BUFFER_WIDTH],
        layer_specs layer_specs_struct);

    void pw_write_back_result_tile(
        fms_dt result_tile[DW_TILE_DEPTH][DW_BUFFER_HEIGHT][DW_BUFFER_WIDTH],
        fms_dt result[MAX_PW_BUFFER_DEPTH][PW_BUFFER_HEIGHT][MAX_PW_BUFFER_WIDTH],
        fms_dt tmp_channels[MAX_PW_BUFFER_DEPTH][PW_BUFFER_HEIGHT + 1][MAX_PW_BUFFER_WIDTH],
        const int starting_d, const int starting_w,
        const layer_specs layer_specs_struct);

    void pw_dw_conv(weights_dt on_chip_weights[][ON_CHIP_WEIGHTS_PORTS],
                    const dw_weights_dt weights[][3 * 3],
                    fms_dt channels[MAX_PW_BUFFER_DEPTH][PW_BUFFER_HEIGHT][MAX_PW_BUFFER_WIDTH],
                    fms_dt result[MAX_PW_BUFFER_DEPTH][PW_BUFFER_HEIGHT][MAX_PW_BUFFER_WIDTH],
                    fms_dt tmp_channels[MAX_PW_BUFFER_DEPTH][PW_BUFFER_HEIGHT + 1][MAX_PW_BUFFER_WIDTH],
                    fms_dt dw_pipe_overlap_buffer[][DW_PIPE_OVERLAP_BUFFER_WIDTH],
                    fms_dt dw_channels_tile[DW_TILE_DEPTH][DW_BUFFER_HEIGHT][DW_BUFFER_WIDTH],
                    fms_dt dw_channels_tile_copy[DW_TILE_DEPTH][DW_BUFFER_HEIGHT][DW_BUFFER_WIDTH],
                    const int starting_h,
                    const int h_offset_in_result,
                    bool fused_pw_dw,
                    const layer_specs pw_layer_specs_struct,
                    const layer_specs dw_layer_specs_struct,
                    const fused_scales_dt fused_scales[],
                    const fused_scales_log_2_shifts_dt fused_scales_log_2_shifts[],
                    const relu_6_fused_scales_dt relu_6_fused_scales[],
                    const biases_dt fused_zero_points[]);
}

#endif