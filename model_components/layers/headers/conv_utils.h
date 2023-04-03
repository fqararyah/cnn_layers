
#ifndef CONV_UTILS
#define CONV_UTILS

#include "../../basic_defs/basic_defs_glue.h"
#include "../../model/headers/model_glue.h"

void padd_fms_tile_top_left(fms_dt channels[MAX_FMS_BUFFER_DEPTH][MIN_FMS_HEIGHT][MIN_FMS_WIDTH],
                            fms_dt padding_top_buffer[CHANNELS_PIPELINE_DEPTH][MAX_PADDING_TOP_LEFT][CHANNELS_TILE_WIDTH],
                            fms_dt padding_left_buffer[CHANNELS_PIPELINE_DEPTH][CHANNELS_TILE_HEIGHT][MAX_PADDING_TOP_LEFT],
                            fms_dt padding_top_left_buffer[CHANNELS_PIPELINE_DEPTH][MAX_PADDING_TOP_LEFT][MAX_PADDING_TOP_LEFT],
                            fms_dt padding_top_right_buffer[CHANNELS_PIPELINE_DEPTH][MAX_PADDING_TOP_LEFT][MAX_PADDING_TOP_LEFT],
                            fms_dt padding_bottom_left_buffer[CHANNELS_PIPELINE_DEPTH][MAX_PADDING_TOP_LEFT][MAX_PADDING_TOP_LEFT],
                            const int starting_d,
                            const int tile_in_h,
                            const int tile_in_w,
                            const int padding_top_left,
                            const int ifms_d,
                            const int num_of_ifm_tiles_h,
                            const int num_of_ifm_tiles_w,
                            fms_dt fms_zero_point);

void padd_fms_tile_bottom_right(fms_dt channels[MAX_FMS_BUFFER_DEPTH][MIN_FMS_HEIGHT][MIN_FMS_WIDTH],
                                fms_dt padding_bottom_buffer[CHANNELS_PIPELINE_DEPTH][MAX_PADDING_BOTTOM_RIGHT][CHANNELS_TILE_WIDTH],
                                fms_dt padding_right_buffer[CHANNELS_PIPELINE_DEPTH][CHANNELS_TILE_HEIGHT][MAX_PADDING_BOTTOM_RIGHT],
                                fms_dt padding_bottom_right_buffer[CHANNELS_PIPELINE_DEPTH][MAX_PADDING_BOTTOM_RIGHT][MAX_PADDING_BOTTOM_RIGHT],
                                const int starting_d,
                                const int tile_in_h,
                                const int tile_in_w,
                                const int padding_bottom_right,
                                const int ifms_d,
                                const int num_of_ifm_tiles_h,
                                const int num_of_ifm_tiles_w,
                                fms_dt fms_zero_point);

void fill_fms_tile(fms_dt channels[MAX_FMS_BUFFER_DEPTH][MIN_FMS_HEIGHT][MIN_FMS_WIDTH],
                   fms_dt padding_top_buffer[CHANNELS_PIPELINE_DEPTH][MAX_PADDING_TOP_LEFT][CHANNELS_TILE_WIDTH],
                   fms_dt padding_left_buffer[CHANNELS_PIPELINE_DEPTH][CHANNELS_TILE_HEIGHT][MAX_PADDING_TOP_LEFT],
                   fms_dt padding_top_left_buffer[CHANNELS_PIPELINE_DEPTH][MAX_PADDING_TOP_LEFT][MAX_PADDING_TOP_LEFT],
                   fms_dt padding_top_right_buffer[CHANNELS_PIPELINE_DEPTH][MAX_PADDING_TOP_LEFT][MAX_PADDING_TOP_LEFT],
                   fms_dt padding_bottom_left_buffer[CHANNELS_PIPELINE_DEPTH][MAX_PADDING_TOP_LEFT][MAX_PADDING_TOP_LEFT],
                   fms_dt padding_bottom_buffer[CHANNELS_PIPELINE_DEPTH][MAX_PADDING_BOTTOM_RIGHT][CHANNELS_TILE_WIDTH],
                   fms_dt padding_right_buffer[CHANNELS_PIPELINE_DEPTH][CHANNELS_TILE_HEIGHT][MAX_PADDING_BOTTOM_RIGHT],
                   fms_dt padding_bottom_right_buffer[CHANNELS_PIPELINE_DEPTH][MAX_PADDING_BOTTOM_RIGHT][MAX_PADDING_BOTTOM_RIGHT],
                   fms_dt channels_tile[CHANNELS_TILE_HEIGHT_PADDED][CHANNELS_TILE_WIDTH_PADDED],
                   const int tile_in_d,
                   const int tile_in_h,
                   const int tile_in_w,
                   const int num_of_ifm_tiles_h,
                   const int num_of_ifm_tiles_w,
                   const int padding_top_left);

void fill_fms_tile_dw(fms_dt channels[MAX_FMS_BUFFER_DEPTH][MIN_FMS_HEIGHT][MIN_FMS_WIDTH],
                      fms_dt padding_top_buffer[CHANNELS_PIPELINE_DEPTH][MAX_PADDING_TOP_LEFT][CHANNELS_TILE_WIDTH],
                      fms_dt padding_left_buffer[CHANNELS_PIPELINE_DEPTH][CHANNELS_TILE_HEIGHT][MAX_PADDING_TOP_LEFT],
                      fms_dt padding_top_left_buffer[CHANNELS_PIPELINE_DEPTH][MAX_PADDING_TOP_LEFT][MAX_PADDING_TOP_LEFT],
                      fms_dt padding_top_right_buffer[CHANNELS_PIPELINE_DEPTH][MAX_PADDING_TOP_LEFT][MAX_PADDING_TOP_LEFT],
                      fms_dt padding_bottom_left_buffer[CHANNELS_PIPELINE_DEPTH][MAX_PADDING_TOP_LEFT][MAX_PADDING_TOP_LEFT],
                      fms_dt padding_bottom_buffer[CHANNELS_PIPELINE_DEPTH][MAX_PADDING_BOTTOM_RIGHT][CHANNELS_TILE_WIDTH],
                      fms_dt padding_right_buffer[CHANNELS_PIPELINE_DEPTH][CHANNELS_TILE_HEIGHT][MAX_PADDING_BOTTOM_RIGHT],
                      fms_dt padding_bottom_right_buffer[CHANNELS_PIPELINE_DEPTH][MAX_PADDING_BOTTOM_RIGHT][MAX_PADDING_BOTTOM_RIGHT],
                      fms_dt channels_tile[CHANNELS_PIPELINE_DEPTH][CHANNELS_TILE_HEIGHT_PADDED][CHANNELS_TILE_WIDTH_PADDED],
                      conv_type layer_type,
                      const int starting_d,
                      const int tile_in_h,
                      const int tile_in_w,
                      const int padding_top_left,
                      const int ifms_d,
                      const int num_of_ifm_tiles_h,
                      const int num_of_ifm_tiles_w,
                      fms_dt fms_zero_point);

void copy_fms_tile_corners(fms_dt padding_top_buffer[CHANNELS_PIPELINE_DEPTH][MAX_PADDING_TOP_LEFT][CHANNELS_TILE_WIDTH],
                           fms_dt padding_left_buffer[CHANNELS_PIPELINE_DEPTH][CHANNELS_TILE_HEIGHT][MAX_PADDING_TOP_LEFT],
                           fms_dt padding_top_left_buffer[CHANNELS_PIPELINE_DEPTH][MAX_PADDING_TOP_LEFT][MAX_PADDING_TOP_LEFT],
                           fms_dt padding_top_right_buffer[CHANNELS_PIPELINE_DEPTH][MAX_PADDING_TOP_LEFT][MAX_PADDING_TOP_LEFT],
                           fms_dt padding_bottom_left_buffer[CHANNELS_PIPELINE_DEPTH][MAX_PADDING_TOP_LEFT][MAX_PADDING_TOP_LEFT],
                           fms_dt padding_bottom_buffer[CHANNELS_PIPELINE_DEPTH][MAX_PADDING_BOTTOM_RIGHT][CHANNELS_TILE_WIDTH],
                           fms_dt padding_right_buffer[CHANNELS_PIPELINE_DEPTH][CHANNELS_TILE_HEIGHT][MAX_PADDING_BOTTOM_RIGHT],
                           fms_dt padding_bottom_right_buffer[CHANNELS_PIPELINE_DEPTH][MAX_PADDING_BOTTOM_RIGHT][MAX_PADDING_BOTTOM_RIGHT],
                           fms_dt padding_top_buffer_dst[CHANNELS_PIPELINE_DEPTH][MAX_PADDING_TOP_LEFT][CHANNELS_TILE_WIDTH],
                           fms_dt padding_left_buffer_dst[CHANNELS_PIPELINE_DEPTH][CHANNELS_TILE_HEIGHT][MAX_PADDING_TOP_LEFT],
                           fms_dt padding_top_left_buffer_dst[CHANNELS_PIPELINE_DEPTH][MAX_PADDING_TOP_LEFT][MAX_PADDING_TOP_LEFT],
                           fms_dt padding_top_right_buffer_dst[CHANNELS_PIPELINE_DEPTH][MAX_PADDING_TOP_LEFT][MAX_PADDING_TOP_LEFT],
                           fms_dt padding_bottom_left_buffer_dst[CHANNELS_PIPELINE_DEPTH][MAX_PADDING_TOP_LEFT][MAX_PADDING_TOP_LEFT],
                           fms_dt padding_bottom_buffer_dst[CHANNELS_PIPELINE_DEPTH][MAX_PADDING_BOTTOM_RIGHT][CHANNELS_TILE_WIDTH],
                           fms_dt padding_right_buffer_dst[CHANNELS_PIPELINE_DEPTH][CHANNELS_TILE_HEIGHT][MAX_PADDING_BOTTOM_RIGHT],
                           fms_dt padding_bottom_right_buffer_dst[CHANNELS_PIPELINE_DEPTH][MAX_PADDING_BOTTOM_RIGHT][MAX_PADDING_BOTTOM_RIGHT],
                           const int starting_d,
                           const int ifms_d,
                           const int padding_top_left,
                           const int padding_bottom_right);

#endif