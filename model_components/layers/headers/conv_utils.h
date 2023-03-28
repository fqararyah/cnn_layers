
#ifndef CONV_UTILS
#define CONV_UTILS

#include "../../basic_defs/basic_defs_glue.h"
#include "../../model/headers/model_glue.h"

void fill_fms_tile(fms_dt channels[MAX_FMS_BUFFER_DEPTH][MIN_FMS_HEIGHT][MIN_FMS_WIDTH],
                   fms_dt channels_tile[CHANNELS_TILE_DEPTH][CHANNELS_TILE_HEIGHT][CHANNELS_TILE_WIDTH],
                   conv_type layer_type,
                   const int starting_d,
                   const int padding_top_left,
                   const int num_of_ifm_tiles_d,
                   const int num_of_ifm_tiles_h,
                   const int num_of_ifm_tiles_w,
                   fms_dt fms_zero_point);

#endif