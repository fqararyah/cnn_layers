#include "parallelism.h"

#if MODEL_ID == mob_v2 and PIPELINE_LENGTH == 0

#ifndef MODEL_GENERAL_SPECS_HEADER
#define MODEL_GENERAL_SPECS_HEADER

const int max_conv_d = 960 / alpha;
const int max_filter_hw_dim = 160;
const int max_std_conv_filter_hw_dim = 3;
const int max_padding_lr = 158;
const int fc_layer_input_size = 1280;
const int MAX_TMP_FMS_BUFFER_DEPTH = ((0 + CHANNELS_TILE_HEIGHT - 1) / CHANNELS_TILE_HEIGHT) * ((0 + CHANNELS_TILE_WIDTH - 1) / CHANNELS_TILE_WIDTH) * 0;
const int MAX_FMS_BUFFER_DEPTH = ((56 + CHANNELS_TILE_HEIGHT - 1) / CHANNELS_TILE_HEIGHT) * ((56 + CHANNELS_TILE_WIDTH - 1) / CHANNELS_TILE_WIDTH) * 24;
const int MAX_FILTER_DIM_STRIDE_1 = 160;
const int MAX_FILTER_DIM_STRIDE_2 = 96;
const int MAX_DW_LAYER_D = 960;
const int all_on_chip_pw_s_weights = 864;
const int all_dw_off_chip_weights = 64224;
const int all_off_chip_fused_scales_zps = 17024;

#endif

#endif
