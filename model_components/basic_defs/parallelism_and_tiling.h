
#ifndef PARALLELISM_AND_TILING
#define PARALLELISM_AND_TILING

const int pw_tile_d = 2;
const int pw_tile_h = 8;
const int pw_tile_w = 8;
const int pw_tile_hw = pw_tile_h * pw_tile_w;
const int pw_tile_size = pw_tile_d * pw_tile_h * pw_tile_w;
const int pw_conv_parallelism_in = pw_tile_d;
//WARNING, when pw_conv_parallelism_out is changes, generate script should be run
const int pw_conv_parallelism_out = 8; //>= tile_d and >=8: 16, 32, 64 (< 8 is not working for weight load)

const int dw_tile_d = pw_tile_d;
const int dw_tile_h = pw_tile_h;
const int dw_tile_w = pw_tile_w;
const int dw_tile_hw = dw_tile_h * dw_tile_w;
const int dw_tile_size = dw_tile_d * dw_tile_h * dw_tile_w;
const int dw_max_v2_buffer_height = dw_tile_h + (3 - 1);//where 3 is max conv kernel dim and 1 is mi strides
const int dw_max_v2_buffer_width = dw_max_v2_buffer_height;//where 3 is max conv kernel dim and 1 is mi strides

const int max_dw_input_width = 112 + 1 + 1;//where 1 is max padding left and right
const int max_tile_w = pw_tile_w;
const int max_tile_h = pw_tile_h;
const int max_tile_d = pw_tile_d > dw_tile_d ? pw_tile_d : dw_tile_d;

const int fc_layer_parallelism = 128;
const int fc_layer_weights_partitioning_factor = fc_layer_parallelism/2;

const int num_of_weights_in_the_same_filter_and_group = weights_group_items / pw_conv_parallelism_out;
const int num_of_weight_groups_in_the_largest_weight_tile = max_conv_d * pw_conv_parallelism_out / weights_group_items;
const int pw_weights_tile_partitioning_factor = num_of_weights_in_the_same_filter_and_group;

const int main_buffers_partitining_factor = max_tile_d * pw_tile_w * pw_tile_h;

//const int median_number_of_tiles_in_depth = (median_depth / pw_tile_d);
const int median_dw_tiles_in_w = (median_width / dw_tile_w);

#endif
