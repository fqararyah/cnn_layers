#ifndef BOTTLENECKS_CHAIN
#define BOTTLENECKS_CHAIN

struct bottlenecks_chain_specs
{
    int chain_input_depth;
    int chain_input_height;
    int chain_input_width;
    int chain_output_depth;
    int chain_output_height;
    int chain_output_width;
    int chain_output_num_tiles_d;
    int chain_output_num_tiles_h;
    int chain_output_num_tiles_w;

    int chain_max_filter_dim;
    int first_filter_dim;
    int chain_max_strides;
    int first_dw_padding_left;
    int first_dw_padding_right;
    int first_dw_padding_top;
    int first_dw_padding_bottom;
    int chain_max_rows_at_once;
    int chain_input_height;
};

const bottlenecks_chain_specs _4_6_chain = {
    layer_4_pw_depth,
    layer_4_pw_ifm_height,
    layer_4_pw_ifm_width,
    layer_6_pw_num_fils,
    layer_6_pw_ofm_height,
    layer_6_pw_ofm_width,
    (layer_6_pw_num_fils / pw_tile_d) + ((layer_6_pw_num_fils / pw_tile_d) != 0), 
    (layer_6_pw_ofm_height / pw_tile_h) + ((layer_6_pw_ofm_height % pw_tile_h) != 0), 
    (layer_6_pw_ofm_width / pw_tile_w) + ((layer_6_pw_ofm_width % pw_tile_w) != 0), 

    layer_5_dw_filter_size,
    layer_5_dw_filter_size,
    layer_5_dw_strides,
    layer_5_dw_padding_left,
    layer_5_dw_padding_right,
    layer_5_dw_padding_top,
    layer_5_dw_padding_left,
    1,
    2 // chain_max_rows_at_once * chain_max_strides
};

#endif