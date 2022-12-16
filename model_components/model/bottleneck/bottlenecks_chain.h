#ifndef BOTTLENECKS_CHAIN
#define BOTTLENECKS_CHAIN

const int chain_max_strides = 2;
const int chain_input_width = 112;
const int chain_input_depth = 16;
const int chain_output_width = 56;
const int chain_output_height = 56;

const int chain_max_filter_dim = 3;
const int chain_max_rows_at_once = 1;
const int chain_input_height = chain_max_rows_at_once * chain_max_strides;

#endif