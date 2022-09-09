
#ifndef SESL_SPECS
#define SESL_SPECS

// l1 (second and third layers)
const int _4_stages_layer_1_rows_at_once = 1;
const int _5_stages_layer_1_rows_at_once = 1;
const int _7_stages_layer_1_rows_at_once = 2;

// l2 (third and forth layers)
const int _5_stages_layer_2_rows_at_once = 1;
const int _4_stages_layer_2_rows_at_once = 1;
const int _7_stages_layer_2_rows_at_once = 2;

// l3 (fifth layer)
const int _5_stages_layer_3_rows_at_once = 1;
const int _4_stages_layer_3_rows_at_once = 1;
const int _7_stages_layer_3_rows_at_once = 2;

// l4 (sixth and seventh layers)
const int _7_stages_layer_4_rows_at_once = 2;

//*****variable based on experiment*****
const int layer_0_parallelism_ofms = 1;

const int layer_1_pw_parallelism_in = 32;
const int layer_1_pw_parallelism_out = 1;
const int layer_2_pw_parallelism_in = 32;
const int layer_2_pw_parallelism_out = 1;
const int layer_3_pw_parallelism_in = 16;
const int layer_3_pw_parallelism_out = 2;
const int layer_4_pw_parallelism_in = 96;
const int layer_4_pw_parallelism_out = 1;

const int layer_1_dw_parallelism = layer_1_pw_parallelism_out;
const int layer_2_dw_parallelism = layer_2_pw_parallelism_out;
const int layer_3_dw_parallelism = layer_3_pw_parallelism_out;

const int layer_1_pw_parallelism_in_input_partitioning_factor =
	layer_1_pw_parallelism_in / 2 >= layer_0_parallelism_ofms ? layer_1_pw_parallelism_in / 2 : layer_0_parallelism_ofms;
const int dw_layer_2_parallelism_input_partitioning_factor =
	layer_2_dw_parallelism / 2 >= layer_1_pw_parallelism_out ? layer_2_dw_parallelism / 2 : layer_1_pw_parallelism_out;
const int dw_layer_3_parallelism_input_partitioning_factor =
	layer_3_dw_parallelism / 2;

#endif