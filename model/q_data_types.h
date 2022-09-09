#include "ap_int.h"
#include "ap_fixed.h"

#ifndef Q_DATA_TYPES
#define Q_DATA_TYPES

//input_image
const int input_image_dt_width = 8;
const int input_image_dt_offset = input_image_dt_width - 1;
const int input_image_pss_dt_width = input_image_dt_width + 8 + 5; // 5 is ceil(log(27, 2))

//weights
const int weights_dt_width = 8;
const int weights_dt_offset = weights_dt_width - 1;
const int layer_0_weights_dt_width = 8;
const int layer_0_weights_dt_offset = layer_0_weights_dt_width - 1;
const int dw_weights_dt_width = 8;
const int dw_weights_dt_offset = dw_weights_dt_width - 1;

//fms
const int fms_dt_width = 8;
const int fms_dt_offset = fms_dt_width - 1;

//pss
const int pss_dt_width = weights_dt_width + fms_dt_width + 10; // 10 is log(1024, 2) since 1024 is the depth of the deepest filter
const int pss_dt_offset = pss_dt_width - 1;
const int first_conv_pss_width = fms_dt_width + layer_0_weights_dt_width + 4;
const int dw_pss_dt_width = dw_weights_dt_width + fms_dt_width + 4; // 4 is ceil(log(9, 2))
const int dw_pss_dt_offset = dw_pss_dt_width - 1;					// 11;
const int fc_weights_dt_width = 8;
const int fc_out_dt_width = fms_dt_width + fc_weights_dt_width + 11; //11 is ceil(log(fc_layer_input_size))

struct normalization_scheme {
	const ap_fixed<17, 12> zero_point;
	const ap_fixed<17, 12> ratio_pss_to_fms; // pow(2, fms_dt_width) / pow(2, dw_pss_dt_width);
};

typedef ap_int<layer_0_weights_dt_width> layer_0_weights_dt;
typedef ap_int<weights_dt_width> weights_dt;
typedef ap_int<dw_weights_dt_width> dw_weights_dt;
typedef ap_int<fms_dt_width> fms_dt;
typedef ap_int<pss_dt_width> pss_dt;	   // partial sums
typedef ap_int<dw_pss_dt_width> dw_pss_dt; // partial sums
typedef ap_int<first_conv_pss_width> first_conv_pss_dt;
typedef ap_uint<weights_group_items * weights_dt_width> weights_grp_dt;
typedef ap_uint<11> counters_dt;
typedef ap_uint<input_image_dt_width> input_image_dt;
typedef ap_uint<input_image_dt_width * input_image_group_items> input_image_grp_dt;
typedef ap_int<input_image_pss_dt_width> input_image_pss_dt;
typedef ap_int<fc_weights_dt_width> fc_weights_dt;
typedef ap_int<fc_out_dt_width> fc_out_dt;
#endif
