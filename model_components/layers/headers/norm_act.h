#ifndef NORM_ACT
#define NORM_ACT

#include "../../basic_defs/basic_defs_glue.h"

fms_dt pw_relu_norm(pss_dt pss, fms_quantization_scheme normalization, const int layer_relu);

pss_f_dt pw_relu_norm_no_q(pss_dt pss, fms_quantization_scheme normalization,
		const int layer_relu);

fms_dt dw_relu_norm(dw_pss_dt pss, fms_quantization_scheme normalization, const int layer_relu);

fms_dt conv_relu_norm(first_conv_pss_dt pss, fms_quantization_scheme normalization, const int layer_relu);

fms_dt clamp(pss_f_dt val);

#endif
