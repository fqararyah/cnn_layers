#include "../../basic_defs/basic_defs_glue.h"

fms_dt pw_relu_norm(pss_dt pss, fms_quantization_scheme normalization, const int layer_relu = 6);

fms_dt dw_relu_norm(dw_pss_dt pss, fms_quantization_scheme normalization, const int layer_relu = 6);

fms_dt conv_relu_norm(first_conv_pss_dt pss, fms_quantization_scheme normalization, const int layer_relu = 6);
