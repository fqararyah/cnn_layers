#ifndef NORM_ACT
#define NORM_ACT

#include "../../basic_defs/basic_defs_glue.h"

fms_dt relu_norm(pss_dt pss, fms_quantization_scheme normalization,
				 const int layer_relu);

fms_dt pw_relu_norm_6_v2(pss_dt pss,
						 const biases_dt fused_zero_point,
						 const fms_dt ofm_zero_point,
						 const scales_dt fused_scales,
						 const relu_6_fused_scales_dt relu_6_fused_scale,
						 const int layer_relu);

fms_dt pw_relu_norm_6(pss_dt pss, fms_quantization_scheme normalization, const int layer_relu);

pss_f_dt pw_relu_norm_no_q_no_relu(pss_dt pss, fms_quantization_scheme normalization,
								   const int layer_relu);
pss_f_dt pw_relu_norm_no_q_no_relu_v2(pss_dt pss,
									  biases_dt fused_zero_point,
									  scales_dt fused_scale,
									  scales_dt ofm_scale);

fms_dt dw_relu_norm_v2(pss_dt pss,
					   const biases_dt fused_zero_point,
					   const fms_dt ofm_zero_point,
					   const scales_dt fused_scales,
					   const relu_6_fused_scales_dt relu_6_fused_scale,
					   const int layer_relu);
fms_dt dw_relu_norm(dw_pss_dt pss, fms_quantization_scheme normalization, const int layer_relu);

fms_dt conv_relu_norm(first_conv_pss_dt pss, fms_quantization_scheme normalization, const int layer_relu);
fms_dt conv_relu_norm_v2(pss_dt pss,
						 const biases_dt fused_zero_point,
						 const fms_dt ofm_zero_point,
						 const scales_dt fused_scales,
						 const relu_6_fused_scales_dt relu_6_fused_scale,
						 const int layer_relu);

fms_dt clamp(pss_f_dt val);
fms_dt clamp_cpu(float val);

#endif
