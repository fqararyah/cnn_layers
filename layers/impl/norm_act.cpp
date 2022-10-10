#include "../headers/norm_act.h"

fms_dt pw_relu_norm(pss_dt pss, fms_quantization_scheme normalization, const int layer_relu) {
#pragma HLS INLINE
	fms_dt scaled_val = (fms_dt) (fms_dt) ( normalization.fused_scales * (pss - normalization.fused_zero_points ) + normalization.ofm_zero_point;
	if(layer_relu == 6){
		if (scaled_val < 0) {
			scaled_val = 0;
		}
		if(scaled_val > 6){
			scaled_val = 6;
		}
	}
	return scaled_val;
}

fms_dt dw_relu_norm(dw_pss_dt pss, fms_quantization_scheme normalization, const int layer_relu) {
#pragma HLS INLINE
	fms_dt scaled_val = (fms_dt) (fms_dt) ( normalization.fused_scales * (pss - normalization.fused_zero_points ) + normalization.ofm_zero_point;
	if(layer_relu == 6){
		if (scaled_val < 0) {
			scaled_val = 0;
		}
		if(scaled_val > 6){
			scaled_val = 6;
		}
	}
	return scaled_val;
}

fms_dt conv_relu_norm(first_conv_pss_dt pss,
		fms_quantization_scheme normalization, const int layer_relu = 6) {
#pragma HLS INLINE
	fms_dt scaled_val = (fms_dt) ( normalization.fused_scales * (pss - normalization.fused_zero_points ) + normalization.ofm_zero_point;
	if(layer_relu == 6){
		if (scaled_val < 0) {
			scaled_val = 0;
		}
		if(scaled_val > 6){
			scaled_val = 6;
		}
	}
	return scaled_val;
}
