#include "../headers/norm_act.h"
#include  "../../client/quantization_and_biases.h"

fms_dt pw_relu_norm(pss_dt pss, fms_quantization_scheme normalization,
		const int layer_relu) {
#pragma HLS INLINE
	pss_f_dt scaled_pss = (pss_f_dt) (normalization.fused_scales
			* (pss + normalization.fused_zero_point));
	if (layer_relu == 6) {
		if (scaled_pss < 0) {
			scaled_pss = 0;
		}
		if (scaled_pss > 6) {
			scaled_pss = 6;
		}
	}
	scaled_pss /= normalization.ofm_scale;
	scaled_pss += normalization.ofm_zero_point;
	if (scaled_pss < 0) {
		scaled_pss -= quant_half;
	} else {
		scaled_pss += quant_half;
	}
	fms_dt scaled_val = (fms_dt) scaled_pss;
	return scaled_val;
}

fms_dt dw_relu_norm(dw_pss_dt pss, fms_quantization_scheme normalization,
		const int layer_relu) {
#pragma HLS INLINE
	dw_pss_f_dt scaled_pss = (dw_pss_f_dt) (normalization.fused_scales
			* (pss + normalization.fused_zero_point));
	if (layer_relu == 6) {
		if (scaled_pss < 0) {
			scaled_pss = 0;
		}
		if (scaled_pss > 6) {
			scaled_pss = 6;
		}
	}
	scaled_pss /= normalization.ofm_scale;
	scaled_pss += normalization.ofm_zero_point;
	if (scaled_pss < 0) {
		scaled_pss -= quant_dw_half;
	} else {
		scaled_pss += quant_dw_half;
	}
	fms_dt scaled_val = (fms_dt) scaled_pss;
	return scaled_val;
}

fms_dt conv_relu_norm(first_conv_pss_dt pss,
		fms_quantization_scheme normalization, const int layer_relu) {
#pragma HLS INLINE
	pss_f_dt scaled_pss = (pss_f_dt) (normalization.fused_scales
			* (pss + normalization.fused_zero_point));
	if (layer_relu == 6) {
		if (scaled_pss < 0) {
			scaled_pss = 0;
		}
		if (scaled_pss > 6) {
			scaled_pss = 6;
		}
	}
	scaled_pss /= normalization.ofm_scale;
	scaled_pss += normalization.ofm_zero_point;
	if (scaled_pss < 0) {
		scaled_pss -= quant_half;
	} else {
		scaled_pss += quant_half;
	}
	fms_dt scaled_val = (fms_dt) scaled_pss;
	return scaled_val;
}
