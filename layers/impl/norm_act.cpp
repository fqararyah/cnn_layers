#include "../headers/norm_act.h"

fms_dt pw_relu_norm(pss_dt pss, normalization_scheme normalization) {
#pragma HLS INLINE
	fms_dt scaled_val = (fms_dt) ((((ap_fixed<17, 12> ) pss)
			- normalization.zero_point) * normalization.ratio_pss_to_fms);
	if (scaled_val < 0) {
		scaled_val = 0;
	}
	return scaled_val;
}

fms_dt dw_relu_norm(dw_pss_dt pss, normalization_scheme normalization) {
#pragma HLS INLINE
	fms_dt scaled_val = (fms_dt) ((((ap_fixed<17, 12> ) pss)
			- normalization.zero_point) * normalization.ratio_pss_to_fms);
	if (scaled_val < 0) {
		scaled_val = 0;
	}
	return scaled_val;
}

fms_dt conv_relu_norm(first_conv_pss_dt pss,
		normalization_scheme normalization) {
#pragma HLS INLINE
	fms_dt scaled_val = (fms_dt) ((((ap_fixed<17, 12> ) pss)
			- normalization.zero_point) * normalization.ratio_pss_to_fms);
	if (scaled_val < 0) {
		scaled_val = 0;
	}
	return scaled_val;
}
