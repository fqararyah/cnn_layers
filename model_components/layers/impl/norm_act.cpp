#include "../headers/layers_imp_common_includes.h"
#include "../headers/norm_act.h"

fms_dt clamp(pss_f_dt val)
{
#pragma HLS INLINE
	fms_dt ret_val = (fms_dt)val;
	if (val > QUANTIZATION_MAX)
	{
		ret_val = QUANTIZATION_MAX;
	}
	if (val < QUANTIZATION_MIN)
	{
		ret_val = QUANTIZATION_MIN;
	}
	return ret_val;
}

fms_dt pw_relu_norm(pss_dt pss, fms_quantization_scheme normalization,
					const int layer_relu)
{
#pragma HLS INLINE
	pss += normalization.fused_zero_point;
	if (layer_relu == 6)
	{
		if (pss < 0)
		{
			pss = 0;
		}
		if (pss > normalization.relu_6_fused_scale)
		{
			pss = normalization.relu_6_fused_scale;
		}
	}
	pss_f_dt scaled_pss = pss * normalization.fused_scales;
	scaled_pss += normalization.ofm_zero_point;
	if (scaled_pss < 0)
	{
		scaled_pss -= quant_half;
	}
	else
	{
		scaled_pss += quant_half;
	}
	return clamp(scaled_pss);
}

pss_f_dt pw_relu_norm_no_q_no_relu(pss_dt pss, fms_quantization_scheme normalization,
						   const int layer_relu)
{
#pragma HLS INLINE
	pss += normalization.fused_zero_point;
	fused_scales_dt multiplier = normalization.fused_scales * normalization.ofm_scale;
	pss_f_dt scaled_pss = pss * multiplier;
	return scaled_pss;
}

fms_dt dw_relu_norm(dw_pss_dt pss, fms_quantization_scheme normalization,
					const int layer_relu)
{
#pragma HLS INLINE
	pss += normalization.fused_zero_point;
	if (layer_relu == 6)
	{
		if (pss < 0)
		{
			pss = 0;
		}
		if (pss > normalization.relu_6_fused_scale)
		{
			pss = normalization.relu_6_fused_scale;
		}
	}
	dw_pss_f_dt scaled_pss = pss * normalization.fused_scales;
	scaled_pss += normalization.ofm_zero_point;
	if (scaled_pss < 0)
	{
		scaled_pss -= quant_dw_half;
	}
	else
	{
		scaled_pss += quant_dw_half;
	}

	return clamp(scaled_pss);
}

fms_dt conv_relu_norm(first_conv_pss_dt pss,
					  fms_quantization_scheme normalization, const int layer_relu)
{
#pragma HLS INLINE
	pss += normalization.fused_zero_point;
	if (layer_relu == 6)
	{
		if (pss < 0)
		{
			pss = 0;
		}
		if (pss > normalization.relu_6_fused_scale)
		{
			pss = normalization.relu_6_fused_scale;
		}
	}
	pss_f_dt scaled_pss = pss * normalization.fused_scales;
	scaled_pss += normalization.ofm_zero_point;
	if (scaled_pss < 0)
	{
		scaled_pss -= quant_half;
	}
	else
	{
		scaled_pss += quant_half;
	}

	return clamp(scaled_pss);
}
