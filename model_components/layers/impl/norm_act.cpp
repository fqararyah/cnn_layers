#include "../headers/layers_imp_common_includes.h"
#include "../headers/norm_act.h"

fms_dt clamp(pss_f_dt val)
{
#pragma HLS INLINE
	scaled_but_unclamped_fms_dt ret_val = (scaled_but_unclamped_fms_dt)val;
	if (val > QUANTIZATION_MAX)
	{
		ret_val = QUANTIZATION_MAX;
	}
	else if (val < QUANTIZATION_MIN)
	{
		ret_val = QUANTIZATION_MIN;
	}
	return (fms_dt)ret_val;
}

fms_dt pw_relu_norm_6(pss_dt pss, fms_quantization_scheme normalization,
					  const int layer_relu)
{
#pragma HLS INLINE

	pss += normalization.fused_zero_point;

	if (layer_relu == 6 && pss <= 0)
	{
		return normalization.ofm_zero_point;
	}

	pss_f_dt scaled_pss = pss * normalization.fused_scales;
	if (layer_relu != 6 || scaled_pss <= normalization.relu_6_fused_scale)
	{
		scaled_pss += normalization.ofm_zero_point;
		scaled_pss += quant_half - (scaled_pss < 0);
		return clamp(scaled_pss);
	}

	return clamp(normalization.ofm_zero_point + normalization.relu_6_fused_scale + quant_half - (scaled_pss < 0));
}

fms_dt pw_relu_norm_6_v2(pss_dt pss,
						 const biases_dt fused_zero_point,
						 const fms_dt ofm_zero_point,
						 const scales_dt fused_scales,
						 const relu_6_fused_scales_dt relu_6_fused_scale,
						 const int layer_relu)
{
#pragma HLS INLINE

	pss += fused_zero_point;

	if (layer_relu == 6 && pss <= 0)
	{
		return ofm_zero_point;
	}

	pss_f_dt scaled_pss = pss * fused_scales;
	if (layer_relu == 6 && scaled_pss > relu_6_fused_scale)
	{
		scaled_pss = relu_6_fused_scale;
	}

	scaled_pss += ofm_zero_point;
	scaled_pss += quant_half - (scaled_pss < 0);
	return clamp(scaled_pss);
}

fms_dt pw_relu_norm_6_v1(pss_dt pss, fms_quantization_scheme normalization,
						 const int layer_relu)
{
#pragma HLS INLINE

	norm_act_pss_dt na_pss = pss + normalization.fused_zero_point;
	if (layer_relu == 6)
	{
		if (na_pss < 0)
		{
			na_pss = 0;
		}
		if (na_pss > normalization.relu_6_fused_scale)
		{
			na_pss = normalization.relu_6_fused_scale;
		}
	}
	pss_f_dt scaled_pss = na_pss * normalization.fused_scales;
	scaled_pss = scaled_pss / (1 << normalization.fused_scales_log_2_shift);
	scaled_pss += normalization.ofm_zero_point;

	scaled_pss = scaled_pss + quant_half - (scaled_pss < 0);

	return clamp(scaled_pss);
}

fms_dt relu_norm(pss_dt pss, fms_quantization_scheme normalization,
				 const int layer_relu)
{
#pragma HLS INLINE
	norm_act_pss_dt na_pss = pss + normalization.fused_zero_point;
	if (layer_relu && na_pss < 0)
	{
		na_pss = 0;
	}
	pss_f_dt scaled_pss = na_pss * normalization.fused_scales;
	scaled_pss = scaled_pss / (1 << normalization.fused_scales_log_2_shift);
	scaled_pss += normalization.ofm_zero_point;

	scaled_pss = scaled_pss + quant_half - (scaled_pss < 0);

	return clamp(scaled_pss);
}

pss_f_dt pw_relu_norm_no_q_no_relu(pss_dt pss,
								   fms_quantization_scheme normalization, const int layer_relu)
{
#pragma HLS INLINE

	norm_act_pss_dt na_pss = pss + normalization.fused_zero_point;
	fused_scales_dt multiplier = normalization.fused_scales * normalization.ofm_scale;
	pss_f_dt scaled_pss = na_pss * multiplier;

	return scaled_pss;
}

pss_f_dt pw_relu_norm_no_q_no_relu_v2(pss_dt pss,
									  biases_dt fused_zero_point,
									  scales_dt fused_scale,
									  scales_dt ofm_scale,
									  const int layer_relu)
{
#pragma HLS INLINE

	norm_act_pss_dt na_pss = pss + fused_zero_point;
	fused_scales_dt multiplier = fused_scale * ofm_scale;
	pss_f_dt scaled_pss = na_pss * multiplier;

	return scaled_pss;
}

fms_dt dw_relu_norm(dw_pss_dt pss, fms_quantization_scheme normalization,
					const int layer_relu)
{
#pragma HLS INLINE

	pss += normalization.fused_zero_point;

	if (pss <= 0)
	{
		return normalization.ofm_zero_point;
	}

	pss_f_dt scaled_pss = pss * normalization.fused_scales;
	if (scaled_pss <= normalization.relu_6_fused_scale)
	{
		scaled_pss += normalization.ofm_zero_point;
		scaled_pss += quant_half - (scaled_pss < 0);
		return clamp((fms_dt)scaled_pss);
	}

	return clamp((fms_dt)(normalization.ofm_zero_point + normalization.relu_6_fused_scale));
}

fms_dt dw_relu_norm_v2(pss_dt pss,
					   const biases_dt fused_zero_point,
					   const fms_dt ofm_zero_point,
					   const scales_dt fused_scales,
					   const relu_6_fused_scales_dt relu_6_fused_scale,
					   const int layer_relu)
{
#pragma HLS INLINE

	pss += fused_zero_point;

	if (pss <= 0)
	{
		return ofm_zero_point;
	}

	pss_f_dt scaled_pss = pss * fused_scales;
	if (scaled_pss > relu_6_fused_scale)
	{
		scaled_pss = relu_6_fused_scale;
	}

	scaled_pss += ofm_zero_point;
	scaled_pss += quant_half - (scaled_pss < 0);
	return clamp(scaled_pss);
}

fms_dt dw_relu_norm_v1(dw_pss_dt pss, fms_quantization_scheme normalization,
					   const int layer_relu)
{
#pragma HLS INLINE

	norm_act_pss_dt na_pss = pss + normalization.fused_zero_point;
	if (layer_relu == 6)
	{
		if (na_pss < 0)
		{
			na_pss = 0;
		}
		if (na_pss > normalization.relu_6_fused_scale)
		{
			na_pss = normalization.relu_6_fused_scale;
		}
	}

	dw_pss_f_dt scaled_pss = na_pss * normalization.fused_scales;
	scaled_pss = scaled_pss / (1 << normalization.fused_scales_log_2_shift);
	scaled_pss += normalization.ofm_zero_point;

	scaled_pss = scaled_pss + quant_half - (scaled_pss < 0);

	return clamp(scaled_pss);
}

fms_dt conv_relu_norm_v1(first_conv_pss_dt pss,
						 fms_quantization_scheme normalization, const int layer_relu)
{
#pragma HLS INLINE

	layer_0_norm_act_pss_dt na_pss = pss + normalization.fused_zero_point;
	if (layer_relu == 6)
	{
		if (na_pss < 0)
		{
			na_pss = 0;
		}
		if (na_pss > normalization.layer_0_relu_6_fused_scale)
		{
			na_pss = normalization.layer_0_relu_6_fused_scale;
		}
	}

	pss_f_dt scaled_pss = na_pss * normalization.fused_scales;
	scaled_pss = scaled_pss / (1 << normalization.fused_scales_log_2_shift);
	scaled_pss += normalization.ofm_zero_point;

	scaled_pss = scaled_pss + quant_half - (scaled_pss < 0);

	return clamp(scaled_pss);
}

fms_dt conv_relu_norm(first_conv_pss_dt pss,
					  fms_quantization_scheme normalization, const int layer_relu)
{
#pragma HLS INLINE

	pss += normalization.fused_zero_point;

	if (layer_relu == 6 && pss <= 0)
	{
		return normalization.ofm_zero_point;
	}

	pss_f_dt scaled_pss = pss * normalization.fused_scales;
	if (layer_relu != 6 || scaled_pss <= normalization.layer_0_relu_6_fused_scale)
	{
		scaled_pss += normalization.ofm_zero_point;
		scaled_pss += quant_half - (scaled_pss < 0);
		return clamp((fms_dt)scaled_pss);
	}

	return clamp((fms_dt)(normalization.ofm_zero_point + normalization.layer_0_relu_6_fused_scale));
}

fms_dt conv_relu_norm_v2(pss_dt pss,
						 const biases_dt fused_zero_point,
						 const fms_dt ofm_zero_point,
						 const scales_dt fused_scales,
						 const relu_6_fused_scales_dt relu_6_fused_scale,
						 const int layer_relu)
{
#pragma HLS INLINE

	pss += fused_zero_point;

	if (layer_relu == 6 && pss <= 0)
	{
		return ofm_zero_point;
	}

	pss_f_dt scaled_pss = pss * fused_scales;
	if (layer_relu != 6 || scaled_pss <= relu_6_fused_scale)
	{
		scaled_pss += ofm_zero_point;
		scaled_pss += quant_half - (scaled_pss < 0);
		return clamp((fms_dt)scaled_pss);
	}

	return clamp((fms_dt)(ofm_zero_point + relu_6_fused_scale));
}