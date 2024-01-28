#include "../model_components/basic_defs/simulation_constants.h"

#if HW == CPU
#ifndef MAIN_FILE
#define MAIN_FILE

#include "../model_components/model/headers/model_glue.h"
#include "../model_components/layers/headers/conv.h"
#if !ONLY_SEML
#include "../model_components/model/fused_bottlenecks/bottlenecks_glue.h"
#endif
//#include "../model_components/model/pipelined_engines/pipeline_main.h"

void top_func(
	fms_grp_dt input_image[input_image_depth * input_image_num_fms_groups_in_a_channel],
	weights_grp_dt off_chip_weights[all_off_chip_pw_s_weights],
	weights_dt off_chip_dw_weights[all_dw_off_chip_weights],
	fused_scales_dt off_chip_fused_scales[all_off_chip_fused_scales_zps],
	biases_dt off_chip_fused_zero_points[all_off_chip_fused_scales_zps],
	weights_grp_dt on_chip_weights_src[all_on_chip_pw_s_weights_groups],
	fms_dt fc_input[fc_layer_input_size],
	const int model_configs_list_src[2 * max_conv_layers],
	const soft_pipe_specs_struct soft_pipe_specs[max_conv_layers]);

#endif

#endif