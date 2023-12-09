#ifndef MAIN_FILE
#define MAIN_FILE

#include "../model_components/model/headers/model_glue.h"
#if !ONLY_SEML
#include "../model_components/model/fused_bottlenecks/bottlenecks_glue.h"
#endif
//#include "../model_components/model/pipelined_engines/pipeline_main.h"

void top_func(
	fms_grp_dt input_image[input_image_depth * input_image_num_fms_groups_in_a_channel],
	weights_grp_dt off_chip_weights[all_pw_s_weights],
	weights_grp_dt on_chip_weights_src[all_on_chip_pw_s_weights],
	fms_dt fc_input[fc_layer_input_size],
	const int model_configs_list_src[2 * max_conv_layers]);

#endif
