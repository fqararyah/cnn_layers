#ifndef MAIN_FILE
#define MAIN_FILE

#include "../model_components/model/headers/model_glue.h"
#include "../model_components/basic_defs/basic_defs_glue.h"
#include "../model_components/layers/headers/norm_act.h"
#include "../model_components/layers/headers/pooling.h"
#include "../model_components/model/fused_bottlenecks/bottlenecks_glue.h"


void top_func(fms_grp_dt input_image[input_image_depth * input_image_num_fms_groups_in_a_channel],
		weights_grp_dt off_chip_weights[all_pw_s_weights], fms_dt fc_input[fc_layer_input_size],
		int *ready_to_receive_a_new_input_ptr);

#endif
