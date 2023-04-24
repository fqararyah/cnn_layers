#ifndef MAIN_FILE
#define MAIN_FILE

#include "../model_components/model/headers/model_glue.h"
#include "../model_components/model/fused_bottlenecks/bottlenecks_glue.h"
//#include "../model_components/model/pipelined_engines/pipeline_main.h"

void top_func(fms_grp_dt input_image[input_image_depth * input_image_num_fms_groups_in_a_channel],
		weights_grp_dt off_chip_weights[all_pw_weights], fms_dt fc_input[fc_layer_input_size],
		int *ready_to_receive_a_new_input_ptr);

#endif
