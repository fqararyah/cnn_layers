
#if MODEL_ID == MOB_V2
#include "mob_v2_layers_specs.h"
#elif MODEL_ID == RESNET50
#include "resnet50_layers_specs.h"
#endif
#include "dw_weights.h"
#include "dw_weights_v2.h"
#if MODEL_ID == MOB_V2
#include "mob_v2_on_chip_weights_v2.h"
#include "../../model/headers/quantization_and_biases.h"
#include "../../model/headers/mob_v2_quantization_and_biases_v2.h"
#elif MODEL_ID == RESNET50
#include "../../model/headers/resnet50_quantization_and_biases_v2.h"
#endif
#include "../pipelined_engines/pipelined_engines_specs.h"
#include "../pipelined_engines/pipeline_main.h"

#include "../SEML/headers/seml.h"