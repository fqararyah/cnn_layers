
#if MODEL_ID == MOB_V2
#include "mob_v2_layers_specs.h"
#elif MODEL_ID == RESNET50
#include "resnet50_layers_specs.h"
#endif
#include "dw_weights.h"
#include "dw_weights_v2.h"
#include "on_chip_conv_pw_weights.h"
#include "on_chip_conv_weights_v2.h"
#include "on_chip_pw_weights_v2.h"
#include "quantization_and_biases.h"
#include "quantization_and_biases_v2.h"
#include "../pipelined_engines/pipelined_engines_specs.h"
#include "../pipelined_engines/pipeline_main.h"

#include "../SEML/headers/seml.h"