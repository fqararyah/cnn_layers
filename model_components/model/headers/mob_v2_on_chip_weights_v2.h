#include "../../basic_defs/basic_defs_glue.h"
#include "mob_v2_layers_specs.h"
#if FIRST_PART_IMPLEMENTATION ==PIPELINED_ENGINES_MODE

#if PIPELINE_LENGTH == 0
#include "mob_v2_on_chip_weights_v2_pipe_0.h"
#elif PIPELINE_LENGTH == 6
#include "mob_v2_on_chip_weights_v2_pipe_6.h"
#endif

#endif