import code_generation_constants as cgc
import dw_weights_gen
import biases_and_quantization_gen
import prepare_off_chip_weights
import prepare_off_chip_weights_fpga
if cgc.FIBHA_VERSION == 1:
    import bott_arch_layers_specs_gen
    import bott_arch_seml_engine_calls_gen
elif cgc.FIBHA_VERSION == 2:
    import layer_specs_gen
    import bott_arch_seml_engine_calls_gen_v2
import on_chip_conv_and_pw_weights_gen
