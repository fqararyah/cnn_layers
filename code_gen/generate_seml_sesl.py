import code_generation_constants as cgc
import dw_weights_gen
print('dw_weights_gen done!')
import biases_and_quantization_gen_v2
print('biases_and_quantization_gen_v2 done!')
import prepare_off_chip_weights
print('prepare_off_chip_weights done!')
import prepare_off_chip_weights_fpga
print('prepare_off_chip_weights_fpga done!')
import layer_specs_gen
print('layer_specs_gen done!')
import fibha_seml_engine_calls_gen
print('fibha_seml_engine_calls_gen done!')
if cgc.FIBHA_VERSION == 1:
    import on_chip_conv_and_pw_weights_gen
    print('on_chip_conv_and_pw_weights_gen done!')
elif cgc.FIBHA_VERSION == 2:
    import on_chip_conv_and_pw_weights_gen_v2
    print('on_chip_conv_and_pw_weights_gen_v2 done!')