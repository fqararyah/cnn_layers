import filecmp
import os
from pickle import FALSE

CACHING = FALSE

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
#...................................................................................
import code_generation_constants as cgc
import layer_specs_gen
print('layer_specs_gen done!')
import general_specs_gen
print('general_specs_gen done!')
# import fibha_seml_engine_calls_gen
# print('fibha_seml_engine_calls_gen done!')
import dw_weights_gen
print('dw_weights_gen done!')
import biases_and_quantization_gen_v3
print('biases_and_quantization_gen_v3 done!')
#...................................................................................
prepare_off_chip_weights_file = 'prepare_off_chip_weights.py'
prepare_off_chip_weights_cached_file = '__cached_prepare_off_chip_weights.py'

if not os.path.exists(prepare_off_chip_weights_cached_file):
    open(prepare_off_chip_weights_cached_file, 'x')

if filecmp.cmp(prepare_off_chip_weights_file, prepare_off_chip_weights_cached_file):
    print(bcolors.WARNING + ' prepare_off_chip_weights cached vesrion is up to date!!!')
else:
    import prepare_off_chip_weights
    print('prepare_off_chip_weights done!')
    
    # with open(prepare_off_chip_weights_file, 'r') as src, open(prepare_off_chip_weights_cached_file, 'w') as dst:
    #     for line in src:
    #         dst.write(line)
#...................................................................................
prepare_off_chip_weights_file = 'prepare_off_chip_weights_fpga.py'
prepare_off_chip_weights_cached_file = '__cached_prepare_off_chip_weights_fpga.py'

if not os.path.exists(prepare_off_chip_weights_cached_file):
    open(prepare_off_chip_weights_cached_file, 'x')
if filecmp.cmp(prepare_off_chip_weights_file, prepare_off_chip_weights_cached_file):
    print(bcolors.WARNING + ' prepare_off_chip_weights cached vesrion is up to date!!!')
else:
    import prepare_off_chip_weights_fpga
    print('prepare_off_chip_weights_fpga done!')

    # with open(prepare_off_chip_weights_file, 'r') as src, open(prepare_off_chip_weights_cached_file, 'w') as dst:
    #     for line in src:
    #         dst.write(line)
#...................................................................................

import on_chip_conv_and_pw_weights_gen
print('on_chip_conv_and_pw_weights_gen done!')
import on_chip_conv_and_pw_weights_gen_v2
print('on_chip_conv_and_pw_weights_gen_v2 done!')