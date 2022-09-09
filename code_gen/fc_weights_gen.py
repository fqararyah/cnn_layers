import utils

utils.set_globals('mob_v2', 'mobilenetv2')

bit_width = 8

fc_layer_specs_file = 'fc_layer.txt'
fc_weights_h_file = './out/fc_weights.h'

fc_rows = 0
fc_cols = 0
with open(utils.input_folder + fc_layer_specs_file, 'r') as f:
    for line in f:
        if 'rows' in line:
            fc_rows = int(line.replace(' ', '').replace(
                '\n', '').split(utils.DELIMITER)[1])
        if 'cols' in line:
            fc_cols = int(line.replace(' ', '').replace(
                '\n', '').split(utils.DELIMITER)[1])

fc_weights_declaration_string = 'const static fc_weights_dt fc_weights[' + str(
    fc_rows) + ']' + '[' + str(fc_cols) + ']= {\n'

with open(fc_weights_h_file, 'w') as f:
    f.write('#ifndef FC_WEIGHTS\n#define FC_WEIGHTS\n')
    f.write(fc_weights_declaration_string)
    for i in range(fc_rows):
        f.write('{\n')
        for j in range(fc_cols):
            f.write (str(-1 + ((i*j) % (2**(bit_width-1)))))
            if j < fc_cols - 1:
                f.write(',')
        f.write('\n}')
        if(i < fc_rows - 1):
            f.write(',\n')
    f.write('};\n')

    f.write('#endif\n')
