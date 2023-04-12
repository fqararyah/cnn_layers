
import sys
sys.path.append("/media/SSD2TB/wd/cnn_layers/")

from code_gen import code_generation_constants as cgc
from code_gen import utils


utils.set_globals(cgc.MODEL_NAME, cgc.MODEL_NAME)

PRECISION = 8

def calc_fms_memory(model_dag, starting_layer):
    current_layer = model_dag[starting_layer]
    ifms_shape = current_layer['ifms_shape']
    ifms_size = 1
    for i in ifms_shape:
        ifms_size *= i
    ofms_shape = current_layer['ofms_shape']
    ofms_size = 1
    for i in ofms_shape:
        ofms_size *= i
    ofms_size = ofms_shape[0] * ofms_shape[1] * ofms_shape[2]
    node_children = current_layer['children']
    ofms_copies = 1
    if len(node_children) > 1:
        ofms_size *= 2
        ofms_copies = 2

    total_mem = ifms_size + ofms_size
    max_mem = total_mem
    layers_participation_dict = {}
    layers_participation_dict[starting_layer] = {
        'ifms': ifms_size, 'ofms': ofms_size, 'ofms_copies': ofms_copies}

    for layer_index in range(starting_layer + 1, len(model_dag)):
        current_layer = model_dag[layer_index]
        ifms_shape = current_layer['ifms_shape']
        ifms_size = 1
        for i in ifms_shape:
            ifms_size *= i
        if 'pad' in current_layer['name']:#padding is inplace
            ifms_size = 0
        ofms_shape = current_layer['ofms_shape']
        ofms_size = 1
        for i in ofms_shape:
            ofms_size *= i

        node_children = current_layer['children']
        ofms_copies = 1
        if len(node_children) > 1:
            ofms_size *= 2
            ofms_copies = 2

        total_mem += ifms_size + ofms_size
        layers_participation_dict[layer_index] = {
            'ifms': ifms_size, 'ofms': ofms_size, 'ofms_copies': ofms_copies}

        for layer, participation in layers_participation_dict.items():
            if layer not in current_layer['parents']:
                continue
            parent_layer_children = model_dag[layer]['children']
            if participation['ifms'] > 0:
                total_mem -= participation['ifms']
                participation['ifms'] = 0
            if (participation['ofms_copies'] == 2 and len(parent_layer_children) > 1 and parent_layer_children[-2] == layer_index):
                total_mem -= participation['ofms'] / 2
                participation['ofms'] /= 2
                participation['ofms_copies'] = 1
            elif participation['ofms_copies'] == 1 and parent_layer_children[-1] == layer_index:
                total_mem -= participation['ofms']
                participation['ofms'] = 0
                participation['ofms_copies'] = 0

        if total_mem > max_mem:
            max_mem = total_mem

        _str = str(layer_index) + ':\n'
        for layer, participation in layers_participation_dict.items():
            if (participation['ifms'] != 0 or participation['ofms_copies'] != 0):
                _str += str(layer) + ': ('
            if participation['ifms'] != 0:
                _str += str(participation['ifms']) + ' + '
            if participation['ofms_copies'] != 0:
                _str += str(participation['ofms']) + \
                    ' * ' + str(participation['ofms_copies'])
            if (participation['ifms'] != 0 or participation['ofms_copies'] != 0):
                _str += ') + '

        _str += ('\n****************')
        #print(_str)

    return max_mem


model_dag = utils.read_model_dag()

conv_layer_index = 0
for i in range(0, 30):
    if 'type' in model_dag[i] and model_dag[i]['type'] in cgc.CONV_LAYER_TYPES:
        fms_mem = calc_fms_memory(model_dag, i)
        print(i, '->', conv_layer_index, ':', fms_mem * PRECISION / 8000000, 'MB')
        conv_layer_index += 1
