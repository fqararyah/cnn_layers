from fcntl import F_GETLEASE

FIBHA_VERSION = 2
FIBHA_VERSION_POSTFIX = '' if FIBHA_VERSION == 1 else '_v' + str(FIBHA_VERSION)
MODEL_NAME = 'mob_v2'
PIPELINE = True
PILELINE_LEN = 6
FIRST_LAYER_TO_GENERATE = PILELINE_LEN #PILELINE_LEN
LAST_LAYER_TO_GENERATE = -1
DEBUGGING = True
LAYERS_TO_DEBUG = [20]#[33,34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51]
#LAYERS_TO_DEBUG = [11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22,
#                   23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52]
START_CODE_GENERATION_SIGNAL = 'begin_code_generation'
END_CODE_GENERATION_SIGNAL = 'end_code_generation'
