# import serial
# import os
# dump_dir = './out/raw/'
# predictions_dir = './out/predictions/'
# ser = serial.Serial('/dev/ttyUSB0', 115200)
# delimiter = "::"

# cmd = "cd /media/sd-mmcblk0p1\r"
# ser.write(cmd.encode())
# cmd="./fiba_v2 ./binary_container_1.xclbin 1 10 1\r"
# ser.write(cmd.encode())

# response = ''
# while True:
#     response_line = str(ser.readline())
#     if '::EOF::' in str(response_line):
#         break
    
#     response += response_line

# with open(dump_dir + 'out.txt', 'w') as f:
#     f.write(response)

# splits = response.split(delimiter)
# for i in range(len(splits) - 1):
#     if 'predictions_json' in splits[i]:
#         with open(predictions_dir + 'out.json', 'w') as f:
#             f.write(splits[i + 1])
#         break

# os.system("python3 ./evaluate_accuracy.py ./out/predictions/out.json")


import subprocess
import PySimpleGUI as sg

def evaluate_accuracy():
    out_dict = {'top1': '', 'top5': ''}
    proc = subprocess.Popen(['python3', 'evaluate_accuracy.py',  './out/predictions/out.json'], 
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

    script_output = str(proc.communicate()[0]).lower().replace(' ', '')

    splits = script_output.split('::')

    for i in range(len(splits) - 1):
        if 'top1' in splits[i]:
            out_dict['top1'] = splits[i+1]
        elif 'top5' in splits[i]:
            out_dict['top5'] = splits[i+1]

    return out_dict

font = ("Arial", 15)
font2 = ("Arial", 15, 'bold')
sg.theme('Dark2')   # Add a touch of color
# All the stuff inside your window.
layout = [ 
    [
        sg.Frame('Inputs',[
            [sg.Text('Accelerator:', font= font), sg.Combo(['FiBHA', 'FiBHA small', 'Baseline', 'Baseline small'], font=font,
                        expand_x=True, readonly=False, key='fibha_combo')],
            [sg.Text('Number of Images:', font= font), sg.InputText(font=font, size=15)],
            [sg.Checkbox('Report Energy', True, font= font, key='check1'),
             sg.Checkbox('Report accuracy', True, font= font, key='check2')] 
        ], font=font2)
    ],
    [sg.Button('Run Accelerator', font=font2, expand_x=True), sg.Button('Close', font=font2)],
    [
        sg.Frame('Outputs',[
         [sg.Text('Latency (ms):', font= font, expand_x=True), sg.Text('', font= font, key='out_latenct', background_color='#eeeeee', size=9)],
         [sg.Text('Energy (J / inference):', font= font, expand_x=True), sg.Text('', font= font, key='out_energy', background_color='#eeeeee', size=9)],
         [sg.Text('Top1 accuracy:', font= font, expand_x=True), sg.Text('', font= font, key='out_top1_accuracy', background_color='#eeeeee', size=9)],
         [sg.Text('Top5 accuracy:', font= font, expand_x=True), sg.Text('', font= font, key='out_top5_accuracy', background_color='#eeeeee', size=9)],
         ], font=font2, expand_x=True)
    ] 
]

# Create the Window
window = sg.Window('Window Title', layout)
# Event Loop to process "events" and get the "values" of the inputs
while True:
    event, values = window.read()
    if event == sg.WIN_CLOSED or event == 'Close': # if user closes window or clicks cancel
        break
    window['out_top1_accuracy'].update(evaluate_accuracy()['top1'])
    window['out_top5_accuracy'].update(evaluate_accuracy()['top5'])

window.close()

