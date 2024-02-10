# import serial
# import os
# dump_dir = './out/raw/'
# predictions_dir = './out/predictions/'
# ser = serial.Serial('/dev/ttyUSB0', 115200)
# delimiter = "::"

# cmd = "cd /media/sd-mmcblk0p1\r"
# ser.write(cmd.encode())
# cmd="./fiba_v2 ./binary_container_1.xclbin 1 100 1\r"
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
#     else:
#         print(splits[i])

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

font = ("Arial", 12)
font2 = ("Arial", 15, 'bold')
sg.theme('Dark2')   # Add a touch of color
# All the stuff inside your window.

toprow = ['Accelerator', 'Latency (ms)', 'Energy (J / inference)', 'Top1 accuracy', 'Top5 accuracy']
rows = []
results_table = sg.Table(values=rows, headings=toprow,
   justification='center',
   key='results_table',
   expand_x=True,
   expand_y=True,
   font=font)

input_frame = sg.Frame('Inputs',[
            [sg.Text('Accelerator:', font= font), sg.Combo(['FiBHA', 'FiBHA small', 'Baseline', 'Baseline small'], font=font,
                        readonly=False, key='fibha_combo'),
            sg.Text('Number of Images:', font= font), sg.InputText(font=font, size=4),
            sg.Checkbox('Report Energy', True, font= font, key='check1'),
             sg.Checkbox('Report accuracy', True, font= font, key='check2')] 
        ], font=font2, expand_x=True)

buttons_frame = sg.Button('Run Accelerator', font=font2, expand_x=True), sg.Button('Close', font=font2)

output_frame = sg.Frame('Outputs',[[results_table]], font=font2, expand_x=True)

layout = [ 
    [input_frame],
    [buttons_frame],
    [output_frame]
]

# Create the Window
window = sg.Window('Window Title', layout)
# Event Loop to process "events" and get the "values" of the inputs
while True:
    event, values = window.read()
    if event == sg.WIN_CLOSED or event == 'Close': # if user closes window or clicks cancel
        break
    rows.append([values['fibha_combo'],0, 0, evaluate_accuracy()['top1'], evaluate_accuracy()['top5']])
    results_table.update(rows)

window.close()

