import subprocess
import PySimpleGUI as sg
import serial

dump_dir = './out/raw/'
predictions_dir = './out/predictions/'
delimiter = "::"
response_keys = {'predictions': ['predictions_json'],
                 'time': ['fibha_average_time', 'fc_average_time'],
                 'power': ['pl_power_val', 'ps_power_val', 'mgt_power_val'],
                }

def run_command(report_energy, num_of_images, report_accuracy):
    ser = serial.Serial('/dev/ttyUSB0', 115200)

    cmd = "cd /media/sd-mmcblk0p1\r"
    ser.write(cmd.encode())
    cmd = "./fiba_v2 ./binary_container_1.xclbin {} {} {}\r".format(report_energy, num_of_images, report_accuracy)
    print(cmd)
    # ser.write(cmd.encode())

    # response = ''
    # while True:
    #     response_line = str(ser.readline())
    #     if '::EOF::' in str(response_line):
    #         break
        
    #     response += response_line

    # with open(dump_dir + 'out.txt', 'w') as f:
    #     f.write(response)

def read_response():
    response = ''
    with open(dump_dir + 'out.txt', 'r') as f:
        for line in f:
            response += line

    return response

def extract_data(response):
    running_time = 0
    energy = 0
    splits = response.split(delimiter)
    for i in range(len(splits) - 1):
        if splits[i] in response_keys['predictions']:
            with open(predictions_dir + 'predictions.json', 'w') as f:
                f.write(splits[i + 1])
            break
        elif splits[i] in response_keys['time']:
            running_time += float(splits[i+1])
        elif splits[i] in response_keys['power']:
            energy += float(splits[i+1])

    energy *= running_time / 1000

    return [running_time, energy]


def evaluate_accuracy():
    out_dict = {'top1': '', 'top5': ''}
    proc = subprocess.Popen(['python3', 'evaluate_accuracy.py',  './out/predictions/predictions.json'], 
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
            [sg.Text('Accelerator:', font= font), 
             sg.Combo(['FiBHA', 'FiBHA small', 'Baseline', 'Baseline small'], font=font,
                        readonly=False, key='fibha_combo', default_value='FiBHA'),
            sg.Text('Number of Images:', font= font), 
            sg.Combo([100, 200, 400, 800, 1000], font=font,
                        readonly=False, key='num_images_combo', default_value=100),
            sg.Checkbox('Report Energy', True, font= font, key='energy_check'),
             sg.Checkbox('Report accuracy', True, font= font, key='accuracy_check')] 
        ], font=font2, expand_x=True)

buttons_frame = [sg.Button('Run Accelerator', font=font2, expand_x=True), sg.Button('Close', font=font2)]

output_frame = sg.Frame('Outputs',[[results_table]], font=font2, expand_x=True)

layout = [ 
    [input_frame],
    buttons_frame,
    [output_frame]
]

# Create the Window
window = sg.Window('Window Title', layout)
# Event Loop to process "events" and get the "values" of the inputs
while True:
    event, values = window.read()
    if event == sg.WIN_CLOSED or event == 'Close': # if user closes window or clicks cancel
        break
    
    accelerator_instance = values['fibha_combo']
    report_energy = values['energy_check']
    report_accuracy = values['accuracy_check']
    num_of_images = values['num_images_combo']

    run_command(int(report_energy), num_of_images, int(report_accuracy))

    accuracy_vals = {}
    experiment_row = []
    experiment_row.append(accelerator_instance)

    response = read_response()
    running_time, energy = extract_data(response)
    
    experiment_row.append(running_time)
    
    if report_energy:
        experiment_row.append(energy)
    else:
        experiment_row.append('_')

    if report_accuracy:
        accuracy_vals = evaluate_accuracy()
        experiment_row.append(accuracy_vals['top1'])
        experiment_row.append(accuracy_vals['top5'])
    else:
        experiment_row.append('_')
        experiment_row.append('_')

    rows.append(experiment_row)
    results_table.update(rows)

window.close()

