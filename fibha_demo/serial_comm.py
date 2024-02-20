import subprocess
import PySimpleGUI as sg
import serial
import sys
import time

dump_dir = './out/raw/'
predictions_dir = './out/predictions/'
delimiter = "::"
response_keys = {'predictions': ['predictions_json'],
                 'time': ['fibha_average_time', 'fc_average_time'],
                 'power': ['pl_power_val', 'ps_power_val', 'mgt_power_val'],
                }

accelerator_binaries = {'FiBHA': 'fibha_full', 'FiBHA small': 'fibha_quarter',
                         'Baseline': 'seml_full', 'Baseline small': 'seml_quarter'}

def load_accelerator(accelerator_instance):
    connected_to_board = False
    booting_started_str = 'the system is going down for reboot now'
    loading_is_done_str = 'PetaLinux 2021.2 zcu102_base_sw ttyPS0'
    if len(sys.argv) == 2:
        connected_to_board = bool(sys.argv[1])
    print('>>>>>', connected_to_board)
    if connected_to_board:
        ser = serial.Serial('/dev/ttyUSB0', 115200)

        cmd = "cd /media/sd-mmcblk0p1\r"
        ser.write(cmd.encode())
        cmd = "ls\r"
        ser.write(cmd.encode())
        loading_executed = False
        booting_started = False
        booting_done = False
        while True:
            response_line = str(ser.readline())
            print(str(response_line))
            if 'is_loaded.txt' in str(response_line): 
                if accelerator_instance + '_is_loaded.txt' not in str(response_line):
                    cmd = "./replace_bin.sh " + accelerator_instance + '\r'
                    #print('YES')
                    if not loading_executed:
                        loading_executed = True
                        ser.write(cmd.encode()) 
                else:
                    break
            if booting_started_str in response_line.lower():
                booting_started = True
            if booting_started and (loading_is_done_str.lower() in response_line.lower()):
                break
    else:
        time.sleep(5)

def run_accelerator(accelerator_instance, report_energy, num_of_images, report_accuracy):
    
    connected_to_board = False
    if len(sys.argv) == 2:
        connected_to_board = bool(sys.argv[1])
        
    if connected_to_board:
        ser = serial.Serial('/dev/ttyUSB0', 115200)

        cmd = "cd /media/sd-mmcblk0p1\r"
        ser.write(cmd.encode())
                    
        # cmd = "./{} ./binary_container_{}.xclbin {} {} {}\r".format(accelerator_instance, accelerator_instance,
        #                                                             str(report_energy), str(num_of_images), str(report_accuracy))
        #cmd = './' + accelerator_instance + ' ./binary_container_' + accelerator_instance + '.xclbin ' + str(report_energy) + ' ' + str(num_of_images) + ' ' + str(report_accuracy) +'\r'
        cmd = './run_acc.sh {} {} {}\r'.format(report_energy, num_of_images, report_accuracy)
        print(cmd)
        ser.write(cmd.encode())

        response = ''
        while True:
            response_line = str(ser.readline())
            print(response_line)
            if '::EOF::' in str(response_line):
                break
            
            response += response_line

        with open(dump_dir + accelerator_instance + '_out.txt', 'w') as f:
            f.write(response)

def read_response(accelerator_instance):
    response = ''
    with open(dump_dir + accelerator_instance + '_out.txt', 'r') as f:
        for line in f:
            response += line

    return response

def extract_data(response, accelerator_instance):
    running_time = 0
    energy = 0
    splits = response.split(delimiter)
    for i in range(len(splits) - 1):
        if splits[i] in response_keys['predictions']:
            with open(predictions_dir + accelerator_instance + '_predictions.json', 'w') as f:
                f.write(splits[i + 1])
            break
        elif splits[i] in response_keys['time']:
            running_time += float(splits[i+1])
        elif splits[i] in response_keys['power']:
            energy += float(splits[i+1])

    energy *= running_time / 1000

    return [running_time, energy]


def evaluate_accuracy(accelerator_instance):
    out_dict = {'top1': '', 'top5': ''}
    proc = subprocess.Popen(['python3', 'evaluate_accuracy.py',  './out/predictions/' + accelerator_instance + '_predictions.json'], 
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

toprow = ['Accelerator', '# Images', 'Latency (ms)', 'Energy (J / inference)', 'Top1 accuracy', 'Top5 accuracy']
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
            sg.Combo([100, 200, 300, 400], font=font,
                        readonly=False, key='num_images_combo', default_value=100),
            sg.Checkbox('Report Energy', True, font= font, key='energy_check'),
             sg.Checkbox('Report accuracy', True, font= font, key='accuracy_check')] 
        ], font=font2, expand_x=True)

buttons_frame = [sg.Button('Load Accelerator', font=font2),
                 sg.Button('Run Accelerator', font=font2, expand_x=True), 
                 sg.Button('Close', font=font2)]

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
    accelerator_instance = values['fibha_combo']
    accelerator_instance_binary = accelerator_binaries[accelerator_instance]
    report_energy = values['energy_check']
    report_accuracy = values['accuracy_check']
    num_of_images = values['num_images_combo']
    if event == sg.WIN_CLOSED or event == 'Close': # if user closes window or clicks cancel
        break
    elif event == 'Load Accelerator':
        sg.popup_auto_close('Rebooting the System!\nThis will take one minute.', title='Rebooting', font=font2)
        load_accelerator(accelerator_instance_binary)
        sg.popup_auto_close('Done Rebooting!', title='Done', font=font2)
    elif event == 'Run Accelerator':
        run_accelerator(accelerator_instance_binary, int(report_energy), num_of_images, int(report_accuracy))

        accuracy_vals = {}
        experiment_row = []
        experiment_row.append(accelerator_instance)
        experiment_row.append(num_of_images)

        response = read_response(accelerator_instance_binary)
        running_time, energy = extract_data(response, accelerator_instance_binary)
        
        experiment_row.append(int(running_time))
        
        if report_energy:
            experiment_row.append(energy)
        else:
            experiment_row.append('_')

        if report_accuracy:
            accuracy_vals = evaluate_accuracy(accelerator_instance_binary)
            experiment_row.append(accuracy_vals['top1'])
            experiment_row.append(accuracy_vals['top5'])
        else:
            experiment_row.append('_')
            experiment_row.append('_')

        rows.append(experiment_row)
        results_table.update(rows)

window.close()

