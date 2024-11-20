import csv
from scapy.all import *
from scapy.layers.inet import TCP
import os
from tqdm import tqdm
import random
import gen_main_csv_trans

# Folder paths
pcap_file_folders = [
     os.path.join(gen_main_csv_trans.ROOT_DIR, "workdir", "pcaps", "gmail"),
     os.path.join(gen_main_csv_trans.ROOT_DIR, "workdir", "pcaps", "mysql"),
     os.path.join(gen_main_csv_trans.ROOT_DIR, "workdir", "pcaps", "outlook"),
     os.path.join(gen_main_csv_trans.ROOT_DIR, "workdir", "pcaps", "skype"),
     os.path.join(gen_main_csv_trans.ROOT_DIR, "workdir", "pcaps", "smb"),
     os.path.join(gen_main_csv_trans.ROOT_DIR, "workdir", "pcaps", "torrent"),
     os.path.join(gen_main_csv_trans.ROOT_DIR, "workdir", "pcaps", "weibo"),
     os.path.join(gen_main_csv_trans.ROOT_DIR, "workdir", "pcaps", "wow"),
]

temp_session_dir = os.path.join(gen_main_csv_trans.ROOT_DIR, "workdir", "Temp")
sharkpath = "C:\\Program Files\\Wireshark"
    

protocol_max = 10000


def remove_files_in_dir(directory):
   
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
        except Exception as e:
            print(f"Failed to delete {file_path}. Reason: {e}")

def split_pcaps(folder_path):
 
    if not os.path.exists(folder_path):
        print("Folder not found")
        return
    for filename in os.listdir(folder_path):
        if filename.endswith(".pcap"):
            file_path = os.path.join(folder_path, filename)
            print(f"Splitting {file_path}")
            command = "\"" + sharkpath + "/editcap.exe " + "\"" + " -F pcap " + str(file_path) + \
                      " - | SplitCap.exe -r - -s session -o " + str(temp_session_dir)
            os.system(command)

def read_protocol_states():
    states = {}
    counter = 0
    for filename in os.listdir(temp_session_dir):
        if filename.endswith(".pcap"):
            file_path = os.path.join(temp_session_dir, filename)
            packets = rdpcap(file_path)
            tcp_packets = [packet for packet in packets if TCP in packet]
            for packet in tcp_packets:
                tcp_flag = packet[TCP].flags
                if tcp_flag not in states.values():
                    states[counter] = tcp_flag
                    counter += 1
    return states

def generate_transition_matrix(file_path, states, matrix):
    packets = rdpcap(file_path)
    last_state = None
    tcp_packets = [packet for packet in packets if TCP in packet]
    for packet in tcp_packets:
        tcp_flag = packet[TCP].flags
        if last_state:
            find_key_last = -1
            for key, value in states.items():
                if value == last_state:
                    find_key_last = key
            if find_key_last == -1:
                print("Fatal error: Unknown flag")
                exit(1)

            find_key_this = -1
            for key, value in states.items():
                if value == tcp_flag:
                    find_key_this = key
            if find_key_this == -1:
                print("Fatal error: Unknown flag")
                exit(1)
            matrix[find_key_last][find_key_this] = 1
        last_state = tcp_flag
    return matrix

def gen_csv(id, packet, matrix, label, states, packets, filename, session):
   
    next_states = []
    features = {}

    if not packet.haslayer(TCP):
        return -1
    tcp_flag = packet[TCP].flags
    find_key_this = -1

    for key, value in states.items():
        if value == tcp_flag:
            find_key_this = key
    if find_key_this == -1:
        print("Fatal error: Unknown flag")
        exit(1)
    next_keys = matrix[find_key_this]

    for index,key_2 in enumerate(next_keys):
        if key_2 == 1:
            next_states.append(int(states[index]))

    next_states = str(next_states)
    if packet.haslayer(TCP):
        tcp_header = packet[TCP]
        features = {
            "Source Port": tcp_header.sport,
            "Destination Port": tcp_header.dport,
            "Sequence Number": tcp_header.seq,
            "Acknowledgment Number": tcp_header.ack,
            "Data Offset": tcp_header.dataofs,
            "Reserved": tcp_header.reserved,
            "Flags": int(packet[TCP].flags),
            "Window Size": tcp_header.window,
            "Checksum": tcp_header.chksum,
            "Urgent Pointer": tcp_header.urgptr,
        }
    data_row = [id] + [features[key] for key in gen_main_csv_trans.MAIN_HEADER[1:-3]] + [next_states, session, label]
    with open(filename, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(data_row)
    return 1

def process_protocol(pcap_folder, max_sessions, session_counter, packet_counter):

    split_pcaps(pcap_folder)
    for filename in os.listdir(temp_session_dir):
        if "UDP" in filename:
            os.unlink(os.path.join(temp_session_dir, filename))

    states = read_protocol_states()
    matrix_size = len(states)
    transition_matrix = [[0] * matrix_size for _ in range(matrix_size)]

    # Build transition matrix
    for filename in os.listdir(temp_session_dir):
        if filename.endswith(".pcap"):
            file_path = os.path.join(temp_session_dir, filename)
            generate_transition_matrix(file_path, states, transition_matrix)

    # Select sessions for training
    session_files = [f for f in os.listdir(temp_session_dir) if f.endswith(".pcap")]
    selected_sessions = random.sample(session_files, min(len(session_files), max_sessions))

    protocol = os.path.basename(pcap_folder)

    for session_file in tqdm(selected_sessions):
        file_path = os.path.join(temp_session_dir, session_file)
        packets = rdpcap(file_path)
        session_added = False
        for packet in packets:
            if gen_csv(packet_counter, packet, transition_matrix, protocol, states, packets,
                       gen_main_csv_trans.main_csv, session_counter) > 0:
                packet_counter += 1
                session_added = True
        if session_added:
            session_counter += 1

    remove_files_in_dir(temp_session_dir)
    return session_counter, packet_counter

if __name__ == '__main__':

    with open(gen_main_csv_trans.main_csv, 'w') as csv_file:
        csv_file.write(','.join(gen_main_csv_trans.MAIN_HEADER) + '\n')

    session_counter, packet_counter = 1, 0
    for pcap_folder in pcap_file_folders:
         session_counter, packet_counter = process_protocol(pcap_folder, protocol_max, session_counter, packet_counter)

    gen_main_csv_trans.gen_main_csv([os.path.basename(pcap_folder) for pcap_folder in pcap_file_folders])
