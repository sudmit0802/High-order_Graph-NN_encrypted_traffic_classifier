# -*- coding: gbk -*-
import os
import json
import csv
from scapy.all import *
from tqdm import tqdm

ROOT_DIR = "C:\\Users\\sudmit\\Desktop\\High-order_Graph-NN_encrypted_traffic_classifier"

main_csv = os.path.join(ROOT_DIR, "workdir", "mainCSV", "main.csv")
session_csv_dir = os.path.join(ROOT_DIR, "workdir", "test", "raw", "session")
attr_csv = os.path.join(ROOT_DIR, "workdir", "test", "raw", "nodeattrs.csv")
edge_csv = os.path.join(ROOT_DIR, "workdir", "test", "raw", "edge.csv")
graph_csv = os.path.join(ROOT_DIR, "workdir", "test", "raw", "node2graphID.csv")
label_csv = os.path.join(ROOT_DIR, "workdir", "test", "raw", "graphid2label.csv")


ATTR_HEADER = ["ID", "Source Port", "Destination Port", "Sequence Number", "Acknowledgment Number",
               "Data Offset", "Reserved", "Flags", "Window Size", "Checksum", "Urgent Pointer"]

MAIN_HEADER = ATTR_HEADER + ["next_packages", "Session", "Label"]

GRAPH_HEADER = ["node_id", "graph_id"]
LABEL_HEADER = ["graph_no", "label"]


def delete_file(file_path):
    try:
        if os.path.isfile(file_path) or os.path.islink(file_path):
            os.unlink(file_path)
    except Exception as e:
        print(f"Failed to delete {file_path}. Reason: {e}")

def write_csv(file_path, header, rows, mode='w'):
    with open(file_path, mode=mode, newline='') as outfile:
        writer = csv.writer(outfile)
        if mode == 'w':
            writer.writerow(header)
        writer.writerows(rows)

def filter_columns(row, columns):
    return [row[col] for col in columns if col in row]


def generate_nodeattrs():
    delete_file(attr_csv)
    print("Generating nodeattrs.csv...")
    rows = []
    with open(main_csv, mode='r', newline='') as reader:
        csv_reader = csv.DictReader(reader)
        for row in tqdm(csv_reader, desc="Processing rows", leave=False):
            rows.append(filter_columns(row, ATTR_HEADER))
    write_csv(attr_csv, ATTR_HEADER, rows)

def generate_edge():
        print("Generating edge.csv...")
        delete_file(edge_csv)

        for filename in os.listdir(session_csv_dir):
            delete_file(os.path.join(session_csv_dir, filename))

        with open(main_csv, mode='r', newline='') as infile_session:
                reader_session = csv.DictReader(infile_session)
                for row_session in reader_session:
                    selected_columns = ["Session"]
                    selected_row = [row_session[col] for col in selected_columns if col in row_session]
                    session = selected_row[0]
                    file_path_session = os.path.join(session_csv_dir, f"{session}.csv")
                    file_exists_before_open = os.path.exists(file_path_session)
                    with open(file_path_session, mode='a', newline='') as outfile:
                        writer = csv.writer(outfile)
                        if not file_exists_before_open:
                            writer.writerow(MAIN_HEADER)
                        row_to_write = [row_session[col] for col in MAIN_HEADER if col in row_session]
                        writer.writerow(row_to_write)
        

        with open(edge_csv, mode='w', newline='') as outfile:

            writer = csv.writer(outfile)
            writer.writerow(['source_node', 'destination_node'])

            with open(main_csv, mode='r', newline='') as infile:

                reader = csv.DictReader(infile)

                for row in tqdm(reader, desc="Processing rows", leave=False, mininterval=30,
                                miniters=500):

                    selected_columns = ["ID", "next_packages", "Session"]
                    selected_row = [row[col] for col in selected_columns if col in row]
                    array = json.loads(selected_row[1])
                    file_to_ergonic = os.path.join(session_csv_dir, f"{selected_row[2]}.csv")


                    with open(file_to_ergonic, mode='r', newline='') as infile_2:
                        reader2 = csv.DictReader(infile_2)
                        counter = 0
                        all_counter = 0
                        for row_2 in reader2:
                            all_counter += 1
                            selected_columns_2 = ["ID", "Flags"]

                            selected_row_2 = [row_2[col_2] for col_2 in selected_columns_2 if col_2 in row_2]

                            if int(selected_row_2[1]) in array and selected_row[0] <= selected_row_2[0]:
                                writer.writerow([selected_row[0], selected_row_2[0]])
                                counter += 1


def generate_graph():
    delete_file(graph_csv)
    print("Generating graph.csv...")
    rows = []
    
    with open(main_csv, mode='r', newline='') as reader:
        csv_reader = csv.DictReader(reader)
        for row in tqdm(csv_reader, desc="Processing rows", leave=False):
            rows.append(filter_columns(row, ["ID", "Session"]))
    write_csv(graph_csv, GRAPH_HEADER, rows)


def generate_label(labels):
    delete_file(label_csv)
    print("Generating label.csv...")
    last_session = None
    rows = []
    label_dict = {label : i for i, label in enumerate(labels)}
    with open(main_csv, mode='r', newline='') as reader:
        csv_reader = csv.DictReader(reader)
        for row in tqdm(csv_reader):
            session, label = row.get("Session"), row.get("Label")
            if session and session != last_session:
                rows.append([session, label_dict.get(label, -1)])
                last_session = session
    write_csv(label_csv, LABEL_HEADER, rows)



def gen_main_csv(labels):
    
    generate_nodeattrs()
    generate_edge()
    generate_graph()
    generate_label(labels)

