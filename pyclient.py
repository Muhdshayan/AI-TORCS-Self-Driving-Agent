#!/usr/bin/env python
'''
Created on Apr 4, 2012
Client code for connecting to the TORCS SCRC server with model-driven control.
Logs verbose telemetry data (received and sent messages) to a CSV file.
Each key/value pair from the verbose strings is logged in a separate column.
'''

import sys
import argparse
import socket
import driver  # Model-driven driver module
import pandas as pd
import os
import time
import re

# --- Telemetry Logging Functions ---

def parse_verbose_message(msg):
    result = {}
    # Find all parenthesized groups
    groups = re.findall(r'\(([^)]+)\)', msg)
    for group in groups:
        parts = group.split()
        if not parts:
            continue
        key = parts[0]
        values = parts[1:]
        if len(values) == 1:
            result[key] = values[0]
        else:
            for i, val in enumerate(values, start=1):
                result[f"{key}_{i}"] = val
    return result


def log_telemetry(received_msg, sent_msg):
    log_data = {"timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}

    # Parse received and sent messages into dictionaries
    rec_dict = parse_verbose_message(received_msg)
    sent_dict = parse_verbose_message(sent_msg)

    # Prefix keys and update log_data
    for k, v in rec_dict.items():
        log_data[f"r_{k}"] = v
    for k, v in sent_dict.items():
        log_data[f"s_{k}"] = v

    # No manual keys for model-driven mode
    log_data["keys_pressed"] = ""

    df = pd.DataFrame([log_data])
    header = not os.path.exists("telemetry_verbose_log.csv")
    df.to_csv("telemetry_verbose_log.csv", mode="a", header=header, index=False)


if __name__ == '__main__':
    # Configure the argument parser
    parser = argparse.ArgumentParser(description='Python client to connect to the TORCS SCRC server.')
    parser.add_argument('--host', action='store', dest='host_ip', default='localhost',
                        help='Host IP address (default: localhost)')
    parser.add_argument('--port', action='store', type=int, dest='host_port', default=3001,
                        help='Host port number (default: 3001)')
    parser.add_argument('--id', action='store', dest='id', default='SCR',
                        help='Bot ID (default: SCR)')
    parser.add_argument('--maxEpisodes', action='store', dest='max_episodes', type=int, default=1,
                        help='Maximum number of learning episodes (default: 1)')
    parser.add_argument('--maxSteps', action='store', dest='max_steps', type=int, default=0,
                        help='Maximum number of steps (default: 0)')
    parser.add_argument('--track', action='store', dest='track', default=None,
                        help='Name of the track')
    parser.add_argument('--stage', action='store', dest='stage', type=int, default=3,
                        help='Stage (0 - Warm-Up, 1 - Qualifying, 2 - Race, 3 - Unknown)')

    arguments = parser.parse_args()

    # Print summary
    print('Connecting to server host ip:', arguments.host_ip, '@ port:', arguments.host_port)
    print('Bot ID:', arguments.id)
    print('Maximum episodes:', arguments.max_episodes)
    print('Maximum steps:', arguments.max_steps)
    print('Track:', arguments.track)
    print('Stage:', arguments.stage)
    print('*********************************************')

    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    except socket.error:
        print('Could not make a socket.')
        sys.exit(-1)

    sock.settimeout(1.0)

    shutdownClient = False
    curEpisode = 0
    verbose = True

    # Initialize the driver with model-driven control
    d = driver.Driver(arguments.stage)

    # Initialization phase: send init string until server identifies the client.
    while not shutdownClient:
        while True:
            buf_init = arguments.id + d.init()
            if verbose:
                print('Sending init to server:', buf_init)
            try:
                sock.sendto(buf_init.encode('utf-8'), (arguments.host_ip, arguments.host_port))
            except socket.error:
                print("Failed to send data...Exiting...")
                sys.exit(-1)

            try:
                buf, addr = sock.recvfrom(1024)
                buf = buf.decode('utf-8')
            except socket.error:
                continue

            if '***identified***' in buf:
                if verbose:
                    print('Server response:', buf)
                break

        currentStep = 0
        # Main loop for one episode
        while True:
            try:
                buf, addr = sock.recvfrom(1024)
                received_msg = buf.decode('utf-8')
            except socket.error:
                continue

           # if verbose:
            #    print('Received:', received_msg)

            # Check for shutdown or restart messages
            if '***shutdown***' in received_msg:
                print('Client Shutdown')
                shutdownClient = True
                break
            if '***restart***' in received_msg:
                print('Client Restart')
                break

            currentStep += 1
            # Determine control message
            if arguments.max_steps and currentStep >= arguments.max_steps:
                sent_msg = '(meta 1)'
            else:
                sent_msg = d.drive(received_msg)

            #if verbose:
             #   print('Sending:', sent_msg)

            try:
                sock.sendto(sent_msg.encode('utf-8'), (arguments.host_ip, arguments.host_port))
            except socket.error:
                print("Failed to send data...Exiting...")
                sys.exit(-1)

            # Log telemetry
            log_telemetry(received_msg, sent_msg)

        curEpisode += 1
        if curEpisode >= arguments.max_episodes:
            shutdownClient = True

    sock.close()