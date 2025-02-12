import subprocess
from datetime import datetime
import time
import sys

duration = '10'
if len(sys.argv) > 1:
  duration = sys.argv[1]

logs = []
signal_up = datetime.utcnow()
load_full = None

p = subprocess.Popen(['./gpu_burn', '-stts', '0', duration])

while p.poll() is None:
  out = subprocess.check_output(['nvidia-smi', '--query-gpu=timestamp,power.draw', '--format=csv']).decode()
  ts, power_draw = out.split('\n')[-2].split(',')
  wattage = float(power_draw.strip().split(' ')[0])
  if not load_full and wattage > 340:
    load_full = datetime.strptime(ts, '%Y/%m/%d %H:%M:%S.%f')
  logs.append({ 'ts': ts, 'wattage': wattage })
  time.sleep(0.1)

signal_down = datetime.utcnow()
load_empty = None

while not load_empty:
  out = subprocess.check_output(['nvidia-smi', '--query-gpu=timestamp,power.draw', '--format=csv']).decode()
  ts, power_draw = out.split('\n')[-2].split(',')
  wattage = float(power_draw.strip().split(' ')[0])
  if not load_empty and wattage < 40:
    load_empty = datetime.strptime(ts, '%Y/%m/%d %H:%M:%S.%f')
  logs.append({ 'ts': ts, 'wattage': wattage })
  time.sleep(0.1)

print('\n'.join([f'{l["ts"]}, {l["wattage"]}' for l in logs]))
print(f'signal up: {signal_up}')
print(f'load full: {load_full} (+{str(load_full - signal_up)})')
print(f'signal down: {signal_down} (+{str(signal_down - load_full)})')
print(f'load empty: {load_empty} (+{str(load_empty - signal_down)})')