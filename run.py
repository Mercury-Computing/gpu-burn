import subprocess
from datetime import datetime
import time
import sys

duration = '10'
if len(sys.argv) > 1:
  duration = sys.argv[1]

load_high_threshold = 340
load_low_threshold = 100
load_empty_threshold = 40

logs = []
signal_up = datetime.utcnow()
load_full = None

p = subprocess.Popen(['./gpu_burn', '-stts', '0', duration])

while p.poll() is None:
  out = subprocess.check_output(['nvidia-smi', '--query-gpu=timestamp,power.draw', '--format=csv']).decode()
  ts, power_draw = out.split('\n')[-2].split(',')
  wattage = float(power_draw.strip().split(' ')[0])
  if not load_full and wattage > load_high_threshold:
    load_full = datetime.strptime(ts, '%Y/%m/%d %H:%M:%S.%f')
  logs.append({ 'ts': ts, 'wattage': wattage })
  time.sleep(0.1)

signal_down = datetime.utcnow()
load_low = None
load_empty = None

while not load_empty:
  out = subprocess.check_output(['nvidia-smi', '--query-gpu=timestamp,power.draw', '--format=csv']).decode()
  ts, power_draw = out.split('\n')[-2].split(',')
  wattage = float(power_draw.strip().split(' ')[0])
  if not load_low and wattage < load_low_threshold:
    load_low = datetime.strptime(ts, '%Y/%m/%d %H:%M:%S.%f')
  if not load_empty and wattage < load_empty_threshold:
    load_empty = datetime.strptime(ts, '%Y/%m/%d %H:%M:%S.%f')
  logs.append({ 'ts': ts, 'wattage': wattage })
  time.sleep(0.1)

print('\n'.join([f'{l["ts"]}, {l["wattage"]}' for l in logs]))
print(f'signal up: {signal_up}')
print(f'load high (>340W): {load_full} (+{str(load_full - signal_up)})')
print(f'signal down: {signal_down} (+{str(signal_down - signal_up)})')
print(f'load low (<100W): {load_low} (+{str(load_low - signal_down)})')
print(f'load empty (<40W): {load_empty} (+{str(load_empty - signal_down)})')