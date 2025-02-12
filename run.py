import subprocess
from datetime import datetime, timedelta
import time
import sys

duration = 10
if len(sys.argv) > 1:
  duration = int(sys.argv[1])

load_high_threshold = 340
load_low_threshold = 90
load_empty_threshold = 40

logs = []
signal_up = datetime.utcnow()
load_full = None

print(signal_up, " - sending signal up")

p = subprocess.Popen('./gpu_burn')

while datetime.utcnow() - signal_up < timedelta(seconds=duration):
# while p.poll() is None:
  out = subprocess.check_output(['nvidia-smi', '--query-gpu=timestamp,power.draw', '--format=csv']).decode()
  ts, power_draw = out.split('\n')[-2].split(',')
  wattage = float(power_draw.strip().split(' ')[0])
  if not load_full and wattage > load_high_threshold:
    load_full = datetime.strptime(ts, '%Y/%m/%d %H:%M:%S.%f')
    print(load_full, " - gpu load is full")
  logs.append({ 'ts': ts, 'wattage': wattage })
  time.sleep(0.1)

signal_down = datetime.utcnow()
print(signal_down, " - sending signal down")
p.terminate()

load_low = None
load_empty = None

while not load_empty:
  out = subprocess.check_output(['nvidia-smi', '--query-gpu=timestamp,power.draw', '--format=csv']).decode()
  ts, power_draw = out.split('\n')[-2].split(',')
  wattage = float(power_draw.strip().split(' ')[0])
  if not load_low and wattage < load_low_threshold:
    load_low = datetime.strptime(ts, '%Y/%m/%d %H:%M:%S.%f')
    print(load_low, " - gpu load is low")
  if not load_empty and wattage < load_empty_threshold:
    load_empty = datetime.strptime(ts, '%Y/%m/%d %H:%M:%S.%f')
    print(load_empty, " - gpu load is at baseline")
  logs.append({ 'ts': ts, 'wattage': wattage })
  time.sleep(0.1)

# print('\n'.join([f'{l["ts"]}, {l["wattage"]}' for l in logs]))
# print(f'signal up: {signal_up}')
# print(f'load high (>{load_high_threshold}W): {load_full} (+{str(load_full - signal_up)})')
# print(f'signal down: {signal_down} (+{str(signal_down - signal_up)})')
# print(f'load low (<{load_low_threshold}W): {load_low} (+{str(load_low - signal_down)})')
# print(f'load empty (<{load_empty_threshold}W): {load_empty} (+{str(load_empty - signal_down)})')