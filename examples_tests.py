from datetime import datetime, timedelta, timezone
import os
import pandas as pd

from data_management.io.kp import KpNiemegk, KpSWPC, KpOMNI
from data_management.io.omni import OMNILowRes, OMNIHighRes
from data_management.io.kp import read_kp_from_multiple_models
from data_management.io.hp import Hp30GFZ, Hp60GFZ, Hp30Ensemble
from data_management.io.solar_wind import SWACE, SWOMNI, read_solar_wind_from_multiple_models

from matplotlib import pyplot as plt

# os.environ['RT_KP_NIEMEGK_STREAM_DIR'] = '/home/bhaas/FLAG_TEST/Niemegk/'
# os.environ['RT_KP_SWPC_STREAM_DIR'] = '/home/bhaas/FLAG_TEST/SWPC/'
# os.environ['OMNI_LOW_RES_STREAM_DIR'] = '/home/bhaas/FLAG_TEST/OMNI_LOW_RES/'
# os.environ['KP_ENSEMBLE_FORECAST_DIR'] = '/PAGER/WP3/data/outputs/SWIFT_ENSEMBLE/'
# os.environ['RT_HP_GFZ_STREAM_DIR'] = '/home/bhaas/FLAG_TEST/Hp/'
# os.environ['HP30_ENSEMBLE_FORECAST_DIR'] = '/PAGER/WP3/data/outputs/HP30/'

start_time = datetime.today() - timedelta(days=60)
end_time = datetime.today() - timedelta(days=55)

start_time = datetime(start_time.year, start_time.month, start_time.day) + timedelta(seconds=1)
end_time = datetime(end_time.year, end_time.month, end_time.day) - timedelta(seconds=1)


#KpNiemegk().download_and_process(datetime.today()-timedelta(days=5), datetime.today()+timedelta(days=2), reprocess_files=True)
#print(KpNiemegk().read(start_time, end_time))

#KpSWPC().download_and_process(datetime.today(), reprocess_files=True)
#print(KpSWPC().read(datetime(2024, 5, 10)))

#OMNIHighRes().download_and_process(start_time, end_time, cadence_min=1, reprocess_files=False)
#print(SWOMNI().read(start_time, end_time))

#Hp60GFZ().download_and_process(datetime.today(), datetime.today(), reprocess_files=True)
#print(Hp60GFZ().read(datetime.today()-timedelta(days=365), datetime.today()+timedelta(days=10)))

#print(Hp30Ensemble().read(datetime.today(), datetime.today()+timedelta(hours=30)))

# data_kp_all = read_kp_from_multiple_models(datetime.today()-timedelta(days=12), datetime.today()+timedelta(days=2))
# print(data_kp_all[0])
# print(data_kp_all[0].loc["2024-09-21 11:00:00", "file_name"])

#data_omni = data_kp_all[0].loc[data_kp_all[0]['model'] == 'omni']
#data_niemegk = data_kp_all[0].loc[data_kp_all[0]['model'] == 'niemegk']

#plt.subplots(1,1,figsize=(19/2,10/2))
#plt.step(data_omni.index, data_omni['kp'], 'b')
#plt.step(data_niemegk.index, data_niemegk['kp'], 'r')

# for ie in range(len(data_kp_all)):
#      data_ensemble = data_kp_all[ie].loc[data_kp_all[ie]['model'] == 'ensemble']
#      plt.step(data_ensemble.index, data_ensemble['kp'], 'g')

# plt.legend(["Omni", "Niemegk", "Ensemble"])
# plt.grid()
# plt.savefig('test.png')

#SWACE().download_and_process(datetime.today(), reprocess_files=True)
#print(SWACE().read(datetime.today()-timedelta(hours=4), datetime.today()+timedelta(hours=3)))


data_sw_all = read_solar_wind_from_multiple_models(datetime.now(timezone.utc)-timedelta(hours=2), datetime.now(timezone.utc)+timedelta(hours=1))
#with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
print(data_sw_all[0])

data_omni = data_sw_all[0].loc[data_sw_all[0]['model'] == 'omni']
data_ace = data_sw_all[0].loc[data_sw_all[0]['model'] == 'ace']
data_swift = data_sw_all[0].loc[data_sw_all[0]['model'] == 'swift']

plt.subplots(1,1,figsize=(19/2,10/2))
plt.plot(data_omni.index, data_omni['speed'], 'b')
plt.plot(data_ace.index, data_ace['speed'], 'r')

for ie in range(len(data_sw_all)):
     data_ensemble = data_sw_all[ie].loc[data_sw_all[ie]['model'] == 'swift']
     plt.step(data_ensemble.index, data_ensemble['speed'], 'g')

plt.legend(["Omni", "ACE", "SWIFT"])
plt.grid()
plt.savefig('sw_test.png')