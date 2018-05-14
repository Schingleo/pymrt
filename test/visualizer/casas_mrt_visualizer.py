import matplotlib as mpl
mpl.use('TkAgg')

import pickle
from pymrt.visualizer.mrt._casas_playback import plot_observation


result_filename = 'result/tm004_161219_7d_stepmrt_result.pkl'

if __name__ == '__main__':
    fp = open(result_filename, 'rb')
    result = pickle.load(fp)
    # _plot_ota(result['ota'], result['time'], result['steps'], frame=100,
    #           steps_before=50, total_steps=60)
    mrt_playback_on_site(result)