import os
import scipy.io


class RawFormat(object):
    def __init__(self):
        pass

    @staticmethod
    def matlab(data, filename):
        dic = dict()
        dic["t"] = [ind.strftime('%Y-%m-%d %H:%M:%S') for ind in data.index]
        dic["kp"] = data["kp"].values
        scipy.io.savemat(filename, dic)

    @staticmethod
    def csv(data, filename, index, header=True):
        if os.path.exists(filename):
            os.remove(filename)
        f = open(filename, 'a')
        if header:
            f.write('# Most recent {} index forecast'.format(index))
            f.write('\n')
            f.write('# Produced by GFZ Section 2.7')
            f.write('\n\n')
        data["forecast"] = data["forecast"].apply(RawFormat._reformat_index)
        data.to_csv(f, header=True, index=True)
        if header:
            f.write('\n')
            f.write('# For more information contact Dr. Ruggero Vasile (ruggero@gfz-potsdam.de)')

    @staticmethod
    def omniweb(data, filename):
        if os.path.exists(filename):
            os.remove(filename)
        data["year"] = data.index.map(lambda x: x.year)
        data["doy"] = data.index.map(lambda x: x.timetuple().tm_yday)
        data["hour"] = data.index.map(lambda x: x.hour)
        kp = data["kp"].apply(RawFormat._reformat_index)
        data.drop(["kp"], 1, inplace=True)
        data["kp"] = kp
        data.to_csv(filename, header=False, index=False, sep=" ")

    # TODO To finish
    @staticmethod
    def wdc(data, filename, forecast_date=None):
        if os.path.exists(filename):
            os.remove(filename)
        if forecast_date is None:
            forecast_date = data.index[0]
        forecast = data["kp"].apply(RawFormat._reformat_index_plusminus)
        values = "".join(list(forecast.values))
        f = open(filename, 'a')
        f.write(forecast_date.strftime("%Y%m%d%H"))
        f.write(' ')
        f.write(values)
        f.close()

    @staticmethod
    def _reformat_index(x):
        if abs(x - int(x)) < 0.00001:
            return int(x)
        elif x - int(x) < 0.4:
            return float(int(x) + 0.3)
        else:
            return float(int(x) + 0.7)

    @staticmethod
    def _reformat_index_plusminus(x):
        if abs(x - int(x)) < 0.00001:
            return str(int(x))
        elif x - int(x) < 0.4:
            return str(int(x)) + "+"
        else:
            return str(int(x) + 1) + "-"
