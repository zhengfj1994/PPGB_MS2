import os
import pickle
import pandas as pd
from tqdm.notebook import tqdm
from joblib import Parallel, delayed
from pandas.core.frame import DataFrame

def batch_ms2_matching(MSMSFilePath, step2FilePath, step3FilePath, int_threshold, MS2_mz_tolerance, n_jobs = -1):
    if not os.path.exists(step3FilePath): os.makedirs(step3FilePath)

    rawCsvData = pd.read_csv(MSMSFilePath, encoding = u'gbk')
    csvData = rawCsvData

    def my_task(i):

        InstrumentManufacturer = csvData['InstrumentManufacturer'][i]
        IonSource = csvData['IonSource'][i]
        DetectorType = csvData['DetectorType'][i]
        FragmentationType = csvData['FragmentationType'][i]
        CollisionGas = csvData['CollisionGas'][i]
        CE = csvData['CE'][i]
        KE = csvData['KE'][i]
        EBC = csvData['EBC'][i]
        AdductType = csvData['AdductType'][i]

        step2FileName = csvData['AdductType'].values[i] + '_' +csvData['inchikey'].values[i] + '.pkl'
        step3FileName = InstrumentManufacturer + '_' + IonSource + '_' + DetectorType + '_' + FragmentationType + '_' + CollisionGas + '_' + str(CE) + '_' + str(KE) + '_' + str(EBC) + '_' + step2FileName
        
        if step3FileName not in os.listdir(step3FilePath) and step2FileName in os.listdir(step2FilePath): # If ith item is not processed
            load_file = open(step2FilePath + '/' + step2FileName,"rb")
            try:
                fragment_list = pickle.load(load_file)
            except:
                return 'error'

            if 'MSMS' in csvData.columns:
                MSMS = csvData['MSMS'].values[i]
                MSMS = DataFrame(MSMS.split(';'))
                MSMS = MSMS[0].str.split(' ',expand=True)
                MSMS = MSMS[pd.to_numeric(MSMS[1], errors='coerce').isnull() == False]
                MSMS = MSMS.apply(pd.to_numeric)
                MSMS = MSMS[MSMS[1] > int_threshold]
                if len(MSMS) == 0:
                    return 'No fragment'

            for _ in fragment_list:
                if 'MSMS' in csvData.columns:
                    matched_MSMS = MSMS[(MSMS[0]<_[0]+MS2_mz_tolerance) & (MSMS[0]>_[0]-MS2_mz_tolerance)][1]
                    if len(matched_MSMS) == 0:
                        _ += [{'intensity': 0}]
                    else:
                        _ += [{'intensity': sum(matched_MSMS)/sum(MSMS[1])}]
                _+= [{'InstrumentManufacturer': InstrumentManufacturer, 'IonSource': IonSource, 'DetectorType': DetectorType, 'FragmentationType': FragmentationType,
                      'CollisionGas': CollisionGas, 'CE': CE, 'KE': KE, 'EBC': EBC, 'AdductType': AdductType}]

            save_file = open(step3FilePath + '/' + step3FileName,"wb")
            pickle.dump(fragment_list, save_file)
            save_file.close()
            return i
        else:
            return 'done'

    lst1 = range(csvData.shape[0])
    Parallel(n_jobs = n_jobs, verbose = 1)(delayed(my_task)(i) for i in tqdm(lst1))