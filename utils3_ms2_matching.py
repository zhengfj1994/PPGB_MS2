import os
import pickle
import pandas as pd
from tqdm.notebook import tqdm
from joblib import Parallel, delayed
from pandas.core.frame import DataFrame

def batch_ms2_matching(MSMSFilePath, step2FilePath, step3FilePath, n_jobs = -1):
    if not os.path.exists(step3FilePath): os.makedirs(step3FilePath)

    rawCsvData = pd.read_csv(MSMSFilePath, encoding = u'gbk')
    csvData = rawCsvData

    def my_task2(i):

        MSMS = csvData['MSMS'].values[i]
        MSMS = DataFrame(MSMS.split(';'))
        MSMS = MSMS[0].str.split(' ',expand=True)
        MSMS = MSMS[pd.to_numeric(MSMS[1], errors='coerce').isnull() == False]
        MSMS = MSMS.apply(pd.to_numeric)
        MSMS[1] = MSMS[1]/max(MSMS[1])

        fileName = csvData['inchikey'].values[i] + '.pkl'
        if fileName not in os.listdir(step2FilePath):
            return 'none'
        load_file = open(step2FilePath + '/' + fileName,"rb")
        try:
            fragment_list = pickle.load(load_file)
        except:
            return 'error'

        for _ in fragment_list:
            matched_MSMS = MSMS[(MSMS[0]<_.graph['product_mz']+0.01) & (MSMS[0]>_.graph['product_mz']-0.01)][1]
            if len(matched_MSMS) == 0:
                _.graph['intensity_regression1'] = 0
                _.graph['intensity_regression2'] = 0
                _.graph['intensity_classification'] = 0
            else:
                _.graph['intensity_regression1'] = matched_MSMS.mean()
                _.graph['intensity_regression2'] = sum(matched_MSMS)/sum(MSMS[1])
                _.graph['intensity_classification'] = 1
                
        save_file = open(step3FilePath + '/' + fileName,"wb")
        pickle.dump(fragment_list, save_file)
        save_file.close()
        return i

    lst1 = range(csvData.shape[0])
    Parallel(n_jobs = n_jobs, verbose = 1)(delayed(my_task2)(i) for i in tqdm(lst1))