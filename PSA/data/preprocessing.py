import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

idx_dict = {1: "age", 2:"sex", 3:"dre_days0", 4:"dre_days1",
            5:"dre_days2", 6:"psa_level0", 7:"psa_level1",
            8:"psa_level2", 9:"psa_level3", 10:"psa_level4", 11:"psa_level5",
            12:"pros_clinstage", 13:"pros_gleason", 14:"dx_psa",
            15:"dth_days", 16:"race7", 17:"educat", 18:"marital",
            19:"occupat", 20:"cig_stat", 21:"cig_years", 22:"cigpd_f",
            23:"fh_cancer", 24:"pros_fh", 25:"pros_fh_age", 26:"bmi_curc", 27:"prosprob_f",
            28:"surg_age", 29: "surg_any", 30:"surg_biopsy", 31:"surg_prostatectomy", 32:"surg_resection",
            33:"arthrit_f", 34:"bronchit_f", 35:"colon_comorbidity", 36:"diabetes_f", 37:"divertic_f", 38:"emphys_f", 39:"gallblad_f",
            40:"hearta_f", 41:"hyperten_f", 42:"liver_comorbidity", 43:"osteopor_f", 44:"polyps_f",
            45:"stroke_f",46:"psa_days0", 47:"psa_days1", 48:"psa_days2", 49:"psa_days3", 50:"psa_days4", 51:"psa_days5",
            100: "pros_cancer", 101: "is_dead"}

def get_data(idx_list=[]):
    if len(idx_list) == 0 : idx_list = list(idx_dict.values())

    df = pd.read_csv('../../data/pros_data_nov18_d070819.csv')
    df = df[idx_list]
    df = df.dropna()


    return df


def ROS_sampling(cancer_numpy, rate = 1000):
    x = cancer_numpy
    idx = np.random.choice(x.shape[0], x.shape[0] * int(rate/ 100))
    x = x[idx, :]
    return x

if __name__ == "__main__":
    idx_num = [i for i in range(len(idx_dict.values()))]

    rmv_list = [12, 13, 14]
    for rmv in rmv_list :
        idx_num.remove(rmv)

    idx_num = [100,6,7,8,9,10,11,16,17,18,19,20,21,22,23,24,26,27,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,1,2]

    #[100, 3, 4, 5, 6, 7, 8, 9, 10, 11, 46, 47, 48, 49, 50, 51]  #

    #pd.set_option('display.max_columns', None)
    idx_list = [idx_dict[i_num] for i_num in idx_num]

    #print(idx_list)
    df = get_data(idx_list=idx_list)
    #print(len(df))
    #df = get_data(idx_list = idx_)
    #print(df.head())
    #print(df.iloc[[0]])
    #aise ValueError
    training_data = df.sample(frac=0.8, random_state=25)
    testing_data = df.drop(training_data.index)
    u = testing_data[testing_data.iloc[:, 0] == 1]
    #print(len(u))
    #print(u)
    np.save("PSA_valid", testing_data.to_numpy())

    df = training_data
    non_cancer = df[df.iloc[:,0] != 1].to_numpy()
    cancer = df[df.iloc[:,0] == 1].to_numpy()

    rate = 1000
    cancer = ROS_sampling(cancer, rate=rate)
    data = np.vstack((non_cancer, cancer))
    print(data.shape)
    #print(data[0])
    np.random.shuffle(data)
    print(data[0:10])
    np.save("PSA_train_{}".format(rate), data)







