# -*- coding: utf-8 -*-
"""
Created on Sat Jul 27 12:52:26 2019

@author: Desenvolvedor
"""


# Importing Libraries

import glob as gb
import numpy as np
import pandas as pd
from sklearn import svm
#from sklearn import model_selection as msel
#import matplotlib.colors as colors
import matplotlib.pyplot as plt
import seaborn as sns; sns.set(font_scale=1.2)
#from scipy.fftpack import fft
#from sklearn.preprocessing import StandardScaler
#from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV
#from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
#from sklearn.svm import SVC as clf

# Importing data-sets

sess1_files = gb.glob("user_coordinates/*session1*.txt")
sess2_files = gb.glob("user_coordinates/*session2*.txt")


# Importing user data to data-frame
def gen_data(user, session):
    if session==1:
        user1data_path = sess1_files[user]
    else:
        user1data_path = sess2_files[user]
    dfraw = pd.read_csv(user1data_path, sep=",", header=None)
    dfraw.columns = ["x", "y", "z"]
    return dfraw

# Centering data around 0 to compensate gravity and phone calibration
def center(df):
    df_mean = df.mean(axis=0)
    df1 = []
    for _ in df:
        df1.append(df - df_mean)
    dfx = pd.DataFrame(df1[0])
    return dfx


# Windowing data to time-frame(ms) with 50% overlap considering frequency (Hz)
def window_size(window, frequency):
    wdw = int(window * frequency)
    return wdw


# Features Extraction
def magnetude_vector(df):
    for _ in df:
        m = (df.x**2 + df.y**2 + df.z**2)**(1/2)
        df["mag_v"] = m
    return df

def magnetude_vector_fft(df):
    for _ in df:
        x = (df.x_abs1 + df.x_abs2 + df.x_abs3 + df.x_abs4 + df.x_abs5)
        y = (df.y_abs1 + df.y_abs2 + df.y_abs3 + df.y_abs4 + df.y_abs5)
        z = (df.z_abs1 + df.z_abs2 + df.z_abs3 + df.z_abs4 + df.z_abs5)
        m1 = (x**2 + y**2 + z**2)**(1/2)
        df["mag_xyz"] = m1
        m2 = (x**2 + y**2)**(1/2)
        df["mag_xy"] = m2
        x = 0
        y = 0
        z = 0
    return df

def data_feat(df, wdw):
    mean = []
    mini = []
    maxi = []
    std_dev = []
    # Loop with window to extract features
    while wdw < len(df):
        df_aux = df[:wdw]
        mean1 = df_aux.mean().tolist()
        min1 = df_aux.min().tolist()
        max1 = df_aux.max().tolist()
        std1 = df_aux.std().tolist()
        mean.append(mean1)
        mini.append(min1)
        maxi.append(max1)
        std_dev.append(std1)
        # Removing used data for next window
        cut = int(wdw / 2)
        for i in range(0, cut):
            df.drop(i, inplace = True)
        df = df.reset_index(drop=True)
    # Turning list to dataframes, naming columns and concatenating them together
    df_mean = pd.DataFrame(mean)
    df_mean.columns = ["xmean","ymean","zmean","mvmean"]
    df_min = pd.DataFrame(mini)
    df_min.columns = ["xmin","ymmin","zmin","mvmin"]
    df_max = pd.DataFrame(maxi)
    df_max.columns = ["xmax","ymax","zmax","mvmax"]
    df_std = pd.DataFrame(std_dev)
    df_std.columns = ["xstd","ystd","zstd","mvstd"]
    df_feat = pd.concat([df_mean, df_min, df_max, df_std], axis=1, sort =False)
    return df_feat


def data_feat_fft(df, wdw):
    x = []
    y = []
    z = []
    # Loop with window to extract features
    while wdw < len(df):
        df_aux = df[:5]

        x1 = np.fft.fft(df_aux['x'])
        y1 = np.fft.fft(df_aux['y'])
        z1 = np.fft.fft(df_aux['z'])

        x2 = abs(x1)
        y2 = abs(y1)
        z2 = abs(z1)
        
        x.append(abs(x2))
        y.append(abs(y2))
        z.append(abs(z2))
        
        # Removing used data for next window
        cut = int(wdw / 2)
        for i in range(0, cut):
            df.drop(i, inplace = True)
        df = df.reset_index(drop=True)
    # Turning list to dataframes, naming columns and concatenating them together
    df_x_abs = pd.DataFrame(x)
    df_x_abs.columns = ["x_abs1","x_abs2","x_abs3","x_abs4", "x_abs5"]
    df_y_abs = pd.DataFrame(y)
    df_y_abs.columns = ["y_abs1","y_abs2","y_abs3","y_abs4", "y_abs5"]
    df_z_abs = pd.DataFrame(z)
    df_z_abs.columns = ["z_abs1","z_abs2","z_abs3","z_abs4", "z_abs5"]
    df_feat_fft_abs = pd.concat([df_x_abs, df_y_abs, df_z_abs], axis=1, sort =False)
    return df_feat_fft_abs


def user_col(user, df):
    for _ in df:
        df['user'] = user
    return df


# Final prep steps:
def df_train(user, window, genuine):
    df_raw1 = gen_data(user, 1)
    df_center1 = center(df_raw1)
    magnetude_vector(df_center1)
    window = window_size(window, 40)
    train_feat = data_feat(df_center1, window)
    user_lbl = 0
    if genuine:
        user_lbl = 1
    user_col(user_lbl, train_feat)
    return train_feat


def df_test(user, window):
    df_raw1 = gen_data(user, 2)
    df_center1 = center(df_raw1)
    magnetude_vector(df_center1)
    window = window_size(window, 40)
    test_feat = data_feat(df_center1, window)
    return test_feat


def final_df(user, window):
    df_imp = pd.DataFrame()
    df_gen = pd.DataFrame()
    df_spl = pd.DataFrame()
    df_test_gen = pd.DataFrame()
    df_test_imp = pd.DataFrame()
    df = pd.DataFrame()
    df1 = pd.DataFrame()
    # Creating genuine user data-frame
    counter = 0
    for i in range(0, len(sess1_files)):
        if i == user:
            df_gen = df_train(i, window, True)
        else:
            # Creating impostor users data-frame
            if counter == 0:
                df_imp = df_train(i, window, False)
                counter = 1
            else:
                df = df_train(i, window, False)
                df_imp = pd.concat([df_imp, df], ignore_index=True, sort=False)
    # Balancing impostor user data-frame with user data-frame
    df_spl = df_imp.sample(len(df_gen), random_state=1) 
    counter = 0
    for i in range(0, len(sess2_files)):
        # Creating genuine test user data-frame
        if i == user:
            df_test_gen = df_test(i, window)
        else:
            # Creating impostor test users data-frame
            if counter == 0:
                df_test_imp = df_test(i, window)
                counter = 1
            else:
                df1 = df_test(i, window)
                df_test_imp = pd.concat([df_test_imp, df1], ignore_index=True, sort=False)
    return df_gen, df_spl, df_test_gen, df_test_imp


# Final prep steps:
def df_train_fft(user, window, genuine):
    df_raw1 = gen_data(user, 1)
    df_center1 = center(df_raw1)
    window = window_size(window, 40)
    train_feat = data_feat_fft(df_center1, window)
    magnetude_vector_fft(train_feat)
    user_lbl = 0
    if genuine:
        user_lbl = 1
    user_col(user_lbl, train_feat)
    return train_feat


def df_test_fft(user, window):
    df_raw1 = gen_data(user, 2)
    df_center1 = center(df_raw1)
    window = window_size(window, 40)
    test_feat = data_feat_fft(df_center1, window)
    magnetude_vector_fft(test_feat)
    return test_feat


def final_df_fft(user, window):
    df_imp = pd.DataFrame()
    df_gen = pd.DataFrame()
    df_spl = pd.DataFrame()
    df_test_gen = pd.DataFrame()
    df_test_imp = pd.DataFrame()
    df = pd.DataFrame()
    df1 = pd.DataFrame()
    # Creating genuine user data-frame
    counter = 0
    for i in range(0, len(sess1_files)):
        if i == user:
            df_gen = df_train_fft(i, window, True)
        else:
            # Creating impostor users data-frame
            if counter == 0:
                df_imp = df_train_fft(i, window, False)
                counter = 1
            else:
                df = df_train_fft(i, window, False)
                df_imp = pd.concat([df_imp, df], ignore_index=True, sort=False)
    # Balancing impostor user data-frame with user data-frame
    df_spl = df_imp.sample(len(df_gen), random_state=1) 
    counter = 0
    for i in range(0, len(sess2_files)):
        # Creating genuine test user data-frame
        if i == user:
            df_test_gen = df_test_fft(i, window)
        else:
            # Creating impostor test users data-frame
            if counter == 0:
                df_test_imp = df_test_fft(i, window)
                counter = 1
            else:
                df1 = df_test_fft(i, window)
                df_test_imp = pd.concat([df_test_imp, df1], ignore_index=True, sort=False)
    return df_gen, df_spl, df_test_gen, df_test_imp
'''
def plot_df(feat1, feat2, df):
    sns.lmplot(x=feat1, y=feat2, data=df, hue = 'user', palette='Set1', fit_reg=False, scatter_kws={"s": 70})
#    sns.swarmplot(x='user', y='mvstd', data=features)
    plt.xlabel(feat1)
    plt.ylabel(feat2)
    plt.show()

def plot_both(feat1, feat2, df1, df2):
    plt.scatter((df1[feat1]), (df1[feat2]))
    plt.scatter((df2[feat1]), (df2[feat2]))
    plt.xlabel(feat1)
    plt.ylabel(feat2)
    plt.show()
'''

# CLASSIFIER
def svm_svc(x, y, hue, df_tr, df_ts_g, df_ts_i, knl, c):
    # Specify inputs for the model
    coord = df_tr[[x, y]].values
    type_label = np.where(df_tr[hue]==1, 1, 0)
    
    # Feature names
    df_features = df_tr.columns.values[0:].tolist()
    print(df_features)
    
    # Fit the SVM model
    if knl == 'rbf':
        model = svm.SVC(kernel=knl, C=c, gamma='auto')
        model.fit(coord, type_label)
    else:
        model = svm.SVC(kernel=knl, C = c)
        model.fit(coord, type_label)
        # Get the separating hyperplane
        w = model.coef_[0]
        a = -w[0] / w[1]
        xx = np.linspace(-10, 10)
        yy = a * xx - (model.intercept_[0]) / w[1]
        # Plot the parallels to the separating hyperplane that pass through the support vectors
        b = model.support_vectors_[0]
        yy_down = a * xx + (b[1] - a * b[0])
        b = model.support_vectors_[-1]
        yy_up = a * xx + (b[1] - a * b[0])
        sns.lmplot(x, y, data=df_tr, hue=hue, palette='Set1', fit_reg=False)
        plt.plot(xx, yy, linewidth=2, color='black')
        plt.plot(xx, yy_down, 'k--')
        plt.plot(xx, yy_up, 'k--')
    pred_g = model.predict(df_ts_g[[x, y]])
    #print(pred_g)
    pred_i = model.predict(df_ts_i[[x, y]])
    #print(pred_i)
    # Verificando Métricas
    v_g = len(df_ts_g)
    pre_v_v = 0
    for i in range(0,len(pred_g)):
        if pred_g[i] == 1:
            pre_v_v += 1
    pred_v_f = v_g - pre_v_v
    f_i = len(df_ts_i)
    pred_f_f = 0
    for i in range(0,len(pred_i)):
        if pred_i[i] == 0:
            pred_f_f += 1
    pred_f_v = f_i - pred_f_f
    fmr = pred_f_v/f_i
    fnmr = pred_v_f/v_g
    acc_bal = 1 - ((fmr + fnmr)/2)
    print('kernel=', knl)
    print('Parâmetro C=', c)
    print('Total de instâncias de usuários genuínos: ', v_g)
    print('Total de usuário Verdadeiro/predições Verdadeiro: ', pre_v_v)
    print('Total de usuário Verdadeiro/predições Falso: ', pred_v_f)
    print('Total de instâncias de usuários impostores: ', f_i)
    print('Total de usuário Falso/predições Verdadeiro: ', pred_f_v)
    print('Total de usuário Falso/predições Falso: ', pred_f_f)
    print('False Match Rate: ', fmr)
    print('False Non Match Rate: ', fnmr)
    print('Balanced Accuracy: ', acc_bal)
    plt.show()


def svm_teste(df):
    y = np.array(df['user'])
    X = np.array(df.drop(['user'], axis=1))
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=1)
    parameters = {'kernel':['linear', 'rbf'], 'C':[0.1, 0.5, 1, 5, 10]}
    svc = svm.SVC(gamma="auto")
    clf = GridSearchCV(svc, parameters, cv=5)
    clf.fit(X_train, y_train)
    sorted(clf.cv_results_.keys())
    print('score',clf.score(X_test, y_test))
    return (clf.best_params_)


def main():
    user = int(input("Digite o usuário genuíno (1 a 50): "))
    window = int(input("Digite o tamanho da janela (3s ou 5s ou 7s): "))
    user -= 1
    gen_feat, imp_feat, test_gen, test_imp = final_df(user, window)
    result = pd.concat([gen_feat, imp_feat], ignore_index=True, sort=False)
    gen_feat_fft, imp_feat_fft, test_gen_fft, test_imp_fft = final_df_fft(user, window)
    result_fft = pd.concat([gen_feat_fft, imp_feat_fft], ignore_index=True, sort=False)
    #best = svm_teste(result)
    #best = svm_teste(result_fft)
    #print('Os melhores resultador para parâmetros foram:')
    #print(best)
    #kernel = best["kernel"]
    #c = best["C"]
    print("Métricas obtidas por meio de MÉTODOS ESTATÍSTICOS")
    #svm_svc('mvmean', 'mvstd', 'user', result,test_gen, test_imp, kernel, c)
    svm_svc('xmean', 'mvstd', 'user', result,test_gen, test_imp, 'rbf', 1)
    print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")    
    print("Métricas obtidas por meio do MÉTODOS FFT (Fast Fourrier Transform)")
    #svm_svc('mag_xyz', 'mag_xy', 'user', result_fft,test_gen_fft, test_imp_fft, kernel, c)
    svm_svc('mag_xyz', 'mag_xy', 'user', result_fft,test_gen_fft, test_imp_fft, 'rbf', 1)


if __name__ == '__main__':
    main()
