# -*- coding:utf-8 -*-
import numpy as np
import os

all_txt = "D:/[1]DB/[4]etc_experiment/Body_age/OULP-Age/Age_Gender_All.txt"
ori_train = "D:/[1]DB/[4]etc_experiment/Body_age/OULP-Age/GEI_IDList_train.txt"
ori_test = "D:/[1]DB/[4]etc_experiment/Body_age/OULP-Age/GEI_IDList_test.txt"

def main():
    all_name = np.loadtxt(all_txt, "<U200", skiprows=0, usecols=0)
    all_age = np.loadtxt(all_txt, "<U200", skiprows=0, usecols=1)
    all_gender = np.loadtxt(all_txt, "<U200", skiprows=0, usecols=2)

    ori_train_name = np.loadtxt(ori_train, "<U200", skiprows=0, usecols=0)
    ori_train_name = [data.split("000")[-1] for data in ori_train_name]
    ori_test_name = np.loadtxt(ori_test, "<U200", skiprows=0, usecols=0)
    ori_test_name = [data.split("000")[-1] for data in ori_test_name]

    wr_train = open("D:/[1]DB/[4]etc_experiment/Body_age/OULP-Age/train.txt", "w")
    wr_test = open("D:/[1]DB/[4]etc_experiment/Body_age/OULP-Age/test.txt", "w")

    for i in range(len(all_name)):

        for k in range(len(ori_test_name)):
            if all_name[i] == ori_test_name[k]:
                wr_test.write(all_name[i])
                wr_test.write(" ")
                wr_test.write(all_age[i])
                wr_test.write("\n")
                wr_test.flush()

        if i % 1000 == 0:
            print("Made {} images...".format(i))

    for i in range(len(all_name)):

        for j in range(len(ori_train_name)):
            if all_name[i] == ori_train_name[j]:
                wr_train.write(all_name[i])
                wr_train.write(" ")
                wr_train.write(all_age[i])
                wr_train.write("\n")
                wr_train.flush()

        if i % 1000 == 0:
            print("Made {} images...".format(i))

if __name__ == "__main__":
    main()