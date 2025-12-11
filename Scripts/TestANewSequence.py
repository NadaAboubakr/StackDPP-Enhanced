#!/usr/bin/env python
# coding: utf-8

# # Functions

# In[1]:


import cgi
import csv
import itertools
import math
import os
import platform
import re
import string
import sys
from collections import Counter, namedtuple
from pprint import pprint

import _pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def ProcessSPD(spd3file, plen):
    form = cgi.FieldStorage()
    fo = open(spd3file, "r+")

    spd_str = fo.name + fo.read()
    fo.close()
    spd_str = spd_str.split()
    currentpos = 12
    row = 0
    ss_seq = ''
    ASA = []
    phi = []
    psi = []
    theta = []
    tau = []
    hse_up = []
    hse_down = []
    coil = []
    betaSheet = []
    alphaHelix = []
    while (row < plen):
        ss_seq = ss_seq + spd_str[currentpos + 3]
        data = [float(i) for i in spd_str[currentpos + 4:currentpos + 14]]
        ASA.append(data[0])
        phi.append(data[1])
        psi.append(data[2])
        theta.append(data[3])
        tau.append(data[4])
        hse_up.append(data[5])
        hse_down.append(data[6])
        coil.append(data[-3])
        betaSheet.append(data[-2])
        alphaHelix.append(data[-1])
        row = row + 1
        currentpos = currentpos + 13

    return (ss_seq, ASA, phi, psi, theta, tau, hse_up, hse_down, coil,
            betaSheet, alphaHelix)


def ProcessPSSM(pssmfile):
    form = cgi.FieldStorage()

    # Open PSSM file
    fo = open(pssmfile, "r+")
    str = fo.name + fo.read()
    # Close opend file
    fo.close()
    str = str.split()
    p = str[0:22]
    lastpos = str.index('Lambda')
    lastpos = lastpos - (lastpos % 62) - 4
    currentpos = str.index('Last') + 62
    p_seq = ''
    plen = 0
    pssm = {}
    while (currentpos < lastpos):
        p_no = [int(i) for i in str[currentpos]]
        p_seq = p_seq + str[currentpos + 1]

        pssm[plen] = [int(i) for i in str[currentpos + 2:currentpos + 22]]
        currentpos = currentpos + 44
        plen = plen + 1

    allmin = 999
    allmax = -999
    i = 0
    while (i < plen):
        rowmin = min(pssm[i][0:20])
        if rowmin < allmin:
            allmin = rowmin
        rowmax = max(pssm[i][0:20])
        if rowmax > allmax:
            allmax = rowmax
        i = i + 1
    gmax = allmax
    gmin = allmin

    npssm = []

    diff = float(gmax - gmin)
    for i in range(0, plen):
        a = []
        for j in range(0, 20):
            val = float(pssm[i][j] - gmin)
            val = val / diff
            a.append(val)
        npssm.append(a)

    return (plen, npssm)


def BigramPercentileSeparation(sequence):
    features = []
    values = []

    for char1 in string.ascii_uppercase:
        for char2 in string.ascii_uppercase:
            for k in range(1, 11, 1):
                if (char1 != 'B' and char1 != 'J' and char1 != 'O'
                        and char1 != 'U' and char1 != 'X' and char1 != 'Z'
                        and char2 != 'B' and char2 != 'J' and char2 != 'O'
                        and char2 != 'U' and char2 != 'X' and char2 != 'Z'):
                    features.append("{0}{1}-{2}".format(char1, char2, k * 10))

    pos = {}

    lengthOfSeq = len(sequence)

    cur = 0
    for char1 in string.ascii_uppercase:
        for char2 in string.ascii_uppercase:
            if (char1 != 'B' and char1 != 'J' and char1 != 'O' and char1 != 'U'
                    and char1 != 'X' and char1 != 'Z' and char2 != 'B'
                    and char2 != 'J' and char2 != 'O' and char2 != 'U'
                    and char2 != 'X' and char2 != 'Z'):

                track = 0
                count = 0
                cur = cur + 1

                pos[cur, 0] = 0

                for j in range(1, lengthOfSeq, 1):

                    if (sequence[j - 1] == char1 and sequence[j] == char2):
                        track = track + 1
                        pos[cur, track] = j + 1
                        count = count + 1
                pos[cur, 0] = count  # first index contain the size of the row
                # print((pos[cur, 0]))

    # print(pos)
    cur = 0
    for char1 in string.ascii_uppercase:
        for char2 in string.ascii_uppercase:
            if (char1 != 'B' and char1 != 'J' and char1 != 'O' and char1 != 'U'
                    and char1 != 'X' and char1 != 'Z' and char2 != 'B'
                    and char2 != 'J' and char2 != 'O' and char2 != 'U'
                    and char2 != 'X' and char2 != 'Z'):
                cur = cur + 1
                #print(char1,end=' ')
                for k in range(1, 11, 1):
                    # print(k)
                    value = ((k * 10) / 100) * pos[cur, 0]
                    value = math.floor(value + 0.5)  # print(value,end=' ')
                    if (value == 0):
                        values.append("0")
                    else:
                        values.append("{0}".format(
                            (pos[cur, value]) / lengthOfSeq))
    return (['Bigram_' + x for x in features], values)


def MonogramPercentileSeparation(sequence):
    features = []
    values = []

    for char1 in string.ascii_uppercase:
        for k in range(1, 11, 1):
            if (char1 != 'B' and char1 != 'J' and char1 != 'O' and char1 != 'U'
                    and char1 != 'X' and char1 != 'Z'):
                features.append("{0}-{1}".format(char1, k * 10))

    pos = {}
    lengthOfSeq = len(sequence)
    # print(div)
    for char1 in string.ascii_uppercase:
        if (char1 != 'B' and char1 != 'J' and char1 != 'O' and char1 != 'U'
                and char1 != 'X' and char1 != 'Z'):
            # print(dna_string[i])
            track = 0
            count = 0
            cur = (ord(char1) - ord('A'))
            pos[cur, 0] = 0
            for j in range(0, lengthOfSeq, 1):
                if sequence[j] == char1:
                    track = track + 1
                    pos[cur, track] = j + 1
                    count = count + 1
            pos[cur, 0] = count  # first index contain the size of the row

    for char1 in string.ascii_uppercase:
        if (char1 != 'B' and char1 != 'J' and char1 != 'O' and char1 != 'U'
                and char1 != 'X' and char1 != 'Z'):
            cur = (ord(char1) - ord('A'))
            #print(char1,end=' ')
            for k in range(1, 11, 1):
                # print(k)
                value = ((k * 10) / 100) * pos[cur, 0]
                value = math.floor(value + 0.5)  # print(value,end=' ')
                if (value == 0):
                    values.append("0")
                else:
                    values.append("{0}".format(
                        (pos[cur, value]) / lengthOfSeq))

    return (['Monogram_' + x for x in features], values)


def DDE(sequence):
    AA = 'ACDEFGHIKLMNPQRSTVWY'

    myCodons = {
        'A': 4,
        'C': 2,
        'D': 2,
        'E': 2,
        'F': 2,
        'G': 4,
        'H': 2,
        'I': 3,
        'K': 2,
        'L': 6,
        'M': 1,
        'N': 2,
        'P': 4,
        'Q': 2,
        'R': 6,
        'S': 6,
        'T': 4,
        'V': 4,
        'W': 1,
        'Y': 2
    }

    features = ["DDE_" + aa1 + aa2 for aa1 in AA for aa2 in AA]

    myTM = []
    for pair in features:
        myTM.append((myCodons[pair[0]] / 61) * (myCodons[pair[1]] / 61))

    AADict = {}
    for i in range(len(AA)):
        AADict[AA[i]] = i

    values = [0] * 400
    for j in range(len(sequence) - 2 + 1):
        values[AADict[sequence[j]] * 20 +
               AADict[sequence[j + 1]]] = values[AADict[sequence[j]] * 20 +
                                                 AADict[sequence[j + 1]]] + 1
    if sum(values) != 0:
        values = [i / sum(values) for i in values]

    myTV = []
    for j in range(len(myTM)):
        myTV.append(myTM[j] * (1 - myTM[j]) / (len(sequence) - 1))

    for j in range(len(values)):
        values[j] = (values[j] - myTM[j]) / math.sqrt(myTV[j])

    return (features, values)


def Rvalue(aa1, aa2, AADict, Matrix):
    return sum([(Matrix[i][AADict[aa1]] - Matrix[i][AADict[aa2]])**2
                for i in range(len(Matrix))]) / len(Matrix)


def PAAC(sequence, lambdaValue=30, w=0.05, **kw):
    if len(sequence) < lambdaValue + 1:
        print((
            'Error: all the sequence length should be larger than the lambdaValue+1: '
            + str(lambdaValue + 1) + '\n\n'))
        return 0

    dataFile = 'data/PAAC.txt'
    with open(dataFile) as f:
        records = f.readlines()
    AA = ''.join(records[0].rstrip().split()[1:])
    AADict = {}
    for i in range(len(AA)):
        AADict[AA[i]] = i
    AAProperty = []
    AAPropertyNames = []
    for i in range(1, len(records)):
        array = records[i].rstrip().split(
        ) if records[i].rstrip() != '' else None
        AAProperty.append([float(j) for j in array[1:]])
        AAPropertyNames.append(array[0])

    AAProperty1 = []
    for i in AAProperty:
        meanI = sum(i) / 20
        fenmu = math.sqrt(sum([(j - meanI)**2 for j in i]) / 20)
        AAProperty1.append([(j - meanI) / fenmu for j in i])

    features = []
    for aa in AA:
        features.append('PAAC_' + 'Xc1.' + aa)
    for n in range(1, lambdaValue + 1):
        features.append('PAAC_' + 'Xc2.lambda' + str(n))

    values = []
    theta = []
    for n in range(1, lambdaValue + 1):
        theta.append(
            sum([
                Rvalue(sequence[j], sequence[j + n], AADict, AAProperty1)
                for j in range(len(sequence) - n)
            ]) / (len(sequence) - n))
    myDict = {}
    for aa in AA:
        myDict[aa] = sequence.count(aa)
    values = values + [myDict[aa] / (1 + w * sum(theta)) for aa in AA]
    values = values + [(w * j) / (1 + w * sum(theta)) for j in theta]

    return (features, values)


def CalculateKSCTriad(sequence, gap, features, AADict):
    res = []
    for g in range(gap + 1):
        myDict = {}
        for f in features:
            myDict[f] = 0

        for i in range(len(sequence)):
            if i + g + 1 < len(sequence) and i + 2 * g + 2 < len(sequence):
                fea = AADict[sequence[i]] + '.' +                     AADict[sequence[i+g+1]]+'.'+AADict[sequence[i+2*g+2]]
                myDict[fea] = myDict[fea] + 1

        maxValue, minValue = max(myDict.values()), min(myDict.values())
        for f in features:
            res.append((myDict[f] - minValue) / maxValue)

    return res


def CTriad(sequence):
    AAGroup = {
        'g1': 'AGV',
        'g2': 'ILFP',
        'g3': 'YMTS',
        'g4': 'HNQW',
        'g5': 'RK',
        'g6': 'DE',
        'g7': 'C'
    }

    myGroups = sorted(AAGroup.keys())

    AADict = {}
    for g in myGroups:
        for aa in AAGroup[g]:
            AADict[aa] = g

    feature_suffices = [
        f1 + '.' + f2 + '.' + f3 for f1 in myGroups for f2 in myGroups
        for f3 in myGroups
    ]

    features = []
    for f in feature_suffices:
        features.append("CTriad_" + f)

    values = []
    if len(sequence) < 3:
        print(
            'Error: for "CTriad" encoding, the input fasta sequences should be greater than 3. \n\n'
        )
        return 0
    values = values + CalculateKSCTriad(sequence, 0, feature_suffices, AADict)

    return (features, values)


def Count(seq1, seq2):
    sum = 0
    for aa in seq1:
        sum = sum + seq2.count(aa)
    return sum


def CTDC(sequence):
    group1 = {
        'hydrophobicity_PRAM900101': 'RKEDQN',
        'hydrophobicity_ARGP820101': 'QSTNGDE',
        'hydrophobicity_ZIMJ680101': 'QNGSWTDERA',
        'hydrophobicity_PONP930101': 'KPDESNQT',
        'hydrophobicity_CASG920101': 'KDEQPSRNTG',
        'hydrophobicity_ENGD860101': 'RDKENQHYP',
        'hydrophobicity_FASG890101': 'KERSQD',
        'normwaalsvolume': 'GASTPDC',
        'polarity': 'LIFWCMVY',
        'polarizability': 'GASDT',
        'charge': 'KR',
        'secondarystruct': 'EALMQKRH',
        'solventaccess': 'ALFCGIVW'
    }
    group2 = {
        'hydrophobicity_PRAM900101': 'GASTPHY',
        'hydrophobicity_ARGP820101': 'RAHCKMV',
        'hydrophobicity_ZIMJ680101': 'HMCKV',
        'hydrophobicity_PONP930101': 'GRHA',
        'hydrophobicity_CASG920101': 'AHYMLV',
        'hydrophobicity_ENGD860101': 'SGTAW',
        'hydrophobicity_FASG890101': 'NTPG',
        'normwaalsvolume': 'NVEQIL',
        'polarity': 'PATGS',
        'polarizability': 'CPNVEQIL',
        'charge': 'ANCQGHILMFPSTWYV',
        'secondarystruct': 'VIYCWFT',
        'solventaccess': 'RKQEND'
    }
    group3 = {
        'hydrophobicity_PRAM900101': 'CLVIMFW',
        'hydrophobicity_ARGP820101': 'LYPFIW',
        'hydrophobicity_ZIMJ680101': 'LPFYI',
        'hydrophobicity_PONP930101': 'YMFWLCVI',
        'hydrophobicity_CASG920101': 'FIWC',
        'hydrophobicity_ENGD860101': 'CVLIMF',
        'hydrophobicity_FASG890101': 'AYHWVMFLIC',
        'normwaalsvolume': 'MHKFRYW',
        'polarity': 'HQRKNED',
        'polarizability': 'KMHFRYW',
        'charge': 'DE',
        'secondarystruct': 'GNPSD',
        'solventaccess': 'MSPTHY'
    }

    groups = [group1, group2, group3]
    property = ('hydrophobicity_PRAM900101', 'hydrophobicity_ARGP820101',
                'hydrophobicity_ZIMJ680101', 'hydrophobicity_PONP930101',
                'hydrophobicity_CASG920101', 'hydrophobicity_ENGD860101',
                'hydrophobicity_FASG890101', 'normwaalsvolume', 'polarity',
                'polarizability', 'charge', 'secondarystruct', 'solventaccess')

    features = []
    values = []
    for p in property:
        for g in range(1, len(groups) + 1):
            features.append("CTDC_" + p + '.G' + str(g))

    for p in property:
        c1 = Count(group1[p], sequence) / len(sequence)
        c2 = Count(group2[p], sequence) / len(sequence)
        c3 = 1 - c1 - c2
        values = values + [c1, c2, c3]

    return (features, values)


def QSOrder(sequence, nlag=30, w=0.1):
    if len(sequence) < nlag + 1:
        print((
            'Error: all the sequence length should be larger than the nlag+1: '
            + str(nlag + 1) + '\n\n'))
        return 0

    dataFile = 'data/Schneider-Wrede.txt'
    dataFile1 = 'data/Grantham.txt'

    AA = 'ACDEFGHIKLMNPQRSTVWY'
    AA1 = 'ARNDCQEGHILKMFPSTWYV'

    DictAA = {}
    for i in range(len(AA)):
        DictAA[AA[i]] = i

    DictAA1 = {}
    for i in range(len(AA1)):
        DictAA1[AA1[i]] = i

    with open(dataFile) as f:
        records = f.readlines()[1:]
    AADistance = []
    for i in records:
        array = i.rstrip().split()[1:] if i.rstrip() != '' else None
        AADistance.append(array)
    AADistance = np.array([
        float(AADistance[i][j]) for i in range(len(AADistance))
        for j in range(len(AADistance[i]))
    ]).reshape((20, 20))

    with open(dataFile1) as f:
        records = f.readlines()[1:]
    AADistance1 = []
    for i in records:
        array = i.rstrip().split()[1:] if i.rstrip() != '' else None
        AADistance1.append(array)
    AADistance1 = np.array([
        float(AADistance1[i][j]) for i in range(len(AADistance1))
        for j in range(len(AADistance1[i]))
    ]).reshape((20, 20))

    encodings = []
    features = []
    for aa in AA1:
        features.append('QSOrder_' + 'Schneider.Xr.' + aa)
    for aa in AA1:
        features.append('QSOrder_' + 'Grantham.Xr.' + aa)
    for n in range(1, nlag + 1):
        features.append('QSOrder_' + 'Schneider.Xd.' + str(n))
    for n in range(1, nlag + 1):
        features.append('QSOrder_' + 'Grantham.Xd.' + str(n))
    encodings.append(features)

    arraySW = []
    arrayGM = []
    for n in range(1, nlag + 1):
        arraySW.append(
            sum([
                AADistance[DictAA[sequence[j]]][DictAA[sequence[j + n]]]**2
                for j in range(len(sequence) - n)
            ]))
        arrayGM.append(
            sum([
                AADistance1[DictAA1[sequence[j]]][DictAA1[sequence[j + n]]]**2
                for j in range(len(sequence) - n)
            ]))
    myDict = {}
    values = []
    for aa in AA1:
        myDict[aa] = sequence.count(aa)
    for aa in AA1:
        values.append(myDict[aa] / (1 + w * sum(arraySW)))
    for aa in AA1:
        values.append(myDict[aa] / (1 + w * sum(arrayGM)))
    for num in arraySW:
        values.append((w * num) / (1 + w * sum(arraySW)))
    for num in arrayGM:
        values.append((w * num) / (1 + w * sum(arrayGM)))

    return (features, values)


def APAAC(sequence, lambdaValue=30, w=0.05, **kw):
    if len(sequence) < lambdaValue + 1:
        print((
            'Error: all the sequence length should be larger than the lambdaValue+1: '
            + str(lambdaValue + 1) + '\n\n'))
        return 0

    dataFile = 'data/PAAC.txt'
    with open(dataFile) as f:
        records = f.readlines()
    AA = ''.join(records[0].rstrip().split()[1:])
    AADict = {}
    for i in range(len(AA)):
        AADict[AA[i]] = i
    AAProperty = []
    AAPropertyNames = []
    for i in range(1, len(records) - 1):
        array = records[i].rstrip().split(
        ) if records[i].rstrip() != '' else None
        AAProperty.append([float(j) for j in array[1:]])
        AAPropertyNames.append(array[0])

    AAProperty1 = []
    for i in AAProperty:
        meanI = sum(i) / 20
        fenmu = math.sqrt(sum([(j - meanI)**2 for j in i]) / 20)
        AAProperty1.append([(j - meanI) / fenmu for j in i])

    features = []
    for i in AA:
        features.append('APAAC_' + 'Pc1.' + i)
    for j in range(1, lambdaValue + 1):
        for i in AAPropertyNames:
            features.append('APAAC_' + 'Pc2.' + i + '.' + str(j))

    values = []
    theta = []
    for n in range(1, lambdaValue + 1):
        for j in range(len(AAProperty1)):
            theta.append(
                sum([
                    AAProperty1[j][AADict[sequence[k]]] *
                    AAProperty1[j][AADict[sequence[k + n]]]
                    for k in range(len(sequence) - n)
                ]) / (len(sequence) - n))
    myDict = {}
    for aa in AA:
        myDict[aa] = sequence.count(aa)

    values = values + [myDict[aa] / (1 + w * sum(theta)) for aa in AA]
    values = values + [w * value / (1 + w * sum(theta)) for value in theta]

    return (features, values)


def CountD(aaSet, sequence):
    number = 0
    for aa in sequence:
        if aa in aaSet:
            number = number + 1
    cutoffNums = [
        1,
        math.floor(0.25 * number),
        math.floor(0.50 * number),
        math.floor(0.75 * number), number
    ]
    cutoffNums = [i if i >= 1 else 1 for i in cutoffNums]

    values = []
    for cutoff in cutoffNums:
        myCount = 0
        for i in range(len(sequence)):
            if sequence[i] in aaSet:
                myCount += 1
                if myCount == cutoff:
                    values.append((i + 1) / len(sequence) * 100)
                    break
        if myCount == 0:
            values.append(0)
    return values


def CTDD(sequence, **kw):
    group1 = {
        'hydrophobicity_PRAM900101': 'RKEDQN',
        'hydrophobicity_ARGP820101': 'QSTNGDE',
        'hydrophobicity_ZIMJ680101': 'QNGSWTDERA',
        'hydrophobicity_PONP930101': 'KPDESNQT',
        'hydrophobicity_CASG920101': 'KDEQPSRNTG',
        'hydrophobicity_ENGD860101': 'RDKENQHYP',
        'hydrophobicity_FASG890101': 'KERSQD',
        'normwaalsvolume': 'GASTPDC',
        'polarity': 'LIFWCMVY',
        'polarizability': 'GASDT',
        'charge': 'KR',
        'secondarystruct': 'EALMQKRH',
        'solventaccess': 'ALFCGIVW'
    }
    group2 = {
        'hydrophobicity_PRAM900101': 'GASTPHY',
        'hydrophobicity_ARGP820101': 'RAHCKMV',
        'hydrophobicity_ZIMJ680101': 'HMCKV',
        'hydrophobicity_PONP930101': 'GRHA',
        'hydrophobicity_CASG920101': 'AHYMLV',
        'hydrophobicity_ENGD860101': 'SGTAW',
        'hydrophobicity_FASG890101': 'NTPG',
        'normwaalsvolume': 'NVEQIL',
        'polarity': 'PATGS',
        'polarizability': 'CPNVEQIL',
        'charge': 'ANCQGHILMFPSTWYV',
        'secondarystruct': 'VIYCWFT',
        'solventaccess': 'RKQEND'
    }
    group3 = {
        'hydrophobicity_PRAM900101': 'CLVIMFW',
        'hydrophobicity_ARGP820101': 'LYPFIW',
        'hydrophobicity_ZIMJ680101': 'LPFYI',
        'hydrophobicity_PONP930101': 'YMFWLCVI',
        'hydrophobicity_CASG920101': 'FIWC',
        'hydrophobicity_ENGD860101': 'CVLIMF',
        'hydrophobicity_FASG890101': 'AYHWVMFLIC',
        'normwaalsvolume': 'MHKFRYW',
        'polarity': 'HQRKNED',
        'polarizability': 'KMHFRYW',
        'charge': 'DE',
        'secondarystruct': 'GNPSD',
        'solventaccess': 'MSPTHY'
    }

    groups = [group1, group2, group3]
    property = ('hydrophobicity_PRAM900101', 'hydrophobicity_ARGP820101',
                'hydrophobicity_ZIMJ680101', 'hydrophobicity_PONP930101',
                'hydrophobicity_CASG920101', 'hydrophobicity_ENGD860101',
                'hydrophobicity_FASG890101', 'normwaalsvolume', 'polarity',
                'polarizability', 'charge', 'secondarystruct', 'solventaccess')

    features = []
    for p in property:
        for g in ('1', '2', '3'):
            for d in ['0', '25', '50', '75', '100']:
                features.append("CTDD_" + p + '.' + g + '.residue' + d)

    values = []
    for p in property:
        values = values + CountD(group1[p], sequence) +             CountD(group2[p], sequence) + CountD(group3[p], sequence)
    return (features, values)


def Geary(sequence,
          props=[
              'CIDH920105', 'BHAR880101', 'CHAM820101', 'CHAM820102',
              'CHOC760101', 'BIGC670101', 'CHAM810101', 'DAYM780201'
          ],
          nlag=30,
          **kw):
    if len(sequence) < nlag + 1:
        print((
            'Error: all the sequence length should be larger than the nlag+1: '
            + str(nlag + 1) + '\n\n'))
        return 0

    AA = 'ARNDCQEGHILKMFPSTWYV'
    fileAAidx = re.sub('codes$', '',
                       os.path.split(os.path.realpath(__file__))
                       [0]) + r'\data\AAidx.txt' if platform.system(
                       ) == 'Windows' else sys.path[0] + '/data/AAidx.txt'
    with open(fileAAidx) as f:
        records = f.readlines()[1:]
    myDict = {}
    for i in records:
        array = i.rstrip().split('\t')
        myDict[array[0]] = array[1:]

    AAidx = []
    AAidxName = []
    for i in props:
        if i in myDict:
            AAidx.append(myDict[i])
            AAidxName.append(i)
        else:
            print(('"' + i + '" properties not exist.'))
            return None

    AAidx1 = np.array([float(j) for i in AAidx for j in i])
    AAidx = AAidx1.reshape((len(AAidx), 20))

    propMean = np.mean(AAidx, axis=1)
    propStd = np.std(AAidx, axis=1)

    for i in range(len(AAidx)):
        for j in range(len(AAidx[i])):
            AAidx[i][j] = (AAidx[i][j] - propMean[i]) / propStd[i]

    index = {}
    for i in range(len(AA)):
        index[AA[i]] = i

    features = []
    for p in props:
        for n in range(1, nlag + 1):
            features.append("Geary_" + p + '.lag' + str(n))

    values = []
    N = len(sequence)
    for prop in range(len(props)):
        xmean = sum([AAidx[prop][index[aa]] for aa in sequence]) / N
        for n in range(1, nlag + 1):
            if len(sequence) > nlag:
                # if key is '-', then the value is 0
                rn = (N - 1) / (2 * (N - n)) * (
                    (sum([(AAidx[prop][index.get(sequence[j], 0)] -
                           AAidx[prop][index.get(sequence[j + n], 0)])**2
                          for j in range(len(sequence) - n)])) /
                    (sum([(AAidx[prop][index.get(sequence[j], 0)] - xmean)**2
                          for j in range(len(sequence))])))
            else:
                rn = 'NA'
            values.append(rn)

    return (features, values)


def CalculateKSCTriad(sequence, gap, features, AADict):
    res = []
    for g in range(gap + 1):
        myDict = {}
        for f in features:
            myDict[f] = 0

        for i in range(len(sequence)):
            if i + g + 1 < len(sequence) and i + 2 * g + 2 < len(sequence):
                fea = AADict[sequence[i]] + '.' +                     AADict[sequence[i+g+1]]+'.'+AADict[sequence[i+2*g+2]]
                if (fea not in myDict):
                    myDict[fea] = 0
                myDict[fea] = myDict[fea] + 1

        maxValue, minValue = max(myDict.values()), min(myDict.values())
        for f in features:
            res.append((myDict[f] - minValue) / maxValue)

    return res


def KSCTriad(sequence, gap=0, **kw):
    AAGroup = {
        'g1': 'AGV',
        'g2': 'ILFP',
        'g3': 'YMTS',
        'g4': 'HNQW',
        'g5': 'RK',
        'g6': 'DE',
        'g7': 'C'
    }

    myGroups = sorted(AAGroup.keys())

    AADict = {}
    for g in myGroups:
        for aa in AAGroup[g]:
            AADict[aa] = g

    featuresMarkers = [
        f1 + '.' + f2 + '.' + f3 for f1 in myGroups for f2 in myGroups
        for f3 in myGroups
    ]

    features = []
    for g in range(gap + 1):
        for f in featuresMarkers:
            features.append("KSCTriad_" + f + '.gap' + str(g))

    values = []
    if len(sequence) < 2 * gap + 3:
        print(
            'Error: for "KSCTriad" encoding, the input fasta sequences should be greater than (2*gap+3). \n\n'
        )
        return 0
    values = values + CalculateKSCTriad(sequence, gap, features, AADict)

    return (features, values)


def GTPC(sequence, **kw):
    group = {
        'alphaticr': 'GAVLMI',
        'aromatic': 'FYW',
        'postivecharger': 'KRH',
        'negativecharger': 'DE',
        'uncharger': 'STCPNQ'
    }

    groupKey = list(group.keys())
    baseNum = len(groupKey)
    triple = [
        g1 + '.' + g2 + '.' + g3 for g1 in groupKey for g2 in groupKey
        for g3 in groupKey
    ]

    index = {}
    for key in groupKey:
        for aa in group[key]:
            index[aa] = key

    triple_h = ["GTPC_" + item for item in triple]
    features = triple_h

    values = []
    myDict = {}
    for t in triple:
        myDict[t] = 0

    sum = 0
    for j in range(len(sequence) - 3 + 1):
        myDict[index[sequence[j]] + '.' + index[sequence[j + 1]] + '.' +
               index[sequence[j + 2]]] = myDict[index[sequence[j]] + '.' +
                                                index[sequence[j + 1]] + '.' +
                                                index[sequence[j + 2]]] + 1
        sum = sum + 1

    if sum == 0:
        for t in triple:
            values.append(0)
    else:
        for t in triple:
            values.append(myDict[t] / sum)

    return (features, values)


def GDPC(sequence, **kw):
    group = {
        'alphaticr': 'GAVLMI',
        'aromatic': 'FYW',
        'postivecharger': 'KRH',
        'negativecharger': 'DE',
        'uncharger': 'STCPNQ'
    }

    groupKey = list(group.keys())
    baseNum = len(groupKey)
    dipeptide = [g1 + '.' + g2 for g1 in groupKey for g2 in groupKey]

    index = {}
    for key in groupKey:
        for aa in group[key]:
            index[aa] = key

    dipeptide_h = ["GDPC_" + item for item in dipeptide]
    features = dipeptide_h

    values = []
    myDict = {}
    for t in dipeptide:
        myDict[t] = 0

    sum = 0
    for j in range(len(sequence) - 2 + 1):
        myDict[index[sequence[j]] + '.' +
               index[sequence[j + 1]]] = myDict[index[sequence[j]] + '.' +
                                                index[sequence[j + 1]]] + 1
        sum = sum + 1

    if sum == 0:
        for t in dipeptide:
            values.append(0)
    else:
        for t in dipeptide:
            values.append(myDict[t] / sum)

    return (features, values)


def GAAC(sequence, **kw):
    group = {
        'alphatic': 'GAVLMI',
        'aromatic': 'FYW',
        'postivecharge': 'KRH',
        'negativecharge': 'DE',
        'uncharge': 'STCPNQ'
    }

    groupKey = list(group.keys())

    features = []
    for key in groupKey:
        features.append("GAAC_" + key)

    values = []
    count = Counter(sequence)
    myDict = {}
    for key in groupKey:
        for aa in group[key]:
            myDict[key] = myDict.get(key, 0) + count[aa]

    for key in groupKey:
        values.append(myDict[key] / len(sequence))

    return (features, values)


def Moran(sequence,
          props=[
              'CIDH920105', 'BHAR880101', 'CHAM820101', 'CHAM820102',
              'CHOC760101', 'BIGC670101', 'CHAM810101', 'DAYM780201'
          ],
          nlag=30,
          **kw):

    if len(sequence) < nlag + 1:
        print((
            'Error: all the sequence length should be larger than the nlag+1: '
            + str(nlag + 1) + '\n\n'))
        return 0

    AA = 'ARNDCQEGHILKMFPSTWYV'
    fileAAidx = re.sub('codes$', '',
                       os.path.split(os.path.realpath(__file__))
                       [0]) + r'\data\AAidx.txt' if platform.system(
                       ) == 'Windows' else sys.path[0] + '/data/AAidx.txt'

    with open(fileAAidx) as f:
        records = f.readlines()[1:]
    myDict = {}
    for i in records:
        array = i.rstrip().split('\t')
        myDict[array[0]] = array[1:]

    AAidx = []
    AAidxName = []
    for i in props:
        if i in myDict:
            AAidx.append(myDict[i])
            AAidxName.append(i)
        else:
            print(('"' + i + '" properties not exist.'))
            return None

    AAidx1 = np.array([float(j) for i in AAidx for j in i])
    AAidx = AAidx1.reshape((len(AAidx), 20))

    propMean = np.mean(AAidx, axis=1)
    propStd = np.std(AAidx, axis=1)

    for i in range(len(AAidx)):
        for j in range(len(AAidx[i])):
            AAidx[i][j] = (AAidx[i][j] - propMean[i]) / propStd[i]

    index = {}
    for i in range(len(AA)):
        index[AA[i]] = i

    features = []
    for p in props:
        for n in range(1, nlag + 1):
            features.append("Moran_" + p + '.lag' + str(n))

    values = []
    N = len(sequence)
    for prop in range(len(props)):
        xmean = sum([AAidx[prop][index[aa]] for aa in sequence]) / N
        for n in range(1, nlag + 1):
            if len(sequence) > nlag:
                # if key is '-', then the value is 0
                fenzi = sum(
                    [(AAidx[prop][index.get(sequence[j], 0)] - xmean) *
                     (AAidx[prop][index.get(sequence[j + n], 0)] - xmean)
                     for j in range(len(sequence) - n)]) / (N - n)
                fenmu = sum([(AAidx[prop][index.get(sequence[j], 0)] - xmean)**
                             2 for j in range(len(sequence))]) / N
                rn = fenzi / fenmu
            else:
                rn = 'NA'
            values.append(rn)

    return (features, values)


def SOCNumber(sequence, nlag=30, **kw):
    if len(sequence) < nlag + 1:
        print((
            'Error: all the sequence length should be larger than the nlag+1: '
            + str(nlag + 1) + '\n\n'))
        return 0

    dataFile = 'data/Schneider-Wrede.txt'
    dataFile1 = 'data/Grantham.txt'
    AA = 'ACDEFGHIKLMNPQRSTVWY'
    AA1 = 'ARNDCQEGHILKMFPSTWYV'

    DictAA = {}
    for i in range(len(AA)):
        DictAA[AA[i]] = i

    DictAA1 = {}
    for i in range(len(AA1)):
        DictAA1[AA1[i]] = i

    with open(dataFile) as f:
        records = f.readlines()[1:]
    AADistance = []
    for i in records:
        array = i.rstrip().split()[1:] if i.rstrip() != '' else None
        AADistance.append(array)
    AADistance = np.array([
        float(AADistance[i][j]) for i in range(len(AADistance))
        for j in range(len(AADistance[i]))
    ]).reshape((20, 20))

    with open(dataFile1) as f:
        records = f.readlines()[1:]
    AADistance1 = []
    for i in records:
        array = i.rstrip().split()[1:] if i.rstrip() != '' else None
        AADistance1.append(array)
    AADistance1 = np.array([
        float(AADistance1[i][j]) for i in range(len(AADistance1))
        for j in range(len(AADistance1[i]))
    ]).reshape((20, 20))

    features = []
    for n in range(1, nlag + 1):
        features.append('SOCNumber_' + 'Schneider.lag' + str(n))
    for n in range(1, nlag + 1):
        features.append('SOCNumber_' + 'gGrantham.lag' + str(n))

    values = []
    for n in range(1, nlag + 1):
        values.append(
            sum([
                AADistance[DictAA[sequence[j]]][DictAA[sequence[j + n]]]**2
                for j in range(len(sequence) - n)
            ]) / (len(sequence) - n))

    for n in range(1, nlag + 1):
        values.append(
            sum([
                AADistance1[DictAA1[sequence[j]]][DictAA1[sequence[j + n]]]**2
                for j in range(len(sequence) - n)
            ]) / (len(sequence) - n))

    return (features, values)


def descriptors(Sequence, G1, G2, G3):
    # print G1
    import math
    # print "Hello : ", G1, G2, G3
    Amino = 'ARNDCQEGHILKMFPSTWYV'

    S = Sequence
    seq = []

    for j in range(0, len(S)):
        L1 = []
        L2 = []
        L3 = []
        for k in range(0, len(G1)):
            if G1[k] == S[j]:
                L1.append(1)
            else:
                L1.append(0)
        for k in range(0, len(G2)):
            if G2[k] == S[j]:
                L2.append(1)
            else:
                L2.append(0)
        for k in range(0, len(G3)):
            if G3[k] == S[j]:
                L3.append(1)
            else:
                L3.append(0)
        if sum(L1) == 1:
            seq.append(1.0)
        elif sum(L2) == 1:
            seq.append(2.0)
        elif sum(L3) == 1:
            seq.append(3.0)

    length = len(seq)
    G = 3  # G is the number of groups

    # Occurence
    Occ = []
    for j in range(0, len(Amino)):
        count = []
        for k in range(0, len(S)):
            if S[k] == Amino[j]:
                count.append(1.0)
            else:
                count.append(0.0)
        # print count
        Occ.append(sum(count) / length)
    # print Occ

    # Composition
    Comp = []
    for j in range(0, G):
        count = []
        for k in range(0, len(seq)):
            if seq[k] == (j + 1):
                count.append(1.0)
            else:
                count.append(0.0)
        Comp.append(sum(count) / length)
    # print Comp

    # Transition
    t1 = []
    for k in range(0, len(seq)):
        if seq[k] == 1:
            t1.append(1.0)
        else:
            t1.append(0.0)
    t2 = []
    for k in range(0, len(seq)):
        if seq[k] == 2:
            t2.append(1.0)
        else:
            t2.append(0.0)
    t3 = []
    for k in range(0, len(seq)):
        if seq[k] == 3:
            t3.append(1.0)
        else:
            t3.append(0.0)

    # print t2
    # print t3
    T12 = 0
    T13 = 0
    T23 = 0
    for j in range(0, length - 1):
        s1 = t1[j] + t1[j + 1]
        s2 = t2[j] + t2[j + 1]  # sum(t2[j:j+1]);
        s3 = t3[j] + t3[j + 1]  # sum(t3[j:j+1]);
        # print s1, s2, s3
        if s1 >= 1 and s2 >= 1:
            T12 = T12 + 1
        if s1 >= 1 and s3 >= 1:
            T13 = T13 + 1
        if s2 >= 1 and s3 >= 1:
            T23 = T23 + 1
    # print T12, T13, T23

    # Distribution
    P = [0.25, 0.5, 0.75, 1]
    lenP = len(P)
    D1 = []
    D2 = []
    D3 = []

    c1 = []
    r1 = t1
    for m in range(0, len(t1)):
        c1.append(m)
    # Sorting descending order , with index
    for m in range(0, len(r1) - 1):
        for n in range(m + 1, len(r1)):
            if r1[n] > r1[m]:
                # print m, n
                temp = r1[n]
                r1[n] = r1[m]
                r1[m] = temp
                ct = c1[n]
                c1[n] = c1[m]
                c1[m] = ct

    c2 = []
    r2 = t2
    for m in range(0, len(t2)):
        c2.append(m)
    # Sorting descending order , with index
    for m in range(0, len(r2) - 1):
        for n in range(m + 1, len(r2)):
            if r2[n] > r2[m]:
                # print m, n
                temp = r2[n]
                r2[n] = r2[m]
                r2[m] = temp
                ct = c2[n]
                c2[n] = c2[m]
                c2[m] = ct

    c3 = []
    r3 = t3
    for m in range(0, len(t3)):
        c3.append(m)
    # Sorting descending order , with index
    for m in range(0, len(r3) - 1):
        for n in range(m + 1, len(r3)):
            if r3[n] > r3[m]:
                # print m, n
                temp = r3[n]
                r3[n] = r3[m]
                r3[m] = temp
                ct = c3[n]
                c3[n] = c3[m]
                c3[m] = ct

    if r1[1] != 0.0:
        D1.append(float(c1[1]) / float(length))
    else:
        D1.append(0)
    if r2[1] != 0.0:
        D2.append(float(c2[1]) / float(length))
    else:
        D2.append(0)
    if r3[1] != 0.0:
        D3.append(float(c3[1]) / float(length))
    else:
        D3.append(0)

    for k in range(0, lenP):
        P1 = int(math.ceil(sum(t1) * P[k]))
        if P1 != 0:
            D1.append(float(c1[P1]) / float(length))
        else:
            D1.append(0)
        P2 = int(math.ceil(sum(t2) * P[k]))
        if P2 == len(c2):
            D2.append(1)
        else:
            if P2 != 0:
                D2.append(float(c2[P2]) / float(length))
            else:
                D2.append(0)
        P3 = int(math.ceil(sum(t3) * P[k]))
        if P3 != 0:
            D3.append(float(c3[P3]) / float(length))
        else:
            D3.append(0)

    D = [D1, D2, D3]

    feature = []
    feature.append(Comp[0])
    feature.append(Comp[2])
    feature.append(Comp[2])
    feature.append(T12)
    feature.append(T13)
    feature.append(T23)
    feature.append(D1[0])
    feature.append(D1[1])
    feature.append(D1[2])
    feature.append(D1[3])
    feature.append(D1[4])
    feature.append(D2[0])
    feature.append(D2[1])
    feature.append(D2[2])
    feature.append(D2[3])
    feature.append(D2[4])
    feature.append(D3[0])
    feature.append(D3[1])
    feature.append(D3[2])
    feature.append(D3[3])
    feature.append(D3[4])

    return feature


def Dubchak(Sequence):
    #!/usr/bin/python

    # Import module support
    #import support

    #Sequence = 'GPLGSGSKIKLEIYNETDMASASGYTPVPSVSEFQYIETETISNTPSPDLTVMSIDKSVLSPGESATITTIVKDIDGNPVNEVHINKTVARENLKGLWDYGPLKKENVPGKYTQVITYRGHSNERIDISFKYAMSFTKEISIRGRL'

    # Now you can call defined function that module as follows
    # support.print_func(Sequence)

    N = 6  # N=[0,5], if N=5 then use all descriptors

    values = []  # [1, 2]
    for j in range(0, N):
        if j == 1:
            # Hydrophobicity
            G1 = 'RKEDQN'
            G2 = 'GASTPHY'
            G3 = 'CVLIMFW'
            v = descriptors(Sequence, G1, G2, G3)
            for k in range(0, len(v)):
                values.append(v[k])
        if j == 2:
            # Normalized van der Waals volumns
            G1 = ['G', 'A', 'S', 'C', 'T', 'P', 'D']
            G2 = ['N', 'V', 'E', 'Q', 'I', 'L']
            G3 = ['M', 'H', 'K', 'F', 'R', 'Y', 'W']
            v = descriptors(Sequence, G1, G2, G3)
            for k in range(0, len(v)):
                values.append(v[k])
        if j == 3:
            # Polarity
            G1 = ['L', 'I', 'F', 'W', 'C', 'M', 'V', 'Y']
            G2 = ['P', 'A', 'T', 'G', 'S']
            G3 = ['H', 'Q', 'R', 'K', 'N', 'E', 'D']
            v = descriptors(Sequence, G1, G2, G3)
            for k in range(0, len(v)):
                values.append(v[k])
        if j == 4:
            # Polarizability
            G1 = ['G', 'A', 'S', 'D', 'T']
            G2 = ['C', 'P', 'N', 'V', 'E', 'Q', 'A', 'L']
            G3 = ['K', 'M', 'H', 'F', 'R', 'Y', 'W']
            v = descriptors(Sequence, G1, G2, G3)
            for k in range(0, len(v)):
                values.append(v[k])
        if j == 5:
            # Normalized frequency of alpha-helix
            G1 = ['G', 'P', 'N', 'Y', 'C', 'S', 'T']
            G2 = ['R', 'H', 'D', 'V', 'W', 'I']
            G3 = ['Q', 'F', 'K', 'L', 'A', 'M', 'E']
            v = descriptors(Sequence, G1, G2, G3)
            for k in range(0, len(v)):
                values.append(v[k])

        features = ['dubchak_' + str(x) for x in range(0, 105, 1)]

    return (features, values)


def PSSMBigram(plen, npssm):
    col = 20
    bigram_t = []
    bigram = []
    for i in range(0, col):
        for j in range(0, plen - 1):
            a = []
            if j + 1 < len(npssm):
                for k in range(0, col):
                    val = npssm[j][i] * npssm[j + 1][k]
                    a.append(val)
            bigram_t.append(a)
        bb = []
        for j in range(0, col):
            b = []
            for k in range(0, len(bigram_t) - 1):
                if k < len(bigram_t):
                    b.append(bigram_t[k][j])
            bb.append((sum(b) / plen))
        bigram.append(bb)

    maxCol = len(bigram[0])
    for row in bigram:
        rowLength = len(row)
        if rowLength > maxCol:
            maxCol = rowLength
    values = []
    for colIndex in range(maxCol):
        values.append([])
        for row in bigram:
            if colIndex < len(row):
                values[colIndex].append(row[colIndex])

    features = ['pssm_bigram_' + str(x) for x in range(0, 400, 1)]

    values = list(itertools.chain.from_iterable(values))
    return (features, values)


def PSSMAutoCovariance(plen, npssm):
    DF = 10
    col = 20
    # parameter for ACV
    acc_t = []
    acc = []
    for i in range(0, DF):
        for j in range(0, plen - i):
            a = []
            if j + 1 < len(npssm):
                for k in range(0, col):
                    val = npssm[j][k] * npssm[j + i][k]
                    a.append(val)
            acc_t.append(a)
        bb = []
        for j in range(0, col):
            b = []
            for k in range(0, len(acc_t) - 1):
                if k < len(acc_t):
                    b.append(acc_t[k][j])
            bb.append((sum(b) / plen))
        acc.append(bb)
        acc_t = []

    # Finding transpose of acc
    maxCol = len(acc[0])
    for row in acc:
        rowLength = len(row)
        if rowLength > maxCol:
            maxCol = rowLength
    values = []
    for colIndex in range(maxCol):
        values.append([])
        for row in acc:
            if colIndex < len(row):
                values[colIndex].append(row[colIndex])
    features = ['pssm_auto-covariance_' + str(x) for x in range(0, 200, 1)]
    values = list(itertools.chain.from_iterable(values))
    return (features, values)


def OneLeadBigramPSSM(plen, npssm):
    col = 20
    one_l_bigram_t = []
    one_l_bigram = []
    for i in range(0, col):
        for j in range(0, plen - 2):
            a = []
            if j + 2 < len(npssm):
                for k in range(0, col):
                    val = npssm[j][i] * npssm[j + 2][k]
                    a.append(val)
            one_l_bigram_t.append(a)
        bb = []
        for j in range(0, col):
            b = []
            for k in range(0, len(one_l_bigram_t) - 1):
                if k < len(one_l_bigram_t):
                    b.append(one_l_bigram_t[k][j])
            bb.append((sum(b) / plen))
        one_l_bigram.append(bb)
        one_l_bigram_t = []

    maxCol = len(one_l_bigram[0])
    for row in one_l_bigram:
        rowLength = len(row)
        if rowLength > maxCol:
            maxCol = rowLength
    values = []
    for colIndex in range(maxCol):
        values.append([])
        for row in one_l_bigram:
            if colIndex < len(row):
                values[colIndex].append(row[colIndex])

    features = ['one_lead_bigram_' + str(x) for x in range(0, 400, 1)]

    values = list(itertools.chain.from_iterable(values))
    return (features, values)


def PSSMSegmentDistribution(npssm):
    Fp = 10
    values = []
    k = 0
    for j in range(0, 20):
        Tj = 0
        for m in range(0, len(npssm)):
            Tj = Tj + npssm[m][j]

        partialsum = 0
        i = 0
        tp = Fp
        while (tp <= 50):  # in range(Fp,50):
            tpTj = tp * Tj / 100
            while (partialsum <= tpTj and i <= len(npssm)):
                partialsum = partialsum + npssm[i][j]
                i = i + 1
            values.append(i)
            # features.append(i)
            k = k + 1
            tp = tp + Fp
        # print k, tp
        partialsum = 0
        i = len(npssm) - 1
        index = 0
        tp = Fp
        while (tp <= 50):  # for tp in range(Fp,50):
            while (partialsum <= tp * Tj / 100 and i >= 0):
                partialsum = partialsum + npssm[i][j]
                i = i - 1
                index = index + 1
            values.append(index)
            k = k + 1
            tp = tp + Fp

    features = ['segment_distribution_' + str(x) for x in range(0, 200, 1)]

    return (features, values)


def SecondaryStructureComposition(ss_seq):
    values = []
    SS = 'CHE'
    for j in range(0, len(SS)):
        count = 0
        for i in range(0, len(ss_seq)):
            if ss_seq[i] == SS[j]:
                count = count + 1
        values.append(count)

    features = [
        'secondary_structure_composition_' + str(x) for x in range(0, 3, 1)
    ]

    return (features, values)


def SecondaryStructureOccurance(ss_seq):
    occ = SecondaryStructureComposition(ss_seq)[1]
    values = []
    values.append(float(occ[0]) / float(len(ss_seq)))
    values.append(float(occ[1]) / float(len(ss_seq)))
    values.append(float(occ[2]) / float(len(ss_seq)))

    features = [
        'secondary_structure_occurance_' + str(x) for x in range(0, 3, 1)
    ]

    return (features, values)


def ASA_AngleOccurance_ProbCHE(ss_seq, phi, psi, theta, tau, coil, betaSheet,
                               alphaHelix, ASA):
    values = []
    pi = math.pi
    a = b = c = d = e = f = g = h = m = n = p = q = 0.0
    for i in range(0, len(ss_seq)):
        a = a + ASA[i]
        b = b + math.sin(phi[i] * pi / 180)
        c = c + math.cos(phi[i] * pi / 180)
        d = d + math.sin(psi[i] * pi / 180)
        e = e + math.cos(psi[i] * pi / 180)
        f = f + math.sin(theta[i] * pi / 180)
        g = g + math.cos(theta[i] * pi / 180)
        h = h + math.sin(tau[i] * pi / 180)
        m = m + math.cos(tau[i] * pi / 180)
        n = n + coil[i]
        p = p + betaSheet[i]
        q = q + alphaHelix[i]

    values.append(a / len(ss_seq))
    values.append(b / len(ss_seq))
    values.append(c / len(ss_seq))
    values.append(d / len(ss_seq))
    values.append(e / len(ss_seq))
    values.append(f / len(ss_seq))
    values.append(g / len(ss_seq))
    values.append(h / len(ss_seq))
    values.append(m / len(ss_seq))
    values.append(n / len(ss_seq))
    values.append(p / len(ss_seq))
    values.append(q / len(ss_seq))

    features = [
        'ASA_angle_occurance_probability_of_CHE_' + str(x)
        for x in range(0, 12, 1)
    ]

    return (features, values)


def GetAngles(ss_seq, phi, psi, theta, tau):
    a = []
    angles = []
    for i in range(0, len(ss_seq)):
        a.append(math.sin(phi[i] * math.pi / 180))
        a.append(math.cos(phi[i] * math.pi / 180))
        a.append(math.sin(psi[i] * math.pi / 180))
        a.append(math.cos(psi[i] * math.pi / 180))
        a.append(math.sin(theta[i] * math.pi / 180))
        a.append(math.cos(theta[i] * math.pi / 180))
        a.append(math.sin(tau[i] * math.pi / 180))
        a.append(math.cos(tau[i] * math.pi / 180))
        angles.append(a)
        a = []
    return angles


def TorsionalAnglesBigram(ss_seq, phi, psi, theta, tau):
    a_bigram_t = []
    a_bigram = []
    angles = GetAngles(ss_seq, phi, psi, theta, tau)

    a = []
    for i in range(0, 8):
        for j in range(0, len(ss_seq) - 1):
            for k in range(0, 8):
                a.append(angles[j][i] * angles[j + 1][k])
                a_bigram_t.append(a)

        bb = []
        for j in range(0, 8):
            b = []
            for k in range(0, len(a_bigram_t) - 1):
                b.append(a_bigram_t[k][j])
            bb.append((sum(b) / len(ss_seq)))
        a_bigram.append(bb)

    # Finding transpose of a_bigram
    maxCol = len(a_bigram[0])
    for row in a_bigram:
        rowLength = len(row)
        if rowLength > maxCol:
            maxCol = rowLength

    values = []
    for colIndex in range(maxCol):
        values.append([])
        for row in a_bigram:
            if colIndex < len(row):
                values[colIndex].append(row[colIndex])

    features = ['bigram_angle_sine_cosine_' + str(x) for x in range(0, 64, 1)]
    values = list(itertools.chain.from_iterable(values))
    return (features, values)


def TorsionalAnglesAutoCovariance(ss_seq, phi, psi, theta, tau):
    DF = 10  # parameter for ACV
    angles_acc_t = []
    angles_acc = []
    angles = GetAngles(ss_seq, phi, psi, theta, tau)
    a = []
    for i in range(0, DF):
        for j in range(0, len(ss_seq) - i):
            for k in range(0, 8):
                a.append(angles[j][k] * angles[j + i][k])
            angles_acc_t.append(a)

        bb = []
        for j in range(0, 8):
            b = []
            for k in range(0, len(angles_acc_t) - 1):
                b.append(angles_acc_t[k][j])
            bb.append((sum(b) / len(ss_seq)))
        angles_acc.append(bb)
        angles_acc_t = []

    # print angles_acc

    # Finding transpose of angles_acc
    maxCol = len(angles_acc[0])
    for row in angles_acc:
        rowLength = len(row)
        if rowLength > maxCol:
            maxCol = rowLength
    values = []
    for colIndex in range(maxCol):
        values.append([])
        for row in angles_acc:
            if colIndex < len(row):
                values[colIndex].append(row[colIndex])

    features = ['angles_auto_covariance_' + str(x) for x in range(0, 80, 1)]
    values = list(itertools.chain.from_iterable(values))
    return (features, values)


def StructuralProbabilitiesBigram(ss_seq, coil, betaSheet, alphaHelix):
    prob_bigram_t = []
    prob_bigram = []
    a = []
    for i in range(0, 3):
        for j in range(0, len(ss_seq) - 1):
            # for k in range(0,3):
            if i == 0:
                a.append(coil[j] * coil[j + 1])
                a.append(coil[j] * betaSheet[j + 1])
                a.append(coil[j] * alphaHelix[j + 1])
            if i == 1:
                a.append(betaSheet[j] * coil[j + 1])
                a.append(betaSheet[j] * betaSheet[j + 1])
                a.append(betaSheet[j] * alphaHelix[j + 1])
            if i == 2:
                a.append(alphaHelix[j] * coil[j + 1])
                a.append(alphaHelix[j] * betaSheet[j + 1])
                a.append(alphaHelix[j] * betaSheet[j + 1])
            prob_bigram_t.append(a)
            a = []

        bb = []
        for j in range(0, 3):
            b = []
            for k in range(0, len(prob_bigram_t) - 1):
                b.append(prob_bigram_t[k][j])
            bb.append((sum(b) / len(ss_seq)))
        prob_bigram.append(bb)
        prob_bigram_t = []

    # print prob_bigram
    # Finding transpose of prob_bigram
    maxCol = len(prob_bigram[0])
    for row in prob_bigram:
        rowLength = len(row)
        if rowLength > maxCol:
            maxCol = rowLength
    values = []
    for colIndex in range(maxCol):
        values.append([])
        for row in prob_bigram:
            if colIndex < len(row):
                values[colIndex].append(row[colIndex])

    features = ['bigram_probabilities_' + str(x) for x in range(0, 9, 1)]
    values = list(itertools.chain.from_iterable(values))
    return (features, values)


def StructuralProbabilitesAutoCovariance(ss_seq, coil, betaSheet, alphaHelix):
    DF = 10  # parameter for ACV
    prob_acc_t = []
    prob_acc = []
    a = []
    for i in range(0, DF):
        for j in range(0, len(ss_seq) - i):
            a.append(coil[j] * coil[j + i])
            a.append(betaSheet[j] * betaSheet[j + i])
            a.append(alphaHelix[j] * alphaHelix[j + i])
            prob_acc_t.append(a)
            a = []

        bb = []
        for j in range(0, 3):
            b = []
            for k in range(0, len(prob_acc_t) - 1):
                b.append(prob_acc_t[k][j])
            bb.append((sum(b) / len(ss_seq)))
        prob_acc.append(bb)
        prob_acc_t = []

    # print prob_acc

    # Finding transpose of prob_acc
    maxCol = len(prob_acc[0])
    for row in prob_acc:
        rowLength = len(row)
        if rowLength > maxCol:
            maxCol = rowLength
    values = []
    for colIndex in range(maxCol):
        values.append([])
        for row in prob_acc:
            if colIndex < len(row):
                values[colIndex].append(row[colIndex])

    features = [
        'probabilities_auto_covariance_' + str(x) for x in range(0, 30, 1)
    ]
    values = list(itertools.chain.from_iterable(values))
    return (features, values)


##################-----------------------#####################
##################-----------------------#####################
##################-----------------------#####################
##################-----------------------#####################


def NNB(sequence):
    x = 0
    new = np.empty([])
    n = 1
    # nearest neigbour
    count = np.zeros((n, 400))
    for i in range(0, n, 1):
        length = len(sequence[i])
        letter = 0
        finish = 0
        for char1 in string.ascii_uppercase:
            for char2 in string.ascii_uppercase:
                if (char1 != 'B' and char2 != 'B' and char1 != 'J'
                        and char2 != 'J' and char1 != 'O' and char2 != 'O'
                        and char1 != 'U' and char2 != 'U' and char1 != 'X'
                        and char2 != 'X' and char1 != 'Z' and char2 != 'Z'):
                    detect = 0
                    for j in range(0, length, 1):
                        if (sequence[i][j] == char1):  # detect char1
                            total = 0
                            detect = 1
                            flag = 0
                            for k in range(j + 1, length, 1):
                                if (sequence[i][k] == char2
                                    ):  # match with char2
                                    flag = 1
                                    break
                                total = total + 1
                            if (flag == 1):
                                count[i][letter] = total + 1
                                letter = letter + 1
                                finish = 1
                            else:
                                count[i][letter] = 0
                                letter = letter + 1
                                finish = 1
                        if (finish == 1):
                            finish = 0
                            break
                    if (detect == 0):
                        count[i][letter] = 0
                        # print("{0}d1{1}={2}".format(char1,char2,count[i][letter]))
                        letter = letter + 1
    """for i in range(0,n,1):
        for j in range(0,400,1):
            print("{0},".format(count[i][j]))
        print("\n")
    """

    # 2nd nearest neigbour
    count1 = np.zeros((n, 400))
    for i in range(0, n, 1):
        length = len(sequence[i])
        letter = 0
        finish = 0
        skip = 0
        for char1 in string.ascii_uppercase:
            for char2 in string.ascii_uppercase:
                if (char1 != 'B' and char2 != 'B' and char1 != 'J'
                        and char2 != 'J' and char1 != 'O' and char2 != 'O'
                        and char1 != 'U' and char2 != 'U' and char1 != 'X'
                        and char2 != 'X' and char1 != 'Z' and char2 != 'Z'):
                    detect = 0
                    skip = 0
                    for j in range(0, length, 1):
                        if (sequence[i][j] == char1):  # detect char1
                            total = 0
                            detect = 1
                            flag = 0
                            skip = 0
                            for k in range(j + 1, length, 1):
                                if (sequence[i][k] == char2
                                        and skip == 1):  # match with char2
                                    flag = 1
                                    break
                                if (sequence[i][k] == char2 and skip != 1):
                                    skip = skip + 1
                                total = total + 1

                            if (flag == 1):
                                count1[i][letter] = total + 1
                                letter = letter + 1
                                finish = 1
                            else:
                                count1[i][letter] = 0
                                letter = letter + 1
                                finish = 1

                        if (finish == 1):
                            finish = 0
                            break
                    if (detect == 0):
                        count1[i][letter] = 0
                        # print("{0}d2{1}={2}".format(char1,char2,count1[i][letter]))
                        letter = letter + 1

    # 3rd nearest neigbour
    count2 = np.zeros((n, 400))
    for i in range(0, n, 1):
        length = len(sequence[i])
        letter = 0
        finish = 0
        skip = 0
        for char1 in string.ascii_uppercase:
            for char2 in string.ascii_uppercase:
                if (char1 != 'B' and char2 != 'B' and char1 != 'J'
                        and char2 != 'J' and char1 != 'O' and char2 != 'O'
                        and char1 != 'U' and char2 != 'U' and char1 != 'X'
                        and char2 != 'X' and char1 != 'Z' and char2 != 'Z'):
                    detect = 0
                    skip = 0
                    for j in range(0, length, 1):
                        if (sequence[i][j] == char1):  # detect char1
                            total = 0
                            detect = 1
                            flag = 0
                            skip = 0
                            for k in range(j + 1, length, 1):
                                if (sequence[i][k] == char2
                                        and skip == 2):  # match with char2
                                    flag = 1
                                    break
                                if (sequence[i][k] == char2 and skip != 2):
                                    skip = skip + 1
                                total = total + 1

                            if (flag == 1):
                                count2[i][letter] = total + 1
                                letter = letter + 1
                                finish = 1
                            else:
                                count2[i][letter] = 0
                                letter = letter + 1
                                finish = 1

                        if (finish == 1):
                            finish = 0
                            break
                    if (detect == 0):
                        count2[i][letter] = 0
                        # print("{0}d3{1}={2}".format(char1,char2,count2[i][letter]))
                        letter = letter + 1

    # 4th nearest neigbour
    count3 = np.zeros((n, 400))
    for i in range(0, n, 1):
        length = len(sequence[i])
        letter = 0
        finish = 0
        skip = 0
        for char1 in string.ascii_uppercase:
            for char2 in string.ascii_uppercase:
                if (char1 != 'B' and char2 != 'B' and char1 != 'J'
                        and char2 != 'J' and char1 != 'O' and char2 != 'O'
                        and char1 != 'U' and char2 != 'U' and char1 != 'X'
                        and char2 != 'X' and char1 != 'Z' and char2 != 'Z'):
                    detect = 0
                    skip = 0
                    for j in range(0, length, 1):
                        if (sequence[i][j] == char1):  # detect char1
                            total = 0
                            detect = 1
                            flag = 0
                            skip = 0
                            for k in range(j + 1, length, 1):
                                if (sequence[i][k] == char2
                                        and skip == 3):  # match with char2
                                    flag = 1
                                    break
                                if (sequence[i][k] == char2 and skip != 3):
                                    skip = skip + 1
                                total = total + 1

                            if (flag == 1):
                                count3[i][letter] = total + 1
                                letter = letter + 1
                                finish = 1
                            else:
                                count3[i][letter] = 0
                                letter = letter + 1
                                finish = 1

                        if (finish == 1):
                            finish = 0
                            break
                    if (detect == 0):
                        count3[i][letter] = 0
                        # print("{0}d4{1}={2}".format(char1,char2,count3[i][letter]))
                        letter = letter + 1

    # 5th nearest neigbour
    count4 = np.zeros((n, 400))
    for i in range(0, n, 1):
        length = len(sequence[i])
        letter = 0
        finish = 0
        skip = 0
        for char1 in string.ascii_uppercase:
            for char2 in string.ascii_uppercase:
                if (char1 != 'B' and char2 != 'B' and char1 != 'J'
                        and char2 != 'J' and char1 != 'O' and char2 != 'O'
                        and char1 != 'U' and char2 != 'U' and char1 != 'X'
                        and char2 != 'X' and char1 != 'Z' and char2 != 'Z'):
                    detect = 0
                    skip = 0
                    for j in range(0, length, 1):
                        if (sequence[i][j] == char1):  # detect char1
                            total = 0
                            detect = 1
                            flag = 0
                            skip = 0
                            for k in range(j + 1, length, 1):
                                if (sequence[i][k] == char2
                                        and skip == 4):  # match with char2
                                    flag = 1
                                    break
                                if (sequence[i][k] == char2 and skip != 4):
                                    skip = skip + 1
                                total = total + 1

                            if (flag == 1):
                                count4[i][letter] = total + 1
                                letter = letter + 1
                                finish = 1
                            else:
                                count4[i][letter] = 0
                                letter = letter + 1
                                finish = 1

                        if (finish == 1):
                            finish = 0
                            break
                    if (detect == 0):
                        count4[i][letter] = 0
                        letter = letter + 1

    # 6th nearest neigbour
    count5 = np.zeros((n, 400))
    for i in range(0, n, 1):
        length = len(sequence[i])
        letter = 0
        finish = 0
        skip = 0
        for char1 in string.ascii_uppercase:
            for char2 in string.ascii_uppercase:
                if (char1 != 'B' and char2 != 'B' and char1 != 'J'
                        and char2 != 'J' and char1 != 'O' and char2 != 'O'
                        and char1 != 'U' and char2 != 'U' and char1 != 'X'
                        and char2 != 'X' and char1 != 'Z' and char2 != 'Z'):
                    detect = 0
                    skip = 0
                    for j in range(0, length, 1):
                        if (sequence[i][j] == char1):  # detect char1
                            total = 0
                            detect = 1
                            flag = 0
                            skip = 0
                            for k in range(j + 1, length, 1):
                                if (sequence[i][k] == char2
                                        and skip == 5):  # match with char2
                                    flag = 1
                                    break
                                if (sequence[i][k] == char2 and skip != 5):
                                    skip = skip + 1
                                total = total + 1

                            if (flag == 1):
                                count5[i][letter] = total + 1
                                letter = letter + 1
                                finish = 1
                            else:
                                count5[i][letter] = 0
                                letter = letter + 1
                                finish = 1

                        if (finish == 1):
                            finish = 0
                            break
                    if (detect == 0):
                        count5[i][letter] = 0
                        letter = letter + 1

    # 7th nearest neigbour
    count6 = np.zeros((n, 400))
    for i in range(0, n, 1):
        length = len(sequence[i])
        letter = 0
        finish = 0
        skip = 0
        for char1 in string.ascii_uppercase:
            for char2 in string.ascii_uppercase:
                if (char1 != 'B' and char2 != 'B' and char1 != 'J'
                        and char2 != 'J' and char1 != 'O' and char2 != 'O'
                        and char1 != 'U' and char2 != 'U' and char1 != 'X'
                        and char2 != 'X' and char1 != 'Z' and char2 != 'Z'):
                    detect = 0
                    skip = 0
                    for j in range(0, length, 1):
                        if (sequence[i][j] == char1):  # detect char1
                            total = 0
                            detect = 1
                            flag = 0
                            skip = 0
                            for k in range(j + 1, length, 1):
                                if (sequence[i][k] == char2
                                        and skip == 6):  # match with char2
                                    flag = 1
                                    break
                                if (sequence[i][k] == char2 and skip != 6):
                                    skip = skip + 1
                                total = total + 1

                            if (flag == 1):
                                count6[i][letter] = total + 1
                                letter = letter + 1
                                finish = 1
                            else:
                                count6[i][letter] = 0
                                letter = letter + 1
                                finish = 1

                        if (finish == 1):
                            finish = 0
                            break
                    if (detect == 0):
                        count6[i][letter] = 0
                        letter = letter + 1

    # 8th nearest neigbour
    count7 = np.zeros((n, 400))
    for i in range(0, n, 1):
        length = len(sequence[i])
        letter = 0
        finish = 0
        skip = 0
        for char1 in string.ascii_uppercase:
            for char2 in string.ascii_uppercase:
                if (char1 != 'B' and char2 != 'B' and char1 != 'J'
                        and char2 != 'J' and char1 != 'O' and char2 != 'O'
                        and char1 != 'U' and char2 != 'U' and char1 != 'X'
                        and char2 != 'X' and char1 != 'Z' and char2 != 'Z'):
                    detect = 0
                    skip = 0
                    for j in range(0, length, 1):
                        if (sequence[i][j] == char1):  # detect char1
                            total = 0
                            detect = 1
                            flag = 0
                            skip = 0
                            for k in range(j + 1, length, 1):
                                if (sequence[i][k] == char2
                                        and skip == 7):  # match with char2
                                    flag = 1
                                    break
                                if (sequence[i][k] == char2 and skip != 7):
                                    skip = skip + 1
                                total = total + 1

                            if (flag == 1):
                                count7[i][letter] = total + 1
                                letter = letter + 1
                                finish = 1
                            else:
                                count7[i][letter] = 0
                                letter = letter + 1
                                finish = 1

                        if (finish == 1):
                            finish = 0
                            break
                    if (detect == 0):
                        count7[i][letter] = 0
                        letter = letter + 1

    # 9th nearest neigbour
    count8 = np.zeros((n, 400))
    for i in range(0, n, 1):
        length = len(sequence[i])
        letter = 0
        finish = 0
        skip = 0
        for char1 in string.ascii_uppercase:
            for char2 in string.ascii_uppercase:
                if (char1 != 'B' and char2 != 'B' and char1 != 'J'
                        and char2 != 'J' and char1 != 'O' and char2 != 'O'
                        and char1 != 'U' and char2 != 'U' and char1 != 'X'
                        and char2 != 'X' and char1 != 'Z' and char2 != 'Z'):
                    detect = 0
                    skip = 0
                    for j in range(0, length, 1):
                        if (sequence[i][j] == char1):  # detect char1
                            total = 0
                            detect = 1
                            flag = 0
                            skip = 0
                            for k in range(j + 1, length, 1):
                                if (sequence[i][k] == char2
                                        and skip == 8):  # match with char2
                                    flag = 1
                                    break
                                if (sequence[i][k] == char2 and skip != 8):
                                    skip = skip + 1
                                total = total + 1

                            if (flag == 1):
                                count8[i][letter] = total + 1
                                letter = letter + 1
                                finish = 1
                            else:
                                count8[i][letter] = 0
                                letter = letter + 1
                                finish = 1

                        if (finish == 1):
                            finish = 0
                            break
                    if (detect == 0):
                        count8[i][letter] = 0
                        letter = letter + 1

    # 10th nearest neigbour
    count9 = np.zeros((n, 400))
    for i in range(0, n, 1):
        length = len(sequence[i])
        letter = 0
        finish = 0
        skip = 0
        for char1 in string.ascii_uppercase:
            for char2 in string.ascii_uppercase:
                if (char1 != 'B' and char2 != 'B' and char1 != 'J'
                        and char2 != 'J' and char1 != 'O' and char2 != 'O'
                        and char1 != 'U' and char2 != 'U' and char1 != 'X'
                        and char2 != 'X' and char1 != 'Z' and char2 != 'Z'):
                    detect = 0
                    skip = 0
                    for j in range(0, length, 1):
                        if (sequence[i][j] == char1):  # detect char1
                            total = 0
                            detect = 1
                            flag = 0
                            skip = 0
                            for k in range(j + 1, length, 1):
                                if (sequence[i][k] == char2
                                        and skip == 9):  # match with char2
                                    flag = 1
                                    break
                                if (sequence[i][k] == char2 and skip != 9):
                                    skip = skip + 1
                                total = total + 1

                            if (flag == 1):
                                count9[i][letter] = total + 1
                                letter = letter + 1
                                finish = 1
                            else:
                                count9[i][letter] = 0
                                letter = letter + 1
                                finish = 1

                        if (finish == 1):
                            finish = 0
                            break
                    if (detect == 0):
                        count9[i][letter] = 0
                        letter = letter + 1

    # 11th nearest neigbour
    count10 = np.zeros((n, 400))
    for i in range(0, n, 1):
        length = len(sequence[i])
        letter = 0
        finish = 0
        skip = 0
        for char1 in string.ascii_uppercase:
            for char2 in string.ascii_uppercase:
                if (char1 != 'B' and char2 != 'B' and char1 != 'J'
                        and char2 != 'J' and char1 != 'O' and char2 != 'O'
                        and char1 != 'U' and char2 != 'U' and char1 != 'X'
                        and char2 != 'X' and char1 != 'Z' and char2 != 'Z'):
                    detect = 0
                    skip = 0
                    for j in range(0, length, 1):
                        if (sequence[i][j] == char1):  # detect char1
                            total = 0
                            detect = 1
                            flag = 0
                            skip = 0
                            for k in range(j + 1, length, 1):
                                if (sequence[i][k] == char2
                                        and skip == 10):  # match with char2
                                    flag = 1
                                    break
                                if (sequence[i][k] == char2 and skip != 10):
                                    skip = skip + 1
                                total = total + 1

                            if (flag == 1):
                                count10[i][letter] = total + 1
                                letter = letter + 1
                                finish = 1
                            else:
                                count10[i][letter] = 0
                                letter = letter + 1
                                finish = 1

                        if (finish == 1):
                            finish = 0
                            break
                    if (detect == 0):
                        count10[i][letter] = 0
                        letter = letter + 1

    # 12th nearest neigbour
    count11 = np.zeros((n, 400))
    for i in range(0, n, 1):
        length = len(sequence[i])
        letter = 0
        finish = 0
        skip = 0
        for char1 in string.ascii_uppercase:
            for char2 in string.ascii_uppercase:
                if (char1 != 'B' and char2 != 'B' and char1 != 'J'
                        and char2 != 'J' and char1 != 'O' and char2 != 'O'
                        and char1 != 'U' and char2 != 'U' and char1 != 'X'
                        and char2 != 'X' and char1 != 'Z' and char2 != 'Z'):
                    detect = 0
                    skip = 0
                    for j in range(0, length, 1):
                        if (sequence[i][j] == char1):  # detect char1
                            total = 0
                            detect = 1
                            flag = 0
                            skip = 0
                            for k in range(j + 1, length, 1):
                                if (sequence[i][k] == char2
                                        and skip == 11):  # match with char2
                                    flag = 1
                                    break
                                if (sequence[i][k] == char2 and skip != 11):
                                    skip = skip + 1
                                total = total + 1

                            if (flag == 1):
                                count11[i][letter] = total + 1
                                letter = letter + 1
                                finish = 1
                            else:
                                count11[i][letter] = 0
                                letter = letter + 1
                                finish = 1

                        if (finish == 1):
                            finish = 0
                            break
                    if (detect == 0):
                        count11[i][letter] = 0
                        letter = letter + 1

    # 13th nearest neigbour
    count12 = np.zeros((n, 400))
    for i in range(0, n, 1):
        length = len(sequence[i])
        letter = 0
        finish = 0
        skip = 0
        for char1 in string.ascii_uppercase:
            for char2 in string.ascii_uppercase:
                if (char1 != 'B' and char2 != 'B' and char1 != 'J'
                        and char2 != 'J' and char1 != 'O' and char2 != 'O'
                        and char1 != 'U' and char2 != 'U' and char1 != 'X'
                        and char2 != 'X' and char1 != 'Z' and char2 != 'Z'):
                    detect = 0
                    skip = 0
                    for j in range(0, length, 1):
                        if (sequence[i][j] == char1):  # detect char1
                            total = 0
                            detect = 1
                            flag = 0
                            skip = 0
                            for k in range(j + 1, length, 1):
                                if (sequence[i][k] == char2
                                        and skip == 12):  # match with char2
                                    flag = 1
                                    break
                                if (sequence[i][k] == char2 and skip != 12):
                                    skip = skip + 1
                                total = total + 1

                            if (flag == 1):
                                count12[i][letter] = total + 1
                                letter = letter + 1
                                finish = 1
                            else:
                                count12[i][letter] = 0
                                letter = letter + 1
                                finish = 1

                        if (finish == 1):
                            finish = 0
                            break
                    if (detect == 0):
                        count12[i][letter] = 0
                        letter = letter + 1

    # 14th nearest neigbour
    count13 = np.zeros((n, 400))
    for i in range(0, n, 1):
        length = len(sequence[i])
        letter = 0
        finish = 0
        skip = 0
        for char1 in string.ascii_uppercase:
            for char2 in string.ascii_uppercase:
                if (char1 != 'B' and char2 != 'B' and char1 != 'J'
                        and char2 != 'J' and char1 != 'O' and char2 != 'O'
                        and char1 != 'U' and char2 != 'U' and char1 != 'X'
                        and char2 != 'X' and char1 != 'Z' and char2 != 'Z'):
                    detect = 0
                    skip = 0
                    for j in range(0, length, 1):
                        if (sequence[i][j] == char1):  # detect char1
                            total = 0
                            detect = 1
                            flag = 0
                            skip = 0
                            for k in range(j + 1, length, 1):
                                if (sequence[i][k] == char2
                                        and skip == 13):  # match with char2
                                    flag = 1
                                    break
                                if (sequence[i][k] == char2 and skip != 13):
                                    skip = skip + 1
                                total = total + 1

                            if (flag == 1):
                                count13[i][letter] = total + 1
                                letter = letter + 1
                                finish = 1
                            else:
                                count13[i][letter] = 0
                                letter = letter + 1
                                finish = 1

                        if (finish == 1):
                            finish = 0
                            break
                    if (detect == 0):
                        count13[i][letter] = 0
                        letter = letter + 1

    # 15th nearest neigbour
    count14 = np.zeros((n, 400))
    for i in range(0, n, 1):
        length = len(sequence[i])
        letter = 0
        finish = 0
        skip = 0
        for char1 in string.ascii_uppercase:
            for char2 in string.ascii_uppercase:
                if (char1 != 'B' and char2 != 'B' and char1 != 'J'
                        and char2 != 'J' and char1 != 'O' and char2 != 'O'
                        and char1 != 'U' and char2 != 'U' and char1 != 'X'
                        and char2 != 'X' and char1 != 'Z' and char2 != 'Z'):
                    detect = 0
                    skip = 0
                    for j in range(0, length, 1):
                        if (sequence[i][j] == char1):  # detect char1
                            total = 0
                            detect = 1
                            flag = 0
                            skip = 0
                            for k in range(j + 1, length, 1):
                                if (sequence[i][k] == char2
                                        and skip == 14):  # match with char2
                                    flag = 1
                                    break
                                if (sequence[i][k] == char2 and skip != 14):
                                    skip = skip + 1
                                total = total + 1

                            if (flag == 1):
                                count14[i][letter] = total + 1
                                letter = letter + 1
                                finish = 1
                            else:
                                count14[i][letter] = 0
                                letter = letter + 1
                                finish = 1

                        if (finish == 1):
                            finish = 0
                            break
                    if (detect == 0):
                        count14[i][letter] = 0
                        letter = letter + 1

    # 16th nearest neigbour
    count15 = np.zeros((n, 400))
    for i in range(0, n, 1):
        length = len(sequence[i])
        letter = 0
        finish = 0
        skip = 0
        for char1 in string.ascii_uppercase:
            for char2 in string.ascii_uppercase:
                if (char1 != 'B' and char2 != 'B' and char1 != 'J'
                        and char2 != 'J' and char1 != 'O' and char2 != 'O'
                        and char1 != 'U' and char2 != 'U' and char1 != 'X'
                        and char2 != 'X' and char1 != 'Z' and char2 != 'Z'):
                    detect = 0
                    skip = 0
                    for j in range(0, length, 1):
                        if (sequence[i][j] == char1):  # detect char1
                            total = 0
                            detect = 1
                            flag = 0
                            skip = 0
                            for k in range(j + 1, length, 1):
                                if (sequence[i][k] == char2
                                        and skip == 15):  # match with char2
                                    flag = 1
                                    break
                                if (sequence[i][k] == char2 and skip != 15):
                                    skip = skip + 1
                                total = total + 1

                            if (flag == 1):
                                count15[i][letter] = total + 1
                                letter = letter + 1
                                finish = 1
                            else:
                                count15[i][letter] = 0
                                letter = letter + 1
                                finish = 1

                        if (finish == 1):
                            finish = 0
                            break
                    if (detect == 0):
                        count15[i][letter] = 0
                        letter = letter + 1

    # 17th nearest neigbour
    count16 = np.zeros((n, 400))
    for i in range(0, n, 1):
        length = len(sequence[i])
        letter = 0
        finish = 0
        skip = 0
        for char1 in string.ascii_uppercase:
            for char2 in string.ascii_uppercase:
                if (char1 != 'B' and char2 != 'B' and char1 != 'J'
                        and char2 != 'J' and char1 != 'O' and char2 != 'O'
                        and char1 != 'U' and char2 != 'U' and char1 != 'X'
                        and char2 != 'X' and char1 != 'Z' and char2 != 'Z'):
                    detect = 0
                    skip = 0
                    for j in range(0, length, 1):
                        if (sequence[i][j] == char1):  # detect char1
                            total = 0
                            detect = 1
                            flag = 0
                            skip = 0
                            for k in range(j + 1, length, 1):
                                if (sequence[i][k] == char2
                                        and skip == 16):  # match with char2
                                    flag = 1
                                    break
                                if (sequence[i][k] == char2 and skip != 16):
                                    skip = skip + 1
                                total = total + 1

                            if (flag == 1):
                                count16[i][letter] = total + 1
                                letter = letter + 1
                                finish = 1
                            else:
                                count16[i][letter] = 0
                                letter = letter + 1
                                finish = 1

                        if (finish == 1):
                            finish = 0
                            break
                    if (detect == 0):
                        count16[i][letter] = 0
                        letter = letter + 1

    # 18th nearest neigbour
    count17 = np.zeros((n, 400))
    for i in range(0, n, 1):
        length = len(sequence[i])
        letter = 0
        finish = 0
        skip = 0
        for char1 in string.ascii_uppercase:
            for char2 in string.ascii_uppercase:
                if (char1 != 'B' and char2 != 'B' and char1 != 'J'
                        and char2 != 'J' and char1 != 'O' and char2 != 'O'
                        and char1 != 'U' and char2 != 'U' and char1 != 'X'
                        and char2 != 'X' and char1 != 'Z' and char2 != 'Z'):
                    detect = 0
                    skip = 0
                    for j in range(0, length, 1):
                        if (sequence[i][j] == char1):  # detect char1
                            total = 0
                            detect = 1
                            flag = 0
                            skip = 0
                            for k in range(j + 1, length, 1):
                                if (sequence[i][k] == char2
                                        and skip == 17):  # match with char2
                                    flag = 1
                                    break
                                if (sequence[i][k] == char2 and skip != 17):
                                    skip = skip + 1
                                total = total + 1

                            if (flag == 1):
                                count17[i][letter] = total + 1
                                letter = letter + 1
                                finish = 1
                            else:
                                count17[i][letter] = 0
                                letter = letter + 1
                                finish = 1

                        if (finish == 1):
                            finish = 0
                            break
                    if (detect == 0):
                        count17[i][letter] = 0
                        letter = letter + 1

    # 19th nearest neigbour
    count18 = np.zeros((n, 400))
    for i in range(0, n, 1):
        length = len(sequence[i])
        letter = 0
        finish = 0
        skip = 0
        for char1 in string.ascii_uppercase:
            for char2 in string.ascii_uppercase:
                if (char1 != 'B' and char2 != 'B' and char1 != 'J'
                        and char2 != 'J' and char1 != 'O' and char2 != 'O'
                        and char1 != 'U' and char2 != 'U' and char1 != 'X'
                        and char2 != 'X' and char1 != 'Z' and char2 != 'Z'):
                    detect = 0
                    skip = 0
                    for j in range(0, length, 1):
                        if (sequence[i][j] == char1):  # detect char1
                            total = 0
                            detect = 1
                            flag = 0
                            skip = 0
                            for k in range(j + 1, length, 1):
                                if (sequence[i][k] == char2
                                        and skip == 18):  # match with char2
                                    flag = 1
                                    break
                                if (sequence[i][k] == char2 and skip != 18):
                                    skip = skip + 1
                                total = total + 1

                            if (flag == 1):
                                count18[i][letter] = total + 1
                                letter = letter + 1
                                finish = 1
                            else:
                                count18[i][letter] = 0
                                letter = letter + 1
                                finish = 1

                        if (finish == 1):
                            finish = 0
                            break
                    if (detect == 0):
                        count18[i][letter] = 0
                        letter = letter + 1

    # 20th nearest neigbour
    count19 = np.zeros((n, 400))
    for i in range(0, n, 1):
        length = len(sequence[i])
        letter = 0
        finish = 0
        skip = 0
        for char1 in string.ascii_uppercase:
            for char2 in string.ascii_uppercase:
                if (char1 != 'B' and char2 != 'B' and char1 != 'J'
                        and char2 != 'J' and char1 != 'O' and char2 != 'O'
                        and char1 != 'U' and char2 != 'U' and char1 != 'X'
                        and char2 != 'X' and char1 != 'Z' and char2 != 'Z'):
                    detect = 0
                    skip = 0
                    for j in range(0, length, 1):
                        if (sequence[i][j] == char1):  # detect char1
                            total = 0
                            detect = 1
                            flag = 0
                            skip = 0
                            for k in range(j + 1, length, 1):
                                if (sequence[i][k] == char2
                                        and skip == 19):  # match with char2
                                    flag = 1
                                    break
                                if (sequence[i][k] == char2 and skip != 19):
                                    skip = skip + 1
                                total = total + 1

                            if (flag == 1):
                                count19[i][letter] = total + 1
                                letter = letter + 1
                                finish = 1
                            else:
                                count19[i][letter] = 0
                                letter = letter + 1
                                finish = 1

                        if (finish == 1):
                            finish = 0
                            break
                    if (detect == 0):
                        count19[i][letter] = 0
                        letter = letter + 1

    # 21th nearest neigbour
    count20 = np.zeros((n, 400))
    for i in range(0, n, 1):
        length = len(sequence[i])
        letter = 0
        finish = 0
        skip = 0
        for char1 in string.ascii_uppercase:
            for char2 in string.ascii_uppercase:
                if (char1 != 'B' and char2 != 'B' and char1 != 'J'
                        and char2 != 'J' and char1 != 'O' and char2 != 'O'
                        and char1 != 'U' and char2 != 'U' and char1 != 'X'
                        and char2 != 'X' and char1 != 'Z' and char2 != 'Z'):
                    detect = 0
                    skip = 0
                    for j in range(0, length, 1):
                        if (sequence[i][j] == char1):  # detect char1
                            total = 0
                            detect = 1
                            flag = 0
                            skip = 0
                            for k in range(j + 1, length, 1):
                                if (sequence[i][k] == char2
                                        and skip == 20):  # match with char2
                                    flag = 1
                                    break
                                if (sequence[i][k] == char2 and skip != 20):
                                    skip = skip + 1
                                total = total + 1

                            if (flag == 1):
                                count20[i][letter] = total + 1
                                letter = letter + 1
                                finish = 1
                            else:
                                count20[i][letter] = 0
                                letter = letter + 1
                                finish = 1

                        if (finish == 1):
                            finish = 0
                            break
                    if (detect == 0):
                        count20[i][letter] = 0
                        letter = letter + 1

    # 22th nearest neigbour
    count21 = np.zeros((n, 400))
    for i in range(0, n, 1):
        length = len(sequence[i])
        letter = 0
        finish = 0
        skip = 0
        for char1 in string.ascii_uppercase:
            for char2 in string.ascii_uppercase:
                if (char1 != 'B' and char2 != 'B' and char1 != 'J'
                        and char2 != 'J' and char1 != 'O' and char2 != 'O'
                        and char1 != 'U' and char2 != 'U' and char1 != 'X'
                        and char2 != 'X' and char1 != 'Z' and char2 != 'Z'):
                    detect = 0
                    skip = 0
                    for j in range(0, length, 1):
                        if (sequence[i][j] == char1):  # detect char1
                            total = 0
                            detect = 1
                            flag = 0
                            skip = 0
                            for k in range(j + 1, length, 1):
                                if (sequence[i][k] == char2
                                        and skip == 21):  # match with char2
                                    flag = 1
                                    break
                                if (sequence[i][k] == char2 and skip != 21):
                                    skip = skip + 1
                                total = total + 1

                            if (flag == 1):
                                count21[i][letter] = total + 1
                                letter = letter + 1
                                finish = 1
                            else:
                                count21[i][letter] = 0
                                letter = letter + 1
                                finish = 1

                        if (finish == 1):
                            finish = 0
                            break
                    if (detect == 0):
                        count21[i][letter] = 0
                        letter = letter + 1

    # 23th nearest neigbour
    count22 = np.zeros((n, 400))
    for i in range(0, n, 1):
        length = len(sequence[i])
        letter = 0
        finish = 0
        skip = 0
        for char1 in string.ascii_uppercase:
            for char2 in string.ascii_uppercase:
                if (char1 != 'B' and char2 != 'B' and char1 != 'J'
                        and char2 != 'J' and char1 != 'O' and char2 != 'O'
                        and char1 != 'U' and char2 != 'U' and char1 != 'X'
                        and char2 != 'X' and char1 != 'Z' and char2 != 'Z'):
                    detect = 0
                    skip = 0
                    for j in range(0, length, 1):
                        if (sequence[i][j] == char1):  # detect char1
                            total = 0
                            detect = 1
                            flag = 0
                            skip = 0
                            for k in range(j + 1, length, 1):
                                if (sequence[i][k] == char2
                                        and skip == 22):  # match with char2
                                    flag = 1
                                    break
                                if (sequence[i][k] == char2 and skip != 22):
                                    skip = skip + 1
                                total = total + 1

                            if (flag == 1):
                                count22[i][letter] = total + 1
                                letter = letter + 1
                                finish = 1
                            else:
                                count22[i][letter] = 0
                                letter = letter + 1
                                finish = 1

                        if (finish == 1):
                            finish = 0
                            break
                    if (detect == 0):
                        count22[i][letter] = 0
                        letter = letter + 1

    # 24th nearest neigbour
    count23 = np.zeros((n, 400))
    for i in range(0, n, 1):
        length = len(sequence[i])
        letter = 0
        finish = 0
        skip = 0
        for char1 in string.ascii_uppercase:
            for char2 in string.ascii_uppercase:
                if (char1 != 'B' and char2 != 'B' and char1 != 'J'
                        and char2 != 'J' and char1 != 'O' and char2 != 'O'
                        and char1 != 'U' and char2 != 'U' and char1 != 'X'
                        and char2 != 'X' and char1 != 'Z' and char2 != 'Z'):
                    detect = 0
                    skip = 0
                    for j in range(0, length, 1):
                        if (sequence[i][j] == char1):  # detect char1
                            total = 0
                            detect = 1
                            flag = 0
                            skip = 0
                            for k in range(j + 1, length, 1):
                                if (sequence[i][k] == char2
                                        and skip == 23):  # match with char2
                                    flag = 1
                                    break
                                if (sequence[i][k] == char2 and skip != 23):
                                    skip = skip + 1
                                total = total + 1

                            if (flag == 1):
                                count23[i][letter] = total + 1
                                letter = letter + 1
                                finish = 1
                            else:
                                count23[i][letter] = 0
                                letter = letter + 1
                                finish = 1

                        if (finish == 1):
                            finish = 0
                            break
                    if (detect == 0):
                        count23[i][letter] = 0
                        letter = letter + 1

    # 25th nearest neigbour
    count24 = np.zeros((n, 400))
    for i in range(0, n, 1):
        length = len(sequence[i])
        letter = 0
        finish = 0
        skip = 0
        for char1 in string.ascii_uppercase:
            for char2 in string.ascii_uppercase:
                if (char1 != 'B' and char2 != 'B' and char1 != 'J'
                        and char2 != 'J' and char1 != 'O' and char2 != 'O'
                        and char1 != 'U' and char2 != 'U' and char1 != 'X'
                        and char2 != 'X' and char1 != 'Z' and char2 != 'Z'):
                    detect = 0
                    skip = 0
                    for j in range(0, length, 1):
                        if (sequence[i][j] == char1):  # detect char1
                            total = 0
                            detect = 1
                            flag = 0
                            skip = 0
                            for k in range(j + 1, length, 1):
                                if (sequence[i][k] == char2
                                        and skip == 24):  # match with char2
                                    flag = 1
                                    break
                                if (sequence[i][k] == char2 and skip != 24):
                                    skip = skip + 1
                                total = total + 1

                            if (flag == 1):
                                count24[i][letter] = total + 1
                                letter = letter + 1
                                finish = 1
                            else:
                                count24[i][letter] = 0
                                letter = letter + 1
                                finish = 1

                        if (finish == 1):
                            finish = 0
                            break
                    if (detect == 0):
                        count24[i][letter] = 0
                        letter = letter + 1

    # 26th nearest neigbour
    count25 = np.zeros((n, 400))
    for i in range(0, n, 1):
        length = len(sequence[i])
        letter = 0
        finish = 0
        skip = 0
        for char1 in string.ascii_uppercase:
            for char2 in string.ascii_uppercase:
                if (char1 != 'B' and char2 != 'B' and char1 != 'J'
                        and char2 != 'J' and char1 != 'O' and char2 != 'O'
                        and char1 != 'U' and char2 != 'U' and char1 != 'X'
                        and char2 != 'X' and char1 != 'Z' and char2 != 'Z'):
                    detect = 0
                    skip = 0
                    for j in range(0, length, 1):
                        if (sequence[i][j] == char1):  # detect char1
                            total = 0
                            detect = 1
                            flag = 0
                            skip = 0
                            for k in range(j + 1, length, 1):
                                if (sequence[i][k] == char2
                                        and skip == 25):  # match with char2
                                    flag = 1
                                    break
                                if (sequence[i][k] == char2 and skip != 25):
                                    skip = skip + 1
                                total = total + 1

                            if (flag == 1):
                                count25[i][letter] = total + 1
                                letter = letter + 1
                                finish = 1
                            else:
                                count25[i][letter] = 0
                                letter = letter + 1
                                finish = 1

                        if (finish == 1):
                            finish = 0
                            break
                    if (detect == 0):
                        count25[i][letter] = 0
                        letter = letter + 1

    # 27th nearest neigbour
    count26 = np.zeros((n, 400))
    for i in range(0, n, 1):
        length = len(sequence[i])
        letter = 0
        finish = 0
        skip = 0
        for char1 in string.ascii_uppercase:
            for char2 in string.ascii_uppercase:
                if (char1 != 'B' and char2 != 'B' and char1 != 'J'
                        and char2 != 'J' and char1 != 'O' and char2 != 'O'
                        and char1 != 'U' and char2 != 'U' and char1 != 'X'
                        and char2 != 'X' and char1 != 'Z' and char2 != 'Z'):
                    detect = 0
                    skip = 0
                    for j in range(0, length, 1):
                        if (sequence[i][j] == char1):  # detect char1
                            total = 0
                            detect = 1
                            flag = 0
                            skip = 0
                            for k in range(j + 1, length, 1):
                                if (sequence[i][k] == char2
                                        and skip == 26):  # match with char2
                                    flag = 1
                                    break
                                if (sequence[i][k] == char2 and skip != 26):
                                    skip = skip + 1
                                total = total + 1

                            if (flag == 1):
                                count26[i][letter] = total + 1
                                letter = letter + 1
                                finish = 1
                            else:
                                count26[i][letter] = 0
                                letter = letter + 1
                                finish = 1

                        if (finish == 1):
                            finish = 0
                            break
                    if (detect == 0):
                        count26[i][letter] = 0
                        letter = letter + 1

    # 28th nearest neigbour
    count27 = np.zeros((n, 400))
    for i in range(0, n, 1):
        length = len(sequence[i])
        letter = 0
        finish = 0
        skip = 0
        for char1 in string.ascii_uppercase:
            for char2 in string.ascii_uppercase:
                if (char1 != 'B' and char2 != 'B' and char1 != 'J'
                        and char2 != 'J' and char1 != 'O' and char2 != 'O'
                        and char1 != 'U' and char2 != 'U' and char1 != 'X'
                        and char2 != 'X' and char1 != 'Z' and char2 != 'Z'):
                    detect = 0
                    skip = 0
                    for j in range(0, length, 1):
                        if (sequence[i][j] == char1):  # detect char1
                            total = 0
                            detect = 1
                            flag = 0
                            skip = 0
                            for k in range(j + 1, length, 1):
                                if (sequence[i][k] == char2
                                        and skip == 27):  # match with char2
                                    flag = 1
                                    break
                                if (sequence[i][k] == char2 and skip != 27):
                                    skip = skip + 1
                                total = total + 1

                            if (flag == 1):
                                count27[i][letter] = total + 1
                                letter = letter + 1
                                finish = 1
                            else:
                                count27[i][letter] = 0
                                letter = letter + 1
                                finish = 1

                        if (finish == 1):
                            finish = 0
                            break
                    if (detect == 0):
                        count27[i][letter] = 0
                        letter = letter + 1

    # 29th nearest neigbour
    count28 = np.zeros((n, 400))
    for i in range(0, n, 1):
        length = len(sequence[i])
        letter = 0
        finish = 0
        skip = 0
        for char1 in string.ascii_uppercase:
            for char2 in string.ascii_uppercase:
                if (char1 != 'B' and char2 != 'B' and char1 != 'J'
                        and char2 != 'J' and char1 != 'O' and char2 != 'O'
                        and char1 != 'U' and char2 != 'U' and char1 != 'X'
                        and char2 != 'X' and char1 != 'Z' and char2 != 'Z'):
                    detect = 0
                    skip = 0
                    for j in range(0, length, 1):
                        if (sequence[i][j] == char1):  # detect char1
                            total = 0
                            detect = 1
                            flag = 0
                            skip = 0
                            for k in range(j + 1, length, 1):
                                if (sequence[i][k] == char2
                                        and skip == 28):  # match with char2
                                    flag = 1
                                    break
                                if (sequence[i][k] == char2 and skip != 28):
                                    skip = skip + 1
                                total = total + 1

                            if (flag == 1):
                                count28[i][letter] = total + 1
                                letter = letter + 1
                                finish = 1
                            else:
                                count28[i][letter] = 0
                                letter = letter + 1
                                finish = 1

                        if (finish == 1):
                            finish = 0
                            break
                    if (detect == 0):
                        count28[i][letter] = 0
                        letter = letter + 1

    # 30th nearest neigbour
    count29 = np.zeros((n, 400))
    for i in range(0, n, 1):
        length = len(sequence[i])
        letter = 0
        finish = 0
        skip = 0
        for char1 in string.ascii_uppercase:
            for char2 in string.ascii_uppercase:
                if (char1 != 'B' and char2 != 'B' and char1 != 'J'
                        and char2 != 'J' and char1 != 'O' and char2 != 'O'
                        and char1 != 'U' and char2 != 'U' and char1 != 'X'
                        and char2 != 'X' and char1 != 'Z' and char2 != 'Z'):
                    detect = 0
                    skip = 0
                    for j in range(0, length, 1):
                        if (sequence[i][j] == char1):  # detect char1
                            total = 0
                            detect = 1
                            flag = 0
                            skip = 0
                            for k in range(j + 1, length, 1):
                                if (sequence[i][k] == char2
                                        and skip == 29):  # match with char2
                                    flag = 1
                                    break
                                if (sequence[i][k] == char2 and skip != 29):
                                    skip = skip + 1
                                total = total + 1

                            if (flag == 1):
                                count29[i][letter] = total + 1
                                letter = letter + 1
                                finish = 1
                            else:
                                count29[i][letter] = 0
                                letter = letter + 1
                                finish = 1

                        if (finish == 1):
                            finish = 0
                            break
                    if (detect == 0):
                        count29[i][letter] = 0
                        letter = letter + 1

    a = np.array([])
    for i in range(1, 31, 1):
        for char1 in string.ascii_uppercase:
            for char2 in string.ascii_uppercase:
                if (char1 != 'B' and char2 != 'B' and char1 != 'J'
                        and char2 != 'J' and char1 != 'O' and char2 != 'O'
                        and char1 != 'U' and char2 != 'U' and char1 != 'X'
                        and char2 != 'X' and char1 != 'Z' and char2 != 'Z'):
                    a = np.append(a, "{0}d{1}{2}".format(char1, i, char2))

    with open('Normalized_30_nearest_neigbour.csv', 'w') as csvfile:
        fieldnames = a
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for i in range(0, n, 1):
            div = len(sequence)

            for j in range(0, 400, 1):
                csvfile.write("{0},".format((count[i][j]) / div))
            for j in range(0, 400, 1):
                csvfile.write("{0},".format((count1[i][j]) / div))
            for j in range(0, 400, 1):
                csvfile.write("{0},".format((count2[i][j]) / div))
            for j in range(0, 400, 1):
                csvfile.write("{0},".format((count3[i][j]) / div))
            for j in range(0, 400, 1):
                csvfile.write("{0},".format((count4[i][j]) / div))
            for j in range(0, 400, 1):
                csvfile.write("{0},".format((count5[i][j]) / div))
            for j in range(0, 400, 1):
                csvfile.write("{0},".format((count6[i][j]) / div))
            for j in range(0, 400, 1):
                csvfile.write("{0},".format((count7[i][j]) / div))
            for j in range(0, 400, 1):
                csvfile.write("{0},".format((count8[i][j]) / div))
            for j in range(0, 400, 1):
                csvfile.write("{0},".format((count9[i][j]) / div))
            for j in range(0, 400, 1):
                csvfile.write("{0},".format((count10[i][j]) / div))
            for j in range(0, 400, 1):
                csvfile.write("{0},".format((count11[i][j]) / div))
            for j in range(0, 400, 1):
                csvfile.write("{0},".format((count12[i][j]) / div))
            for j in range(0, 400, 1):
                csvfile.write("{0},".format((count13[i][j]) / div))
            for j in range(0, 400, 1):
                csvfile.write("{0},".format((count14[i][j]) / div))
            for j in range(0, 400, 1):
                csvfile.write("{0},".format((count15[i][j]) / div))
            for j in range(0, 400, 1):
                csvfile.write("{0},".format((count16[i][j]) / div))
            for j in range(0, 400, 1):
                csvfile.write("{0},".format((count17[i][j]) / div))
            for j in range(0, 400, 1):
                csvfile.write("{0},".format((count18[i][j]) / div))
            for j in range(0, 400, 1):
                csvfile.write("{0},".format((count19[i][j]) / div))
            for j in range(0, 400, 1):
                csvfile.write("{0},".format((count20[i][j]) / div))
            for j in range(0, 400, 1):
                csvfile.write("{0},".format((count21[i][j]) / div))
            for j in range(0, 400, 1):
                csvfile.write("{0},".format((count22[i][j]) / div))
            for j in range(0, 400, 1):
                csvfile.write("{0},".format((count23[i][j]) / div))
            for j in range(0, 400, 1):
                csvfile.write("{0},".format((count24[i][j]) / div))
            for j in range(0, 400, 1):
                csvfile.write("{0},".format((count25[i][j]) / div))
            for j in range(0, 400, 1):
                csvfile.write("{0},".format((count26[i][j]) / div))
            for j in range(0, 400, 1):
                csvfile.write("{0},".format((count27[i][j]) / div))
            for j in range(0, 400, 1):
                csvfile.write("{0},".format((count28[i][j]) / div))
            for j in range(0, 400, 1):
                csvfile.write("{0},".format((count29[i][j]) / div))

            csvfile.write("0\n")

    df = pd.read_csv('Normalized_30_nearest_neigbour.csv')

    values = df.values.tolist()[0]

    return (['NNB_' + x for x in a], values)


def HSE(ss_seq, hse_up, hse_down):
    pi = math.pi  # 3.141592653589793
    hse_u = hse_d = 0.0
    for i in range(0, len(ss_seq)):
        hse_u = hse_u + hse_up[i]
        hse_d = hse_d + hse_down[i]

    values = []
    values.append(hse_u / len(ss_seq))
    values.append(hse_d / len(ss_seq))

    features = ['HSE_alpha_up', 'HSE_alpha_down']

    return (features, values)


def generateGroupPairs(groupKey):
    gPair = {}
    for key1 in groupKey:
        for key2 in groupKey:
            gPair[key1 + '.' + key2] = 0
    return gPair


def nGAAGP(sequence, gap=5, **kw):
    if gap < 0:
        print(('Error: the gap should be equal or greater than zero' + '\n\n'))
        return 0

    if len(sequence) < gap + 2:
        print((
            'Error: all the sequence length should be greater than the (gap value) + 2 = '
            + str(gap + 2) + '\n\n'))
        return 0

    group = {
        'alphaticr': 'GAVLMI',
        'aromatic': 'FYW',
        'postivecharger': 'KRH',
        'negativecharger': 'DE',
        'uncharger': 'STCPNQ'
    }

    AA = 'ARNDCQEGHILKMFPSTWYV'

    groupKey = list(group.keys())

    index = {}
    for key in groupKey:
        for aa in group[key]:
            index[aa] = key

    gPairIndex = []
    for key1 in groupKey:
        for key2 in groupKey:
            gPairIndex.append(key1 + '.' + key2)

    features = []
    for g in range(gap + 1):
        for p in gPairIndex:
            features.append("CKSAAGP_" + p + '.gap' + str(g))

    values = []
    for g in range(gap + 1):
        gPair = generateGroupPairs(groupKey)
        sum = 0
        for p1 in range(len(sequence)):
            p2 = p1 + g + 1
            if p2 < len(
                    sequence) and sequence[p1] in AA and sequence[p2] in AA:
                gPair[index[sequence[p1]] + '.' +
                      index[sequence[p2]]] = gPair[index[sequence[p1]] + '.' +
                                                   index[sequence[p2]]] + 1
                sum = sum + 1

        if sum == 0:
            for gp in gPairIndex:
                values.append(0)
        else:
            for gp in gPairIndex:
                values.append(gPair[gp] / sum)

    return (features, values)

def ProcessFeatures(featureValues, featureType, aggregatedFeatures, aggregatedValues):
#     print(featureType)
#     print(len(featureValues[0]), len(featureValues[1]))
    return (aggregatedFeatures + featureValues[0], aggregatedValues + featureValues[1])


def format_pssm(file_name):
    # dict with
    # Sequence Amino acids, pssm, weighted Obseved %,
    # info@position, relative_weights, and a dict_list final metrics

    # Read file
    file = open(file_name, 'r')
    data = file.read()
    data = data.split('\n')  # Discard Empty lines

    # Discard leading and trailing white space
    data_trimmed = []
    for idx in range(len(data)):
        line = data[idx].strip()
        if len(line) != 0:
            data_trimmed.append(line)

    # Amino Acids
    amino_acids = data_trimmed[1].split()

    # Tabular Data
    table = data_trimmed[2:-5]

    sequence = []
    pssm = []
    weighted_observed_percentages = []
    info_position = []
    relative_weights = []

    for row in table:
        row_split = row.split()

        position = row_split[0]
        assert (int(position) == (len(sequence) + 1))

        sequence.append(row_split[1])
        pssm.append(np.asarray(row_split[2:22], dtype=int))
        weighted_observed_percentages.append(
            np.asarray(row_split[22:-2], dtype=int))
        info_position.append(float(row_split[-2]))
        relative_weights.append(float(row_split[-1]))

    # Process final metrics
    final_metrics = data_trimmed[-5:]

    line_split = final_metrics[0].split()

    key0 = 'Metric'
    key1 = line_split[0]
    key2 = line_split[1]

    assert (key1 == 'K')
    assert (key2 == 'Lambda')

    metrics = []
    for line in final_metrics[1:]:
        line_split = line.split()

        metrics.append({
            key0: '_'.join(line_split[0:2]),
            key1: line_split[2],
            key2: line_split[3]
        })

    return {
        'seq': str(''.join(sequence)),
        'AA': amino_acids,
        'pssm': pssm,
        'w_obs_perc': weighted_observed_percentages,
        'info_pos': info_position,
        'rel_w': relative_weights,
        'f_metrics': metrics,
    }


def normalize_pssm(pssm):
    n_pssm = []
    for row in pssm:
        mean_row = np.mean(row)
        numerator = row - mean_row
        denom = np.sqrt(np.mean((row - mean_row)**2))
        if denom == 0:
            n_pssm.append(numerator)
        else:
            n_pssm.append(numerator / denom)
        assert (len(row) == len(n_pssm[-1]))
    return n_pssm


def frag_pssm(n_pssm, n):
    L = len(n_pssm)
    gap = int(L / n)

    idx_dist = np.arange(0, L, gap)
    if len(idx_dist) != n:
        idx_dist = idx_dist[:-1]

    f_pssm = []

    for i in range(len(idx_dist) - 1):
        f_pssm.append(n_pssm[idx_dist[i]:idx_dist[i + 1]])

    f_pssm.append(n_pssm[idx_dist[-1]:])

    assert (len(f_pssm) == n)

    for i in range(0, n - 1):
        assert (len(f_pssm[i]) == gap)

    return f_pssm


def featurize_pssm(f_pssm, lambda_):
    part1 = {}
    part2 = {}
    for k in range(len(f_pssm)):
        part1['F' + str(k + 1)] = np.sum(f_pssm[k], axis=0) / len(f_pssm[k])

        for epsilon in range(1, lambda_ + 1):
            part2['Phi' + str(k) + '_' + str(epsilon)] = (np.sum(
                (np.array(f_pssm[k][:-epsilon]) -
                 np.array(f_pssm[k][epsilon:]))**2,
                axis=0) / (len(f_pssm[k]) - epsilon))

    return part1, part2


def LPsePSSM(pssmFile):
    finalValues = {}
    for n in range(1, 6, 1):
        for lambda_ in range(8, 9, 1):
            npssm = normalize_pssm(format_pssm(pssmFile)['pssm'])
            fpssm = frag_pssm(npssm, n)
            part1, part2 = featurize_pssm(fpssm, lambda_)

            try:
                for feature, values in list(part1.items()):
                    for j in range(len(values)):
                        finalValues[str(n) + "_" + feature + "_" + str(j)] = values[j]

                for feature, values in list(part2.items()):
                    for j in range(len(values)):
                        finalValues[str(n) + "_" + feature + "_" + str(j)] = values[j]
            except KeyError as e:
                print((pssmFile, e))
                
    return ([f for f in finalValues.keys()], [v for v in finalValues.values()])

def Featurize(sequence, pssmFile, spiderFile):
    plen, npssm = ProcessPSSM(pssmFile)

    ss_seq, ASA, phi, psi, theta, tau, hse_up, hse_down, coil, betaSheet, alphaHelix = ProcessSPD(
        spiderFile, plen)
    aggregatedFeatures = aggregatedValues = []
    aggregatedFeatures, aggregatedValues = ProcessFeatures(
        BigramPercentileSeparation(sequence), 'BPS', aggregatedFeatures,
        aggregatedValues)
    aggregatedFeatures, aggregatedValues = ProcessFeatures(
        MonogramPercentileSeparation(sequence), 'MPS', aggregatedFeatures,
        aggregatedValues)
    aggregatedFeatures, aggregatedValues = ProcessFeatures(
        DDE(sequence), 'DDE', aggregatedFeatures, aggregatedValues)
    aggregatedFeatures, aggregatedValues = ProcessFeatures(
        QSOrder(sequence), 'QSOrder', aggregatedFeatures, aggregatedValues)
    aggregatedFeatures, aggregatedValues = ProcessFeatures(
        CTDC(sequence), 'CTDC', aggregatedFeatures, aggregatedValues)
    aggregatedFeatures, aggregatedValues = ProcessFeatures(
        CTriad(sequence), 'CTriad', aggregatedFeatures, aggregatedValues)
    aggregatedFeatures, aggregatedValues = ProcessFeatures(
        PAAC(sequence), 'PAAC', aggregatedFeatures, aggregatedValues)
    aggregatedFeatures, aggregatedValues = ProcessFeatures(
        APAAC(sequence), 'APAAC', aggregatedFeatures, aggregatedValues)
    aggregatedFeatures, aggregatedValues = ProcessFeatures(
        CTDD(sequence), 'CTDD', aggregatedFeatures, aggregatedValues)
    aggregatedFeatures, aggregatedValues = ProcessFeatures(
        Geary(sequence), 'Geary', aggregatedFeatures, aggregatedValues)
    aggregatedFeatures, aggregatedValues = ProcessFeatures(
        KSCTriad(sequence), 'KSCTriad', aggregatedFeatures, aggregatedValues)
    aggregatedFeatures, aggregatedValues = ProcessFeatures(
        GTPC(sequence), 'GTPC', aggregatedFeatures, aggregatedValues)
    aggregatedFeatures, aggregatedValues = ProcessFeatures(
        GDPC(sequence), 'GDPC', aggregatedFeatures, aggregatedValues)
    aggregatedFeatures, aggregatedValues = ProcessFeatures(
        GAAC(sequence), 'GAAC', aggregatedFeatures, aggregatedValues)
    aggregatedFeatures, aggregatedValues = ProcessFeatures(
        Moran(sequence), 'Moran', aggregatedFeatures, aggregatedValues)
    aggregatedFeatures, aggregatedValues = ProcessFeatures(
        SOCNumber(sequence), 'SOCNumber', aggregatedFeatures, aggregatedValues)
    aggregatedFeatures, aggregatedValues = ProcessFeatures(
        Dubchak(sequence), 'Dubchak', aggregatedFeatures, aggregatedValues)
    aggregatedFeatures, aggregatedValues = ProcessFeatures(
        PSSMBigram(plen, npssm), 'PSSMBigram', aggregatedFeatures,
        aggregatedValues)
    aggregatedFeatures, aggregatedValues = ProcessFeatures(
        PSSMAutoCovariance(plen, npssm), 'PSSMAutoCovariance',
        aggregatedFeatures, aggregatedValues)
    aggregatedFeatures, aggregatedValues = ProcessFeatures(
        PSSMSegmentDistribution(npssm), 'PSSMSegmentDistribution',
        aggregatedFeatures, aggregatedValues)
    aggregatedFeatures, aggregatedValues = ProcessFeatures(
        OneLeadBigramPSSM(plen, npssm), 'OneLeadBigramPSSM',
        aggregatedFeatures, aggregatedValues)
    aggregatedFeatures, aggregatedValues = ProcessFeatures(
        SecondaryStructureComposition(ss_seq), 'SecondaryStructureComposition',
        aggregatedFeatures, aggregatedValues)
    aggregatedFeatures, aggregatedValues = ProcessFeatures(
        SecondaryStructureOccurance(ss_seq), 'SecondaryStructureOccurance',
        aggregatedFeatures, aggregatedValues)
    aggregatedFeatures, aggregatedValues = ProcessFeatures(
        ASA_AngleOccurance_ProbCHE(ss_seq, phi, psi, theta, tau, coil,
                                   betaSheet, alphaHelix, ASA),
        'ASA_AngleOccurance_ProbCHE', aggregatedFeatures, aggregatedValues)
    aggregatedFeatures, aggregatedValues = ProcessFeatures(
        TorsionalAnglesBigram(ss_seq, phi, psi, theta, tau),
        'TorsionalAnglesBigram', aggregatedFeatures, aggregatedValues)
    aggregatedFeatures, aggregatedValues = ProcessFeatures(
        TorsionalAnglesAutoCovariance(ss_seq, phi, psi, theta, tau),
        'TorsionalAnglesAutoCovariance', aggregatedFeatures, aggregatedValues)
    aggregatedFeatures, aggregatedValues = ProcessFeatures(
        StructuralProbabilitiesBigram(ss_seq, coil, betaSheet, alphaHelix),
        'StructuralProbabilitiesBigram', aggregatedFeatures, aggregatedValues)
    aggregatedFeatures, aggregatedValues = ProcessFeatures(
        StructuralProbabilitesAutoCovariance(ss_seq, coil, betaSheet,
                                             alphaHelix),
        'StructuralProbabilitesAutoCovariance', aggregatedFeatures,
        aggregatedValues)
    aggregatedFeatures, aggregatedValues = ProcessFeatures(
        HSE(ss_seq, hse_up, hse_down), 'HSE', aggregatedFeatures,
        aggregatedValues)
    aggregatedFeatures, aggregatedValues = ProcessFeatures(
        nGAAGP(sequence), 'nGAAGP', aggregatedFeatures, aggregatedValues)
    aggregatedFeatures, aggregatedValues = ProcessFeatures(
        NNB(sequence), 'NNB', aggregatedFeatures, aggregatedValues)
    aggregatedFeatures, aggregatedValues = ProcessFeatures(
        LPsePSSM(pssmFile), 'LPsePSSM', aggregatedFeatures, aggregatedValues)
    return (aggregatedFeatures, aggregatedValues)


# # Featurize

# In[2]:


sequence = "GAAAAMSICPHIQQVFQNEKSKDGVLKTCNAARYILNHSVPKEKFLNTMKCGTCHEINSGATFMCLQCGFCGCWNHSHFLSHSKQIGHIFGINSNNGLLFCFKCEDYIGNIDLINDAILAKYWDDVCTKTMVPSMERRDGLSGLINMGNTCFMSSILQCLIHNPYFIRHSMSQIHSNNCKVRSPDKCFSCALDKIVHELYGALNTKQASSSSTSTNRQTGFIYLLTCAWKINQNLAGYSQQDAHEFWQFIINQIHQSYVLDLPNAKEVSRANNKQCECIVHTVFEGSLESSIVCPGCQNNSKTTIDPFLDLSLDIKDKKKLYECLDSFHKKEQLKDFNYHCGECNSTQDAIKQLGIHKLPSVLVLQLKRFEHLLNGSNRKLDDFIEFPTYLNMKNYCSTKEKDKHSENGKVPDIIYELIGIVSHKGTVNEGHYIAFCKISGGQWFKFNDSMVSSISQEEVLKEQAYLLFYTIRQVN"

pssmFile = "../example.pssm"

spiderFile = "../example.spd33"

selected_features = np.load('../Features/rf452.npz')['feature']

aggregatedFeatures, aggregatedValues = Featurize(sequence, pssmFile, spiderFile)

aggregatedValues = [float(x) for x in aggregatedValues]

df = pd.DataFrame(data=[aggregatedValues], columns=aggregatedFeatures)

test_X = df[selected_features].values


# # Predict

# In[3]:


import _pickle
import numpy as np
import pandas as pd
from mlxtend.evaluate import bias_variance_decomp
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import (AdaBoostClassifier, BaggingClassifier,
                              ExtraTreesClassifier, RandomForestClassifier,
                              StackingClassifier, VotingClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score,
                             confusion_matrix, make_scorer, matthews_corrcoef,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import (GridSearchCV, LeaveOneOut,
                                     RandomizedSearchCV, StratifiedKFold,
                                     cross_validate)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier

data = np.load('../Features/rf452.npz')

X = data['X']
y = data['y']

skf = StratifiedKFold(n_splits=10, random_state=0, shuffle=True)

# make a prediction with a stacking ensemble
# define the base models
level0 = list()

level0.append(('VC',
               VotingClassifier(
                   [
                       ('DT', DecisionTreeClassifier(random_state=0)),
                       ('ABC', AdaBoostClassifier(random_state=0)),
                       ('LDA', LinearDiscriminantAnalysis()),
                   ],
                   voting='soft',
                   n_jobs=-1,
               )))

level0.append(('svm(rbf, tuned)',
               SVC(kernel='rbf',
                   C=5.44,
                   gamma=.00237,
                   random_state=0,
                   probability=True)))

level1 = LogisticRegression(solver='liblinear', random_state=0)

clf = StackingClassifier(estimators=level0,
                         final_estimator=level1,
                         stack_method='predict_proba',
                         cv=skf)

clf.fit(X, y)

test_pred = clf.predict(test_X)[0]

if test_pred == 0:
    print("The provided sequence is Non-Dna-Binding")
else:
    print("The provided sequence is Dna-Binding")


# In[ ]:




