import numpy as np
chars = [i for i in range(48, 48+10)] + [i for i in range(65, 65+26)]
syms = " |-/\\^v<>"
d = {}
for i, c in enumerate(syms):
    d[c] = [0]*9
    d[c][i] = 1
semr = {}
semr['0'] = '/-\\| |\\-/'
semr['1'] = '-|  |  | '
semr['2'] = '--\\ / |--'
semr['3'] = ' -| <  -|'
semr['4'] = ' / / | |-'
semr['5'] = '|- |-\\ -/'
semr['6'] = '/- |-\\\\-/'
semr['7'] = ' -/ /  | '
semr['8'] = '/-\\|-|\\-/'
semr['9'] = '/-\\\\-| -/'
semr['A'] = ' ^ /-\\| |'
semr['B'] = '|-\\|-<|-/'
semr['C'] = ' - <   - '
semr['D'] = '|- | >|- '
semr['E'] = '|- |- |- '
semr['F'] = '|- |- |  '
semr['G'] = '/- | -\\-/'
semr['H'] = '| ||-|| |'
semr['I'] = ' |  |  | '
semr['J'] = ' |  | |/ '
semr['K'] = '| /|< | \\'
semr['L'] = ' |  |  |-'
semr['M'] = '| ||v|| |'
semr['N'] = '| ||\\|| |'
semr['O'] = '/-\\| |\\-/'
semr['P'] = '|-\\|-/|  '
semr['Q'] = '/-\\| |\\-\\'
semr['R'] = '|-\\|-/| \\'
semr['S'] = '/-  \\  -/'
semr['T'] = ' -  |  | '
semr['U'] = '| || |\\-/'
semr['V'] = '| |\\ / - '
semr['W'] = '| ||^|\\ /'
semr['X'] = '\\ / - / \\'
semr['Y'] = '\\ / -  | '
semr['Z'] = '--/ / /--'
sem_data = []
for c in chars:
    vecr = []
    for i in semr[chr(c)]:
        vecr += d[i]
    sem_data.append(vecr)