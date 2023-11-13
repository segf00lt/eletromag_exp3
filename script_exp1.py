# README
# linhas de tendência aparecem em laranja
# os pontos são desenhados com barras de erro
# as linhas esperadas aparecem em verde

import csv
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sys import platform

G = 980.665
M = (50, 0.1)

poly0 = lambda x, a: np.ones(x.shape)*a
poly1 = lambda x, a, b: a*x + b
poly2 = lambda x, a, b, c: a*x**2 + b*x + c
poly3 = lambda x, a, b, c, d: a*x**3 + b*x**2 + c*x + d

with open('data_exp1.csv', newline='') as f:
    data_exp1 = np.array(list(csv.reader(f))[1:], dtype=np.float32)

#with open(r'C:\Users\cisco\Downloads\data_exp1.csv', newline='') as f:
#    data_exp2 = np.array(list(csv.reader(f))[1:], dtype=np.float32)

def numeric_derivative(rise, run, j, errors):
    assert len(rise) == len(run)
    n = len(rise)
    ds = []
    es = []
    i = 0

    while i < j and i < n-1:
        run_dist = run[i+1] - run[i]
        ds.append( (rise[i+1] - rise[i]) / run_dist )
        es.append( (errors[i+1] + errors[i]) / run_dist )
        i += 1
    
    while i < n - j:
        run_dist = run[i+j] - run[i-j]
        ds.append( (rise[i+j] - rise[i-j]) / run_dist )
        es.append( (errors[i+j] + errors[i-j]) / run_dist )
        i += 1

    while i < n:
        run_dist = run[i] - run[i-1]
        ds.append( (rise[i] - rise[i-1]) / run_dist )
        es.append( (errors[i] + errors[i-1]) / run_dist )
        i += 1

    assert len(ds) == n

    return (np.array(ds), np.array(es))

def show_with_trend(X, Y, Y_err, trend_func, fig, pos, title, expect_func=None):
    ax = fig.add_subplot(pos)
    ax.set_facecolor('#000000')
    ax.errorbar(X, Y, yerr=Y_err, fmt='.')
    ax.set_title(title)
    fitparams, pcov = curve_fit(trend_func, X, Y)

    print()
    print("PARÂMETROS PARA TESTE")
    print(fitparams)
    print()

    paramErrors = np.sqrt(np.diag(pcov))

    print()
    print("ERROS DOS PARÂMETROS")
    print(paramErrors)
    print()

    X_model = np.linspace(min(X), max(X), 1000)
    Y_model = trend_func(X_model, *fitparams)
    ax.plot(X_model, Y_model)
    if expect_func is not None:
        ax.plot(X_model, expect_func(X_model))
    return fitparams, paramErrors

def zscore(m1, m2, d1, d2):
    return abs(m1 - m2) / ((d1**2 + d2**2)**0.5)

def zscore_compatible(z):
    if z <= 1:
        print (z)
        return 'is compatible'
    elif 1 < z <= 3:
        print (z)
        return 'maybe compatible'
    else:
        print (z)
        return 'incompatible'

t_exp1 = data_exp1[:,2]
s_exp1 = data_exp1[:,1]
s_exp1 += 1

param = {
    0: "GRAVITY",
    1: "INITIAL VELOCITY",
    2: "INITIAL_ POSITION" 
}
# deslocamento
print()
print("DESLOCAMENTO")
px = 1/plt.rcParams['figure.dpi']  # pixel in inches
FIGURE_SIZE = (1500*px,1000*px)
fig0 = plt.figure(figsize=FIGURE_SIZE)
fig0.set_facecolor('#cccccc')
fitparams, paramErrors = show_with_trend(t_exp1, s_exp1, 0.1, poly2, fig0, 111, f"Deslocamento X Tempo", lambda T: G*0.5*(T**2))

fitparams[0] *= 2
paramErrors[0] *= 2
reference = [G, 0, -1.0]
referenceErrors = [0.001, 0, 0.05]
for i in range(3):
    z = zscore(fitparams[i], reference[i], paramErrors[i], referenceErrors[i])
    print(f"distance {param[i]} measurement {zscore_compatible(z)}")

fig1 = plt.figure(figsize=FIGURE_SIZE)
fig1.set_facecolor('#cccccc')

accel = []
accel_err = []
vel = []
vel_err = []

# velocidade
print()
print("VELOCIDADE")
for p in [221,222,223,224]:
    j = p%220
    v_exp1, err_v_exp1 = numeric_derivative(data_exp1[:,1], t_exp1, j, np.ones(t_exp1.shape)*0.05)
    if(j == 1):
        for i in [1, 2, 3, 4]:
            a, a_err = numeric_derivative(v_exp1, t_exp1, i, err_v_exp1)
            accel.append(a)
            accel_err.append(a_err)
    fitparams, paramErrors = show_with_trend(t_exp1, v_exp1, err_v_exp1, poly1, fig1, p, f"Velocidade X Tempo (derivada com j = {j})", lambda T: T*G)
    for i in range(2):
        z = zscore(fitparams[i], reference[i], paramErrors[i], referenceErrors[i])
        print(f"velocity with j = {j} {param[i]} measurement {zscore_compatible(z)}")
    vel.append(v_exp1)
    vel_err.append(err_v_exp1)

fig2 = plt.figure(figsize=FIGURE_SIZE)
fig2.set_facecolor('#cccccc')

# aceleracao
print()
print("ACELERACAO")
p = 221
for i,a in enumerate(accel):
    j = p%220
    fitparams, paramErrors = show_with_trend(t_exp1, a, accel_err[i], poly0, fig2, p, f"Aceleração X Tempo (derivada com j = {j})", lambda T: np.ones(T.shape)*G)
    
    z = zscore(fitparams[0], reference[0], paramErrors[0], referenceErrors[0])
    print(f"acceleration with j = {j} measurement {zscore_compatible(z)}")

    p += 1


# energia cinetica (mv^2)/2
print()
print("CINETICA")
fig3 = plt.figure(figsize=FIGURE_SIZE)
fig3.set_facecolor('#cccccc')

cineticas = []
cineticas_err = []

for i,v in enumerate(vel):
    cin = 0.5*M[0]*0.001*(v/100)**2
    cin_err = (abs((v/100)**2)/2)*M[1]*0.001 + abs(M[0]*0.001*(v/100))*(vel_err[i]/100)
    fitparams, paramErrors = show_with_trend(t_exp1, cin, cin_err, poly2, fig3, 221+i, f"Energia Cinética X Tempo (velocidade com j = {i+1})", lambda T: 0.5*M[0]*0.001*((T*G) /100)**2)
    reference = [2.404259605, 0, 0]
    referenceErrors = [0.069378833, 0, 0]
    for i in range(3):
        z = zscore(fitparams[i], reference[i], paramErrors[i], referenceErrors[i])
        print(f"kinectic with j = {j} {param[i]} measurement {zscore_compatible(z)}")
    cineticas.append(cin)
    cineticas_err.append(cin_err)


# energia potenical mah
print()
print("POTENCIAL")
fig4 = plt.figure(figsize=FIGURE_SIZE)
fig4.set_facecolor('#cccccc')

potenciais = []
potenciais_err = []

for i,a in enumerate(accel):
    rev_s_exp1 = s_exp1[::-1]/100
    pot = M[0]*0.001 * a*0.01 * rev_s_exp1
    pot_err = abs(a*0.01 * rev_s_exp1)*M[1]*0.001 + abs(M[0]*0.001*rev_s_exp1)*accel_err[i]*0.01 + abs(M[0]*0.001 * a*0.01)*np.ones(s_exp1.shape)*(0.05/100)
    fitparams, paramErrors = show_with_trend(t_exp1, pot, pot_err, poly2, fig4, 221+i, f"Energia Potencial X Tempo (aceleração com j = {i+1})", lambda T: 0.2404259605+(M[0]*0.001*G*0.01*((-0.5*G*T**2)+1)/100))
    potenciais.append(pot)
    potenciais_err.append(pot_err)

    reference = [-2.404259605, 0, 0.26477955]
    referenceErrors = [0.069378833, 0, 0.001265184]
    for i in range(3):
        z = zscore(fitparams[i], reference[i], paramErrors[i], referenceErrors[i])
        print(f"kinectic with j = {j} {param[i]} measurement {zscore_compatible(z)}")

# energia mecânica (cinética + potencial)

print()
print("MECANICA")
fig5 = plt.figure(figsize=FIGURE_SIZE)
fig5.set_facecolor('#cccccc')
fitparams, paramErrors = show_with_trend(t_exp1, potenciais[0] + cineticas[0], potenciais_err[0] + cineticas_err[0], poly0, fig5, 111, f"Energia Mecânica X Tempo", lambda T: np.ones(T.shape)*0.26477955)

reference = [0.26477955, 0, 0.26477955]
referenceErrors = [0.001265184, 0, 0.001265184]
z = zscore(fitparams[0], reference[0], paramErrors[0], referenceErrors[0])
print(f"Mechanical energy measurement {zscore_compatible(z)}")

if platform == 'linux':
    direct = '/home/joao/Documents/ufba/eletromag/queda_livre/img_exp1/'
else:
    direct = 'C:/Users/cisco/Downloads/'

fig0.savefig(direct + 'deslocamento.png')
fig1.savefig(direct + 'velocidade.png')
fig2.savefig(direct + 'acelera.png')
fig3.savefig(direct + 'cinetica.png')
fig4.savefig(direct + 'potencial.png')
fig5.savefig(direct + 'mecanica.png')

