# -*- coding: utf-8 -*-
"""
уравнение Beatie-Bridgman для метана
"""

#%%пакеты

import numpy as np

# %matplotlib qt
import matplotlib.pyplot as plt
from numba import jit
from scipy.integrate import quad, nquad
from scipy.optimize import fsolve, root
from time import time

#%%константы

# A0 = 2.2769
# a = 0.1855e-1
# B0 = 0.5587e-1
# b = -0.1587e-1
# c = 12.83e+4
# R = 0.8206e-1

vcr = 0.8108072179e-1
Tcr = 217.1168402
pcr = 71.71904162142235//0.9999999994

a1 = 3.063888379
b1 = 2.111222494
b2 = -4.829189723
b3 = -0.4736979005
g1 = 0.4132314100
g2 = 1.104843018
g3 = -0.3264093008
d1 = -0.6388837554e-1

Tc1 = 0.8584544774
Tc2 = 0.8711728693
Tc3 = 1.

#%%функции

@jit("float64(float64)")
def alpha(T):
  """
  коэффициент перед 1/v**2
  """
  return a1*T

@jit("float64(float64)")
def beta(T):
  """
  коэффициент перед 1/v**2
  """
  # return R*T*B0 - A0 -R*c/T**2
  return b1*T + b2 + b3/T**2

@jit("float64(float64)")
def gamma(T):
  """
  коэффициент перед 1/v**3
  """
  # return A0*a - R*T*B0*b - R*B0*c/T**2
  return g1*T + g2 + g3/T**2

@jit("float64(float64)")
def delta(T):
  """
  коэффициент перед 1/v**4
  """
  # return R*B0*b*c/T**2
  return d1/T**2

# @jit("float64(float64)")
# def dbeta(T):
#   """
#   коэффициент перед 1/v**2
#   """
#   return R*B0 + 2*R*c/T**3

# @jit("float64(float64)")
# def dgamma(T):
#   """
#   коэффициент перед 1/v**3
#   """
#   return -R*B0*b + 2*R*B0*c/T**3

# @jit("float64(float64)")
# def ddelta(T):
#   """
#   коэффициент перед 1/v**4
#   """
#   return -2*R*B0*b*c/T**3

@jit("float64(float64,float64)")
def p0(v,T):
  # return R*T/v + beta(T)/v**2 + gamma(T)/v**3 + delta(T)/v**4
  return alpha(T)/v + beta(T)/v**2 + gamma(T)/v**3 + delta(T)/v**4

# @jit("float64(float64,float64)")
# def v0(p,T):
#   RT = R*T
#   return RT/p + beta(T)/(RT) + gamma(T)*p/(RT)**2 + delta(T)*p**2/RT**3

@jit("float64(float64,float64)")
def phi0(v,T):
  """
  потенциал Масье-Планка без функции K(T)
  """
  # RT = R*T
  # return np.log(v) - beta(T)/(RT*v) - gamma(T)/(2*RT*v**2) - delta(T)/(3*RT*v**3)
  return (alpha(T)*np.log(v) - beta(T)/v - gamma(T)/(2*v**2) - delta(T)/(3*v**3))/T

@jit("float64(float64,float64)")
def phi0_v(v,T):
  """
  производная потенциала Масье-Планка по v
  """
  # RT = R*T
  # return 1./v + beta(T)/(RT*v**2) + gamma(T)/(RT*v**3) + delta(T)/(RT*v**4)
  return p0(v,T)/T

# @jit("float64(float64,float64)")
# def phi0_T(v,T):
#   """
#   потенциал Масье-Планка без функции K(T)
#   """
#   return beta(T)/(R*T**2*v) - dbeta(T)/(R*T*v) + gamma(T)/(2*R*T**2*v**2) - dgamma(T)/(2*R*T*v**2) + delta(T)/(3*R*T**2*v**3) - ddelta(T)/(3*R*T*v**3)

@jit("float64(float64,float64)")
def phi0_vv(v,T):
  """
  2я производная потенциала Масье-Планка по v
  """
  return -alpha(T)/(T*v**2) - 2*beta(T)/(T*v**3) - 3*gamma(T)/(T*v**4) - 4*delta(T)/(T*v**4)

@jit("float64(float64)")
def Q1(v):
  """
  коэффициент для расчёта T из уравнения phi_vv = 0
  """
  return -10.36492150*(v-0.3431765207)**2/(3.063888379*v**2 + 4.222444988*v + 1.23969423)**2

@jit("float64(float64)")
def R1(v):
  """
  коэффициент для расчёта T из уравнения phi_vv = 0
  """
  return (-0.196372650100000 - 4.446797355*v**6 - 16.8527622200000*v**5 - 59.2813693300000*v**4 + 13.6407964100000*v**3 - 20.8923812400000*v**2 - 0.741506714000000*v)/(v*(3.063888379*v**2+4.222444988*v+1.23969423)**3)

@jit("float64(float64)")
def t1(v):
  """
  коэффициент для расчёта T из уравнения phi_vv = 0
  """
  return (-9.658379446*v + 3.314529054)/(3.063888379*v**2 + 4.222444988*v + 1.23969423)

@jit("float64(float64)")
def T1(v):
  """
  T из уравнения phi_vv = 0
  здесь один корень
  """
  a = np.sqrt(-Q1(v))
  b = np.arccosh(-np.abs(R1(v))/(Q1(v)*a))/3.
  return -2*np.sign(R1(v))*a*np.cosh(b) - t1(v)/3

# @jit("float64(float64,float64)")
# def phi0_vT(v,T):
#   """
#   производная потенциала Масье-Планка по v и T
#   """
#   return -beta(T)/(R*T**2*v**2) + dbeta(T)/(R*T*v**2) - gamma(T)/(R*T**2*v**3) + dgamma(T)/(R*T*v**3) - delta(T)/(R*T**2*v**4) + ddelta(T)/(R*T*v**4)

@jit("float64(float64,float64,float64)")
def PhTrEq1(v1,v2,T):
  """
  Первое уравнение фазового перехода
  сохраняется давление
  """
  return phi0_v(v2,T)-phi0_v(v1,T)
  # return alpha(T) * v1**3 * v2**3 + beta(T) * v1**2 * v2**2 * (v1+v2) + gamma(T)*v1*v2*(v1**2 + v1*v2 + v2**2) + delta(T)*(v1+v2)*(v1**2 + v2**2)

@jit("float64(float64,float64,float64)")
def PhTrEq2(v1,v2,T):
  """
  второе уравнение фазового перехода
  сохраняется химического потенциала Гиббса
  """
  return phi0(v2,T) - phi0(v1,T) - v2*phi0_v(v2,T) + v1*phi0_v(v1,T)
  # return 6 * alpha(T) * T * v1**3 *v2**3 * (np.log(v2) - np.log(v1)) + 6 * beta(T) * (T+1.) * v1**2 * v2**2 * (v2-v1) + 3*gamma(T)*(T+2.)*v1*v2*(v2**2 - v1**2) + 2*delta(T)*(T+3.)*(v2**3 - v1**3)

def PhTrEq12(vT,v2):
  """
  оба уравнения фазового прехода
  эта функция используется в fsolve
  vT -- массив, 0я координата -- v, 1я -- T
  """
  # print(vT)
  return np.array([PhTrEq1(vT[0],v2,vT[1]), PhTrEq2(vT[0],v2,vT[1])])

# @jit("float64[::1](float64,float64,float64)")
# def dPhTrEq1(v1,v2,T):
#   """
#   градиент первого уравнения фазового перехода
#   """
#   return np.array([-phi0_vv(v1,T),
#                    phi0_vT(v2,T) - phi0_vT(v1,T)])

# @jit("float64[::1](float64,float64,float64)")
# def dPhTrEq2(v1,v2,T):
#   """
#   градиент второго уравнения фазового перехода
#   """
#   return np.array([-phi0_v(v1,T) + phi0_v(v1,T) + v1*phi0_vv(v1,T),
#                    phi0_T(v2,T) - phi0_T(v1,T) - v2*phi0_vT(v2,T) + v1*phi0_vT(v1,T)])

# # @jit("float64[::2](float64[::1],float64)")
# def dPhTrEq12(vT,v2):
#   """
#   якобиан уравнений фазового перехода
#   эта функция используется в fsolve
#   vT -- массив, 0я координата -- v, 1я -- T
#   """
#   # print(vT)
#   return np.array([dPhTrEq1(vT[0],v2,vT[1]),
#                    dPhTrEq2(vT[0],v2,vT[1])])

@jit("float64(float64)")
def Q0(T):
  """
  коэффициент для расчёта дискриминантной кривой как корня кубического уравнения
  """
  return (-0.102671383351260e-1 - 9.01008028653404*10**(-14)*T - 1.68591017237454*T**2 - 0.688452057465862e-1*T**3 - 24.3584149530151*T**4 - 5.27791333047631*T**5 + 11058.7001479919*T**6 - 306.703373505428*T**7 - 1.03032744588014*10**5*T**8 - 27249.0805891247*T**9 + 3.09646726903040*10**5*T**10 + 2.32498286145247*10**5*T**11 - 3.33649850221692*10**5*T**12 - 5.39366772144997*10**5*T**13 - 75321.3401404666*T**14 + 3.87166065333298*10**5*T**15 + 3.58312293535191*10**5*T**16 + 68169.6519061252*T**17 - 1.10600323519970*10**5*T**18 - 1.12758938667185*10**5*T**19 - 55240.6709725617*T**20 - 16572.3558855060*T**21 - 3093.09420526207*T**22 - 330.535304046397*T**23 - 15.45327793*T**24)/T**4

@jit("float64(float64)")
def R0(T):
  """
  коэффициент для расчёта дискриминантной кривой как корня кубического уравнения
  """
  return (-0.104112947762068e-2 + 11488.0069903239*T**9 + 109.280771119576*T**7 - 2825.67345235808*T**6 - 1.44281208353296*T**5 - 21.4976201523839*T**4 - 0.104013537130609e-1*T**3 - 0.260372879615839*T**2 - 60.74785159*T**36 - 1949.03401945948*T**35 - 28660.8693430668*T**34 + 4.23665130134360*10**5*T**11 - 4.13528674560120*10**7*T**12 + 1.37221924072496*10**8*T**15 - 1.82205370715555*10**7*T**13 + 5.10774095514033*10**6*T**10 - 2.55711299073380*10**8*T**16 + 1.51460316399907*10**8*T**14 + 2.74582649591096*10**(-14)*T - 1.62566238124746*10**5*T**8 + 4.88749535309691*10**6*T**27 + 2.69981129730455*10**8*T**25 - 4.97516796126606*10**7*T**28 + 1.46809215304105*10**8*T**26 - 2.55623967636244*10**5*T**33 - 1.53845129027144*10**6*T**32 - 6.54706226156492*10**6*T**31 - 1.98846891906645*10**7*T**30 - 4.14903622631106*10**7*T**29 + 1.50359539025417*10**8*T**24 - 2.62797564498804*10**8*T**23 - 5.69728746958600*10**8*T**22 - 4.33785414799620*10**8*T**17 + 6.24483483326689*10**8*T**19 + 7.32732722942569*10**7*T**18 + 4.11303540103179*10**8*T**20 - 2.75172074054556*10**8*T**21)/T**6

@jit("float64(float64)")
def t0(T):
  """
  коэффициент для расчёта дискриминантной кривой как корня кубического уравнения
  """
  return (0.608012770700008 - 11.79319725*T**12 - 126.124310200000*T**11 - 505.821533000000*T**10 - 914.019270800000*T**9 - 455.793081400000*T**8 + 1051.62259800000*T**7 + 1520.27799400001*T**6 - 26.3096158099930*T**5 - 898.432875100011*T**4 + 2.03391098099968*T**3 + 50.1747067100007*T**2 - 4.35242591089163*10**(-15)*T)/T**2

@jit("float64(float64)")
def p1(T):
  """
  дискриминантная кривая (дискриминант от p0 по v)
  зависимость p от T
  2й корень
  """
  a = np.sqrt(-Q0(T))
  b = np.arccos(R0(T)/(Q0(T)*a))/3.
  return 2*a*np.cos(b) - t0(T)/3.

@jit("float64(float64)")
def p2(T):
  """
  дискриминантная кривая (дискриминант от p0 по v)
  зависимость p от T
  2й корень
  """
  a = np.sqrt(-Q0(T))
  b = np.arccos(R0(T)/(Q0(T)*a))/3. + 2*np.pi/3.
  return 2*a*np.cos(b) - t0(T)/3.

@jit("float64(float64)")
def p3(T):
  """
  дискриминантная кривая (дискриминант от p0 по v)
  зависимость p от T
  3й корень
  """
  a = np.sqrt(-Q0(T))
  b = np.arccos(R0(T)/(Q0(T)*a))/3. - 2*np.pi/3.
  return 2*a*np.cos(b) - t0(T)/3.

@jit("float64(float64)")
def PhTrV2(v):
  """
  Апроксимация графика фазового прехода в плоскости (v1,v2)
  аппроксимация гиперболой
  """
  k = 0.7
  b = 1-np.sqrt(k)
  return k/(v-b) + b

def fsolver(f, x0, x1, T, y0=0., drct=1, eps=1e-3, dx=1e-3):
  """
  решаю методом перебора в диапозоне [x0,x1]
  """
  # xM = ()
  
  if drct > 0:
    x = x0
  else:
    x = x1
  # xM = np.append(xM,x)
  
  n1 = 10**4
  dis1 = f(x,T) - y0
  if np.abs(dis1) < eps:
    flag1 = False
  else:
    flag1 = True
  i = 0
  dx1 = dx
  while flag1:
    i = i+1
    x = x + drct*dx1
    if (not x0 < x < x1) or (i > n1):
      flag1 = False
      x = np.nan
    else:
      dis2 = f(x,T) - y0
      # xM = np.append(xM,x)
      if np.abs(dis2) < eps:
        flag1 = False
      else:
        if dis1*dis2 < 0:
          x = x - drct*dx1
          dx1 = 0.5*dx1
        else:
          dis1 = dis2
  
  # return x, xM
  return x

def PhTrSolve(T, p01, p02, v11, v12, v21, v22, eps1=1e-3, eps2=1e-3, dp=1e-2, dv=1e-3):
  """
  решаю систему для фазового перехода перебором
  при заданных T и p нахожу значения два значение v
  далее проверяю выполнение 2го уравнения
  Если не выполнено, то меняю p
  """
  # disM = ()
  # dpM = ()
  # v1M = ()
  # v2M = ()
  # p1M = ()
  
  p00 = p01
  # p1M = np.append(p1M,p00)
  
  v1 = fsolver(p0, v11, v12, T, p00, drct=-1, eps=eps2, dx=1e-5)
  v2 = fsolver(p0, v21, v22, T, p00, drct=1, eps=eps2, dx=1e-1)
  # v1M = np.append(v1M,v1)
  # v2M = np.append(v2M,v2)
  
  if not (v1 == np.nan or v2 == np.nan):
    dis1 = PhTrEq2(v1,v2,T)
    # disM = np.append(disM,dis1)
    if np.abs(dis1) < eps1:
      flag1 = False
    else:
      flag1 = True
  else:
    flag1 = True
  
  scs = 1
  
  dp1 = dp
  # dpM = np.append(dpM,dp1)
  n1 = 10**5
  i = 0
  while flag1:
    i = i + 1
    p00 = p00 + dp1
    # dpM = np.append(dpM,dp1)
    if (not p01 <= p00 <= p02) or i > n1:
      flag1 = False
      v1 = np.nan
      v2 = np.nan
      scs = 0
    else:
      v1 = fsolver(p0, v11, v12, T, p00, drct=-1, eps=eps2*0.001)
      v2 = fsolver(p0, v21, v22, T, p00, drct=1, eps=eps2, dx=1e-1)
      # v1M = np.append(v1M,v1)
      # v2M = np.append(v2M,v2)
      # p1M = np.append(p1M,p00)
      if not (v1 == np.nan or v2 == np.nan):
        dis2 = PhTrEq2(v1,v2,T)
        # disM = np.append(disM,dis2)
        if np.abs(dis1) < eps1:
          flag1 = False
        else:
          flag1 = True
      else:
        flag1 = True
      if np.abs(dis2) < eps1:
        flag1 = False
      else:
        if dis1*dis2 < 0:
          p00 = p00 - dp1
          dp1 = 0.5*dp1
        else:
          dis1 = dis2
  
  return v1, v2, scs
  # return v1, v2, scs, disM, dpM, v1M, v2M, p1M

#%%фазовый переход при Tc1 < T < Tc2

nn = 50
TM1 = np.linspace(Tc1+0.0001,Tc2-0.0001,nn)
v1M1 = np.zeros(nn)
v2M1 = np.zeros(nn)
scsM1 = np.zeros(nn)

tt = time()
for i in range(nn):
  #считаю правую границу для v1 и левую для v2
  #эти границы -- это максимумы графика p0(v,TM1[i])
  p01 = p2(TM1[i])
  p02 = p3(TM1[i])
  v12 = fsolver(p0,0.1,1,TM1[i],y0=p02)
  v21 = fsolver(p0,v12,1,TM1[i],y0=p01)
  
  v1,v2,scs = PhTrSolve(TM1[i], p01, p02, 0.1, v12, v21, 10.)
  v1M1[i] = v1
  v2M1[i] = v2
  scsM1[i] = scs
print(time()-tt)

plt.figure()
plt.plot(scsM1)
plt.grid()
plt.show()

plt.rcParams['font.size'] = '16'

plt.figure(figsize=(8,8))
plt.plot(v1M1,v2M1)
plt.grid()
plt.show()

plt.figure(figsize=(8,8))
# plt.plot(v1M1,v2M1)
# plt.plot(v2M1,v1M1)
plt.plot(v1M1)
plt.plot(v2M1)
plt.legend(['1','2'])
plt.grid()
plt.show()

vv1M1 = np.concatenate((np.flip(v1M1),v2M1))
vv2M1 = np.concatenate((np.flip(v2M1),v1M1))

plt.figure(figsize=(8,8))
plt.plot(vv1M1,vv2M1)
# plt.plot(vv1M1)
plt.xlabel('v1')
plt.ylabel('v2',rotation=np.pi/2)
plt.grid()
plt.show()

pM1 = np.vectorize(p0)(v1M1,TM1)

plt.figure(figsize=(8,8))
plt.plot(TM1,pM1)
plt.xlabel('T')
plt.ylabel('p',rotation=np.pi/2)
plt.grid()
plt.show()


#%%фазовый переход при Tc2 < T

nn = 50
TM2 = np.linspace(Tc2+0.001,Tc3-0.0015,nn)
v1M2 = np.zeros(nn)
v2M2 = np.zeros(nn)
scsM2 = np.zeros(nn)

v0 = 0.174
tt = time()
for i in range(nn):
  #считаю правую границу для v1 и левую для v2
  #эти границы -- это максимумы графика p0(v,TM1[i])
  p01 = p2(TM2[i])
  p02 = p3(TM2[i])
  v12 = fsolver(p0,v0,1,TM2[i],y0=p01)
  v21 = fsolver(p0,1,10,TM2[i],y0=p02)
  
  v1,v2,scs = PhTrSolve(TM2[i],p01,p02,v0,v12,v21,20,eps1=1e-2)
  v1M2[i] = v1
  v2M2[i] = v2
  scsM2[i] = scs
print(time()-tt)

plt.figure()
plt.plot(scsM2)
plt.grid()
plt.show()

plt.rcParams['font.size'] = '16'

plt.figure(figsize=(8,8))
plt.plot(v1M2,v2M2)
plt.grid()
plt.show()

vv1M2 = np.concatenate((v1M2,np.ones(1),np.flip(v2M2)))
vv2M2 = np.concatenate((v2M2,np.ones(1),np.flip(v1M2)))

plt.figure(figsize=(8,8))
plt.plot(vv1M2,vv2M2)
plt.xlabel('v1')
plt.ylabel('v2',rotation=np.pi/2)
plt.grid()
plt.show()

pM2 = np.vectorize(p0)(v1M2,TM2)

plt.figure(figsize=(8,8))
plt.plot(TM2,pM2)
plt.xlabel('T')
plt.ylabel('p',rotation=np.pi/2)
plt.grid()
plt.show()


#%%график phi_vv < 0 в (v,T)

nn = 100
vM = np.linspace(0.01,2.,nn)
TM = np.vectorize(T1)(vM)

plt.rcParams['font.size'] = '16'

plt.figure(figsize=(8,8))
plt.fill_between(vM,TM,1.5*np.ones(nn))
plt.ylim([0.5,1.5])
plt.xlim([0,2])
plt.xlabel('v')
plt.ylabel('T',rotation=np.pi/2)
plt.grid()
plt.show()


#%%график phi_vv = 0 в (T,p)

nn = 1000
vM = np.linspace(0.01,20,nn)
TM = np.vectorize(T1)(vM)
pM = np.vectorize(p0)(vM,TM)

plt.rcParams['font.size'] = '16'

plt.figure(figsize=(8,8))
plt.plot(TM,pM)
plt.ylim([-5.3,1.2])
plt.xlim([0.4,1.1])
plt.xlabel('T')
plt.ylabel('p',rotation=np.pi/2)
plt.grid()
plt.show()
