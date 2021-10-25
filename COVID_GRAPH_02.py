
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import datetime as dt
from matplotlib.patches import Patch


dati_prov = pd.read_csv("https://raw.githubusercontent.com/pcm-dpc/COVID-19/master/dati-province/dpc-covid19-ita-province-latest.csv")
dati_reg = pd.read_csv("https://raw.githubusercontent.com/pcm-dpc/COVID-19/master/dati-regioni/dpc-covid19-ita-regioni.csv")
dati_reg["denominazione_regione"]
dati_ita = pd.read_csv("https://raw.githubusercontent.com/pcm-dpc/COVID-19/master/dati-andamento-nazionale/dpc-covid19-ita-andamento-nazionale.csv")

len(dati_ita.index)-1
i=1
dati_ita.loc[len(dati_ita.index)-i,'totale_ospedalizzati']-dati_ita.loc[len(dati_ita.index)-i-1,'totale_ospedalizzati']

#Inserire regione e data di inizio
Regione="Lombardia"
data_inizio="2021-07-01"


filt_reg = ((dati_reg["denominazione_regione"] == Regione) & (dati_reg["data"] >= data_inizio))
filt_prov= ((dati_prov["denominazione_provincia"] == "Milano"))

x=dati_reg.loc[filt_reg, ["data"]]
nuoviIngressiTIReg=dati_reg.loc[filt_reg, ["ingressi_terapia_intensiva"]]

y=dati_reg.loc[filt_reg, ["terapia_intensiva"]]

y_avg_7=y.rolling(window=5).mean().shift(-2)
print((y_avg_7.count()))


x1=x['data'].astype("datetime64[D]")

print("ultimo data disponibile: " + str(x[-1:].values)[3:13])
y1=y["terapia_intensiva"].values[:]
print("Ultimi 3 dati terapie intensive Regione: " + str(y1[-3:]))
newIN=nuoviIngressiTIReg["ingressi_terapia_intensiva"].values[:]
newIN


dpi=50
deg=5
prevision=3
img = plt.imread(r"C:\Users\a263025\OneDrive - Enel Spa\Amendola\COVID\c6.jpg")
asse_x_poly=np.arange(0,len(y1))

coeff_poly,residuals, rank, singular_values, rcond=np.polyfit(asse_x_poly,y1,deg, full=True)
res = np.poly1d(coeff_poly)
asse_x_poly=np.arange(0,len(y1)+prevision)
der_res=res(asse_x_poly)[1:]-res(asse_x_poly)[:-1]
der2_res=der_res[1:]-der_res[:-1]

derivata=res.deriv()
derivata2=derivata.deriv()

fig1 = plt.figure(dpi=dpi)
fig1.set_size_inches(20,20)

#CALCOLO Y MASSIMO & MINIMO
if y1.max()>res(asse_x_poly).max():
    y_max=y1.max()
else:
    y_max=res(asse_x_poly).max()
if y1.min()>res(asse_x_poly).min():
    y_min=res(asse_x_poly).min()
else:
    y_min=y1.min()
if y_min>0 or y_min <0:
    y_min=0
else:
    y_min=y_min-50
    
y_max=y_max+50

ax=fig1.add_subplot(111)


asse_x=pd.date_range(start=str(x1.values[0]),end=str(x1.values[-1]+pd.Timedelta(prevision, unit='D')),freq="2D")#closed='left'
print(asse_x)
ax.set_xticks(asse_x_poly[::2])
print(asse_x_poly)
print(asse_x_poly[::2])
print(asse_x.strftime("%Y-%m-%d"))
ax.set_xticklabels(asse_x.strftime("%Y-%m-%d"), minor=False,fontsize=12,rotation=90)
ax.set_xlim(asse_x_poly[0], asse_x_poly[-1])

y_step=100
ax.set_yticks(np.arange(0,y_max, step=y_step))
ax.set_yticklabels(np.arange(0,y_max, step=y_step).astype(int), minor=False,fontsize=20,color='magenta')
ax.set_ylim(y_min, y_max)

ln1=ax.plot(y1,'X',markersize=12, color='magenta',label='IC Cases')
ln2=ax.plot(asse_x_poly, res(asse_x_poly),'--', color='black',label= str(deg) + ' deg Polynomial',linewidth=2.5)

ax.imshow(img,extent=[0, len(asse_x_poly), y_min, y_max],aspect='auto', alpha=0.2)
ax.grid(alpha=0.5, color="black")

j=0
point=[]
for i in range(0, len(der2_res)-1):
    if der2_res[i]*der2_res[i+1]<0:
        point.append({'x': i, 'y': der2_res[i]})
        j=j+1
print(point)
for i in range(0, len(point)):
    if i>0:
        ln4=ax.scatter(point[i]['x'], res(asse_x_poly)[point[i]['x']], marker='o', color='r', linewidth=25)
    else:
        ln4=ax.scatter(point[i]['x'], res(asse_x_poly)[point[i]['x']], marker='o', color='r', label=' Inflection points', linewidth=25)
        
#-------To be reviewed-----------
ax_newTI=ax.twinx()  # instantiate a second axes that shares the same x-axis
ln3=ax_newTI.plot(newIN,'--X',markersize=12, color='green',label='INTENSIVE CARE DAILY NEW CASES')
ax_newTI.tick_params(axis='y', labelcolor='green')
ax_newTI.set_ylabel('', color='green',size=20)
#ax_newTI.set_yticklabels(np.arange(0,newIN.max(), step=10).astype(int), minor=False,fontsize=20)
ax_newTI.tick_params(axis='y',labelsize=20)


ax.tick_params(axis='y')
ax.legend(loc=1,fontsize=20)
ax_newTI.legend(loc=3,fontsize=20)
plt.title(label="INTENSIVE CARE " + Regione.upper(),fontsize=30 )


fig2 = plt.figure(dpi=dpi)
fig2.set_size_inches(10,10)
ax2=fig2.add_subplot(111)
ax2.plot(der2_res*10,"-.",color='red', label='der_2')
ax2.plot(der_res,label="der_1")
plt.legend()
plt.grid()

path_fig=r"C:\"
nome="\COVID_" + str(x[-1:].values)[3:13] + "_" + Regione.upper()
fig1.savefig(path_fig + nome +".png")


#dati_ita.loc[len(dati_ita.index)-i,'totale_ospedalizzati']-dati_ita.loc[len(dati_ita.index)-i-1,'totale_ospedalizzati']
Stato="ITALY"
filt_ita = ((dati_ita["data"] >= data_inizio))
x=dati_ita.loc[filt_ita, ["data"]]
#print(x)
y=dati_ita.loc[filt_ita, ["terapia_intensiva"]]
#print(y)
x1=x['data'].astype("datetime64[D]")
#print(x1)
y1=y["terapia_intensiva"].values[:]
#print(y1)

#x=dati_reg.loc[filt_reg, ["data"]]
nuoviIngressiTIITA=dati_ita.loc[filt_ita, ["ingressi_terapia_intensiva"]]
newINITA=nuoviIngressiTIITA["ingressi_terapia_intensiva"].values[:]
#ingressi_terapia_intensiva
print(newINITA)
print("Ultimi 3 dati terapie intensive Regione: " + str(y1[-3:]))

dpi=50
deg=5
prevision=3
img = plt.imread(r"C:\Users\a263025\OneDrive - Enel Spa\Amendola\COVID\c4.jpg")
asse_x_poly=np.arange(0,len(y1))
#coeff_poly=np.polynomial.polynomial.Polynomial.fit(asse_x_poly,y1,9)

coeff_poly,residuals, rank, singular_values, rcond=np.polyfit(asse_x_poly,y1,deg, full=True)
res = np.poly1d(coeff_poly)
asse_x_poly=np.arange(0,len(y1)+prevision)
der_res=res(asse_x_poly)[1:]-res(asse_x_poly)[:-1]
der2_res=der_res[1:]-der_res[:-1]

fig1 = plt.figure(dpi=dpi)
fig1.set_size_inches(20,20)

#CALCOLO Y MASSIMO & MINIMO
if y1.max()>res(asse_x_poly).max():
    y_max=y1.max()
else:
    y_max=res(asse_x_poly).max()
if y1.min()>res(asse_x_poly).min():
    y_min=res(asse_x_poly).min()
else:
    y_min=y1.min()
if y_min>0 or y_min<0:
    y_min=0
else:
    y_min=y_min-50
    
y_max=y_max+50
#####################
ax=fig1.add_subplot(111)
#ax.imshow(img, zorder=0,extent=[0, 100, 0, 800])

#ax.imshow(img, extent=[0, len(y1), 0, y1.max()+50])


asse_x=pd.date_range(start=str(x1.values[0]),end=str(x1.values[-1]+pd.Timedelta(prevision, unit='D')),freq="2D") #,closed='left')
print(asse_x)
print(asse_x_poly[::2])
ax.set_xticks(asse_x_poly[::2])

#print(asse_x_poly[::2])
#print(asse_x.strftime("%Y-%m-%d"))
ax.set_xticklabels(asse_x.strftime("%Y-%m-%d"), minor=False,fontsize=12,rotation=90)
ax.set_xlim(asse_x_poly[0], asse_x_poly[-1])


y_step=100
ax.set_yticks(np.arange(0,y_max, step=y_step))
ax.set_yticklabels(np.arange(0,y_max, step=y_step).astype(int), minor=False,fontsize=20, color='magenta')
ax.set_ylim(y_min, y_max)

ax.plot(y1,'X',markersize=12, color='magenta',label='IC Cases')
ax.plot(asse_x_poly, res(asse_x_poly),'--', color='black',label=str(deg) +' deg Polynomial',linewidth=2.5)
ax.imshow(img,extent=[0, len(asse_x_poly), y_min, y_max], alpha=0.3,aspect='auto')
plt.grid(alpha=0.5, color="black")

j=0
point=[]
for i in range(0, len(der2_res)-1):
    if der2_res[i]*der2_res[i+1]<0:
        point.append({'x': i, 'y': der2_res[i]})
        j=j+1
print(point)
for i in range(0, len(point)):
    if i>0:
        ax.scatter(point[i]['x'], res(asse_x_poly)[point[i]['x']], marker='o', color='r', linewidth=25)
    else:
        ax.scatter(point[i]['x'], res(asse_x_poly)[point[i]['x']], marker='o', color='r', label='Inflection point', linewidth=25)
#ax.plot(asse_x_poly,res(asse_x_poly),"-",np.arange(0,len(y1)),y1,".")

#-------To be reviewed-----------
ax_newTIITA=ax.twinx()  # instantiate a second axes that shares the same x-axis
ln3=ax_newTIITA.plot(newINITA,'--X',markersize=12, color='green',label='INTENSIVE CARE DAILY NEW CASES')
ax_newTIITA.set_yticks(np.arange(0,newINITA.max(), step=50))
ax_newTIITA.tick_params(axis='y', labelcolor='green')
ax_newTIITA.set_ylabel('', color='green',size=20)
ax_newTIITA.set_yticklabels(np.arange(0,newINITA.max(), step=50).astype(int), minor=False,fontsize=20)
#ax_newTIITA.tick_params(axis='y',labelsize=20)
print("ciao",newINITA.max())

ax.tick_params(axis='y')
ax.legend(fontsize=20)
ax_newTIITA.legend(loc=4,fontsize=20)

#ax.tick_params(axis='y')
##ax.patch.set_facecolor('grey')
#plt.legend(fontsize=20)
plt.title(label="INTENSIVE CARE " + Stato,fontsize=30 )

fig2 = plt.figure(dpi=dpi)
fig2.set_size_inches(10,10)
ax2=fig2.add_subplot(111)
ax2.plot(der2_res*10,"-.",color='red', label='der_2')
ax2.plot(der_res,label="der_1")
plt.legend()
plt.grid()
print(res(asse_x_poly)[-1])

path_fig=r"C:\"
nome="\COVID_" + str(x[-1:].values)[3:13] + "_ITA"
fig1.savefig(path_fig + nome +".png")