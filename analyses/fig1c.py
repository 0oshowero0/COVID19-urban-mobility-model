import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from datetime import datetime
import matplotlib.pyplot as plt
from scipy.stats import pearsonr,spearmanr
from sklearn.linear_model import LinearRegression
import datetime
from adjustText import adjust_text
from sklearn.metrics import r2_score

plt.rcParams['ps.fonttype'] = 42
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['font.family'] = ['Arial']
plt.rcParams['font.size'] =  20



confirm = pd.read_csv("./fig1c_data/time_series_covid19_confirmed_US.csv")
mobility = pd.read_csv('./fig1c_data/applemobilitytrends-2020-07-07.csv')


city_list = [['New York','New York City','New York'],['Cook','Cook County','Illinois'],
['Nassau','Nassau County','New York'],
['Los Angeles','Los Angeles','California'],['Bergen','Bergen County','New Jersey'],
['Hudson','Hudson County','New Jersey'],['Philadelphia','Philadelphia','Pennsylvania'],['Middlesex','Middlesex County','Massachusetts'],
['Union','Union County','New Jersey'],['Miami-Dade','Miami-Dade County','Florida'],
['Davidson','Davidson County','Tennessee'],['Will','Will County','Illinois'],['Hillsborough','Hillsborough County','Florida'],
['Suffolk','Suffolk County','New York'],
['New Haven','New Haven County','Connecticut'],
['Oakland','Oakland County','Michigan'],['Orleans','New Orleans','Louisiana'],['Ocean','Ocean County','New Jersey'],
['Harris','Harris County','Texas'],['King','King County','Washington']]


NewYorkCity = {'mobility':np.array([0.000040,0.000010]) ,'first':51,'second':63}  #下降
Cook = {'mobility':np.array([0.000082,0.000030]) ,'first':48,'second':63}  #下降 
Nassau = {'mobility':np.array([0.000042,0.000027]) ,'first':48,'second':63} #下降
Suffolk = {'mobility':np.array([0.000091,0.000043]) ,'first':53,'second':63} #下降
Los_Angeles = {'mobility':np.array([0.000120,0.000069]) ,'first':55,'second':63}#下降
Bergen = {'mobility':np.array([0.000089, 0.000022]) ,'first':50,'second':63}#下降 
Hudson = {'mobility':np.array([ 0.000157,0.000189]) ,'first':60,'second':63} # 上升 
Philadelphia = {'mobility':np.array([0.000129, 0.000037]) ,'first':54,'second':63} #下降
Middlesex = {'mobility':np.array([0.000083,0.000023]) ,'first':56,'second':63} #下降
Union = {'mobility':np.array([0.000191,0.000100]) ,'first':57,'second':63} #下降
Miami_Dade = {'mobility':np.array([0.000125,0.000036]) ,'first':54,'second':63}#下降
New_Haven = {'mobility':np.array([0.000157,0.000242]) ,'first':60,'second':63}#上升
Oakland = {'mobility':np.array([0.000078,0.000029]) ,'first':57,'second':63}#下降
New_Orleans = {'mobility':np.array([0.000087,0.000011]) ,'first':54,'second':63}#下降
Ocean = {'mobility':np.array([0.000043,0.000100]) ,'first':60,'second':63}#上升
Harris = {'mobility':np.array([0.000154,0.000209]) ,'first':60,'second':63}#上升
King = {'mobility':np.array([0.000186,0.000062]) ,'first':48,'second':63}#下降
Davidson = {'mobility':np.array([0.000148,0.000081]) ,'first':53,'second':63}#下降 
Will = {'mobility':np.array([0.000148,0.000226]) ,'first':58,'second':63}#上升
Hillsborough = {'mobility':np.array([ 0.000130,0.000072]) ,'first':57,'second':63}#下降


mobility_city_list = {'Bergen':Bergen,'Middlesex':Middlesex,'Hudson':Hudson,'Philadelphia':Philadelphia,'New York':NewYorkCity, 'Nassau':Nassau,'Cook':Cook, 'Union':Union,'Davidson':Davidson,'Will':Will,'Miami-Dade':Miami_Dade,'Hillsborough':Hillsborough,'Suffolk':Suffolk,'Los Angeles':Los_Angeles,'New Haven':New_Haven,'Oakland':Oakland,'Ocean':Ocean,'Harris':Harris,'King':King,'Orleans':New_Orleans}

data = {}
plot_data = []
county_name = []
for i in range(len(city_list)):
    if mobility_city_list.__contains__(city_list[i][0]):
        epi = confirm[np.logical_and(confirm['Admin2'] ==city_list[i][0],confirm['Province_State'] ==city_list[i][2])].iloc[:,11:].to_numpy().reshape(-1)   # 1月22日开始
        mobi = mobility[np.logical_and(mobility['region'] ==city_list[i][1],mobility['sub-region'] ==city_list[i][2])].iloc[:,6:].to_numpy().mean(axis=0).reshape(-1)    # 1月13日开始 如果有多个mobility直接取平均

        epi_second = epi[mobility_city_list[city_list[i][0]]['second']+1:]
        
        mobility_first = mobi[mobility_city_list[city_list[i][0]]['first']+9:mobility_city_list[city_list[i][0]]['second']+9+1]
        #mobility_first = mobi[:mobility_city_list[city_list[i][0]]['second']+9+1]
        mobility_second = mobi[mobility_city_list[city_list[i][0]]['second']+1+9:len(epi_second)+mobility_city_list[city_list[i][0]]['second']+1+9]

        mobility_first_true = mobility_first.mean()
        mobility_second_true = mobility_second.mean()

        true_drop = (mobility_first_true-mobility_second_true) / mobility_first_true
        pred_drop = (mobility_city_list[city_list[i][0]]['mobility'][0]-mobility_city_list[city_list[i][0]]['mobility'][1])/mobility_city_list[city_list[i][0]]['mobility'][0]
        pred_drop = (pred_drop - 0.1) / 2.7

        tmp = {'true':true_drop,'pred':pred_drop}
        data[city_list[i][0]] = tmp
        plot_data.append([true_drop,pred_drop])
        if city_list[i][0] == 'NY':
            county_name.append('New York City')
        elif city_list[i][0] == 'New_Orleans':
            county_name.append('New Orleans')
        elif city_list[i][0] == 'New_Haven':
            county_name.append('New Haven')
        elif city_list[i][0] == 'Los_Angeles':
            county_name.append('Los Angeles')
        else:
            county_name.append(city_list[i][0])





data = pd.DataFrame(data)
plot_data = np.array(plot_data)
pearsonr_val = pearsonr(plot_data[:,0], plot_data[:,1])
print(pearsonr_val)

fig, ax = plt.subplots(figsize=(8, 6))

plt.xlabel('Mobility Reduction in Real Data',size=20)
plt.ylabel('Estimated Mobility Reduction',size=20)
sns.regplot(plot_data[:,0],plot_data[:,1],ci=99,line_kws={'color':"#4c72b0"},scatter_kws={"s": 30,'color':'k','marker':'o'})
a = np.min((plot_data[:,0].min()-0.03,plot_data[:,1].min()-0.03))
b = np.max((plot_data[:,0].max()+0.03,plot_data[:,1].max()+0.03))

plt.xlim(-0.6,0.6)
plt.ylim(-0.6,0.6)

plt.grid(which='major',ls='--', alpha=0.8)

vals = ax.get_xticks()
plt.xticks(vals, ['{:,.0%}'.format(x) for x in vals])

vals = ax.get_yticks()
plt.yticks(vals, ['{:,.0%}'.format(x) for x in vals])


ax.tick_params(axis='both', which='major', labelsize=13)


clf = LinearRegression().fit(plot_data[:,0].reshape(-1,1), plot_data[:,1].reshape(-1,1))
params = [clf.coef_,clf.intercept_]
print(params)


def label_point(x, y, val, ax):
    tmp = []
    for i in range(len(val)):
        if i %2 == 1:
            tmp.append(ax.text(x[i], y[i], str(val[i]),ha='center', va='center',fontdict = {'size': 12, 'color': 'black'}))
        else:
            tmp.append(ax.text(x[i], y[i], str(val[i]),ha='center', va='center',fontdict = {'size': 12, 'color': 'black'}))
    return tmp


texts = label_point(plot_data[:,0],plot_data[:,1], county_name, plt.gca())
adjust_text(texts,only_move={'text': 'y'},arrowprops=dict(arrowstyle='-', 
                        color='gray',
                        lw=1))
xlim = ax.get_xlim()
ylim = ax.get_ylim()
r2 = pearsonr(plot_data[:,0].reshape(-1),plot_data[:,1].reshape(-1))[0] 
plt.text((xlim[1]-xlim[0]) / 500 + xlim[0] *0.85 , (ylim[1]-ylim[0]) / 8 *6.8 + ylim[0],'Pearson\'s R: '+"%(r2).3f"%{'r2':r2},fontdict = {'size': 13, 'color': 'black'})
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.tight_layout()
plt.show()

