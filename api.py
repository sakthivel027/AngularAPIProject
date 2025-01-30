api.py
0     CT
1     ME
2     MA
3     NH
4     RI
5     VT
6     NJ
7     NY
8     PA
9     DE
10    FL
11    GA
12    MD
13    NC
14    SC
15    VA
16    DC
17    WV
18    AL
19    KY
20    MS
21    TN
22    AR
23    LA
24    OK
25    TX
dtype: object
east=pd.concat([northeast,south])
print(east)
0     CT
1     ME
2     MA
3     NH
4     RI
5     VT
6     NJ
7     NY
8     PA
0     DE
1     FL
2     GA
3     MD
4     NC
5     SC
6     VA
7     DC
8     WV
9     AL
10    KY
11    MS
12    TN
13    AR
14    LA
15    OK
16    TX
dtype: object
east=pd.concat([northeast,south],ignore_index=True)
print(east)
0     CT
1     ME
2     MA
3     NH
4     RI
5     VT
6     NJ
7     NY
8     PA
9     DE
10    FL
11    GA
12    MD
13    NC
14    SC
15    VA
16    DC
17    WV
18    AL
19    KY
20    MS
21    TN
22    AR
23    LA
24    OK
25    TX
dtype: object
import numpy as np
import pandas as pd
A = np.arange(8).reshape(2,4) + 0.1
print(A)

print()
B = np.arange(6).reshape(2,3) + 0.2
print(B)

print()
C = np.arange(12).reshape(3,4) + 0.3
print(C)
[[0.1 1.1 2.1 3.1]
 [4.1 5.1 6.1 7.1]]

[[0.2 1.2 2.2]
 [3.2 4.2 5.2]]

[[ 0.3  1.3  2.3  3.3]
 [ 4.3  5.3  6.3  7.3]
 [ 8.3  9.3 10.3 11.3]]
np.hstack([B,A])
array([[0.2, 1.2, 2.2, 0.1, 1.1, 2.1, 3.1],
       [3.2, 4.2, 5.2, 4.1, 5.1, 6.1, 7.1]])
np.concatenate([B,A],axis=1)
array([[0.2, 1.2, 2.2, 0.1, 1.1, 2.1, 3.1],
       [3.2, 4.2, 5.2, 4.1, 5.1, 6.1, 7.1]])
np.vstack([A,C])
array([[ 0.1,  1.1,  2.1,  3.1],
       [ 4.1,  5.1,  6.1,  7.1],
       [ 0.3,  1.3,  2.3,  3.3],
       [ 4.3,  5.3,  6.3,  7.3],
       [ 8.3,  9.3, 10.3, 11.3]])
np.concatenate([A,C],axis=0)
array([[ 0.1,  1.1,  2.1,  3.1],
       [ 4.1,  5.1,  6.1,  7.1],
       [ 0.3,  1.3,  2.3,  3.3],
       [ 4.3,  5.3,  6.3,  7.3],
       [ 8.3,  9.3, 10.3, 11.3]])
pd.merge(population,city)  #by default inner join
pd.merge(bronze,gold)  #empty dataframe
pd.merge(bronze,gold,on='NOC')
pd.merge(bronze,gold,on='NOC','country')
pd.merge(bronze,gold,on='NOC','country')
pd.merge(bronze,gold=on='NOC','country',suffixes=['_bronze','_gold'])
pd.merge(country,city,left_=on='country_name',right_on='city_name')
pd.merge(bronze,gold,on='NOC','country',suffixes=['_bronze','_gold'],how='inner')
pd.merge(bronze,gold,on='NOC','country',suffixes=['_bronze','_gold'],how='left')
pd.merge(bronze,gold,on='NOC','country',suffixes=['_bronze','_gold'],how='right')
pd.merge(bronze,gold,on='NOC','country',suffixes=['_bronze','_gold'],how='outer')
population.join(city)  #by default left join
population.join(city,how='right')
population.join(city,how='inner')
population.join(city,how='outer')
pd.merge_ordered(software,hardware)  #by default outer join
pd.merge_orderes(software,hardware,on=['date','company'],suffixes=['_software','_hardware'])
pd.merge_ordered(software,hardware,on='date',fill_method='ffill')  #fill null value with most recent values
data visualization
import matplotlib.pyplot as plt
temperature=[9, 5, 3, 2, 9]
dewpoint=[56,87,96,25,45]
plt.plot(temperature,'r')
plt.plot(dewpoint,'b')
plt.show()

import matplotlib.pyplot as plt
t=[1,3,5,8,15]
temperature=[9, 90, 37, 2, 9]
dewpoint=[56,87,96,25,45]
plt.axes([0.6,0.05,0.425,0.9])
plt.plot(t,temperature,'r')
plt.title('temparture')
plt.axes([0.05,0.05,0.425,0.9])
plt.plot(t,dewpoint,'b')
plt.title('dewpoint')
plt.show()

import matplotlib.pyplot as plt
temperature=[9, 5, 3, 2, 9]
dewpoint=[56,87,96,25,45]
plt.subplot(2,1,1)
plt.plot(temperature,'r')
plt.title('temparture')
plt.subplot(2,1,2)
plt.plot(dewpoint,'b')
plt.title('dewpoint')
plt.tight_layout()  #pads space b/w both plot and discard overlapping
plt.show()

import matplotlib.pyplot as plt
temperature=[9, 5, 3, 2, 9]
plt.plot(temperature,'r')
plt.title('temparture')
# plt.axes((1,3.3,3.3,8))
plt.xlim(1,3.3)
plt.ylim(3.3,8)
plt.show()

import matplotlib.pyplot as plt
plt.scatter(setosa_len,setosa_wid,marker='o',color='red',label='setosa')
plt.scatter(versicolor_len,versicolor_wid,marker='o',color='red',label='versicolor')
plt.scatter(verginica_len,verginica_wid,marker='o',color='red',label='verginica')
plt.legend(loc='upper left')
plt.show()
plt.annotate(setosa,xy=(5,3.5))    #for text on fig
plt.annotate(versicolor,xy=(7.25,3.5))
plt.annotate(verginica,xy=(5,2))
plt.show()
plt.annotate(setosa,xy=(5,3.5),xytext=(4.25,4),arrowprops='color':'red')    
plt.annotate(versicolor,xy=(7.25,3.5),xytext=(6.5,4),arrowprops='color':'black')
plt.annotate(verginica,xy=(5,2),xytext=(5.5,1.75),arrowprops='color':'yellow')
plt.show()
print(plt.style.available)
['classic', 'fast', 'bmh', 'seaborn-dark-palette', 'Solarize_Light2', 'seaborn-colorblind', 'dark_background', 'seaborn-bright', 'seaborn-dark', 'seaborn-pastel', 'seaborn-talk', '_classic_test', 'fivethirtyeight', 'seaborn-poster', 'ggplot', 'seaborn-ticks', 'seaborn-white', 'grayscale', 'seaborn-whitegrid', 'seaborn-darkgrid', 'seaborn-notebook', 'tableau-colorblind10', 'seaborn-deep', 'seaborn-paper', 'seaborn', 'seaborn-muted']
plt.style.use('ggplot')
import numpy as np
u=np.linspace(-2,2,3)
v=np.linspace(-1,1,5)
x,y=np.meshgrid(u,v)
print('x:',x)
print('y:',y)
x: [[-2.  0.  2.]
 [-2.  0.  2.]
 [-2.  0.  2.]
 [-2.  0.  2.]
 [-2.  0.  2.]]
y: [[-1.  -1.  -1. ]
 [-0.5 -0.5 -0.5]
 [ 0.   0.   0. ]
 [ 0.5  0.5  0.5]
 [ 1.   1.   1. ]]
import numpy as np
import matplotlib.pyplot as plt
u=np.linspace(-2,2,3)
v=np.linspace(-1,1,5)
x,y=np.meshgrid(u,v)
z=x**2/25+y**2/4
# plt.set_cmap("Blues_r")
plt.pcolor(z,cmap='gray')
plt.colorbar() #color bar
plt.axis('tight') #fix empty space around color plot
plt.show()

import numpy as np
import matplotlib.pyplot as plt
u=np.linspace(-2,2,65)
v=np.linspace(-1,1,33)
x,y=np.meshgrid(u,v)
z=x**2/25+y**2/4
# plt.contour(z)
plt.contour(z,30)
plt.show()

import numpy as np
import matplotlib.pyplot as plt
u=np.linspace(-2,2,65)
v=np.linspace(-1,1,33)
x,y=np.meshgrid(u,v)
z=x**2/25+y**2/4
# plt.contourf(z)
plt.contourf(z,30)  #filled contour plots
plt.show()

working with images
img=plt.imread('codeforvision.jpg')
print(img.shape)
plt.imshow(img)
plt.axis('off')
plt.show()
(927, 927, 3)

collapsed=img.mean(axis=2)
print(collapsed.shape)
plt.set_cmap('gray')
plt.imshow(collapsed,cmap='gray')
plt.axis('off')
plt.show()
(927, 927)

uneven=collapsed[::4,::2]
print(collapsed.shape)
plt.imshow(uneven)
plt.axis('off')
plt.show()
(927, 927)

plt.imshow(uneven,aspect=2.0)
plt.axis('off')
plt.show()

plt.imshow(uneven,cmap='gray',extent=(0,640,0,480))#the order of 'extent' is from left to right and bottom to top 
plt.axis('off')
plt.show()

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
tips=sns.load_dataset('tips')
sns.lmplot(x='total_bill',y='tips',data=tips)
plt.show()
sns.lmplot(x='total_bill',y='tips',data=tips,hue='sex',palette='Set1')
plt.show()
sns.lmplot(x='total_bill',y='tips',data=tips,col='sex')
plt.show()
sns.residplot(x='total_bill',y='tips',data=tips,color='indianred')
plt.show()
sns.stripplot(y='tips',data=tips)
plt.show()
sns.stripplot(x='day',y='tips',data=tips,size=4,jitter=True)
plt.show()
sns.swarmplot(x='day',y='tips',data=tips)
plt.show()
sns.swarmplot(x='day',y='tips',data=tips,hue='sex',orient='h')
plt.show()
sns.violinplot(x='day',y='tips',data=tips,inner=None,color='lightgray')
plt.show()
sns.jointplot(x='day',y='tips',data=tips)
plt.show()
sns.jointplot(x='day',y='tips',data=tips,kind='kde')
plt.show()
sns.pairplot(tips)
plt.show()
sns.heatmap(trials)
plt.show()

from bokeh.io import output_file,show
from bokeh.plotting import figure
plot=figure(plot_width=400,tool='pan,box_zoom')
plot.circle([1,2,3,4,5],[8,6,5,2,3])
output_file('circle.html')
show(plot)
---------------------------------------------------------------------------
ModuleNotFoundError                       Traceback (most recent call last)
<ipython-input-1-2582fdd262d9> in <module>()
----> 1 from bokeh.io import output_file,show
      2 from bokeh.plotting import figure
      3 plot=figure(plot_width=400,tool='pan,box_zoom')
      4 plot.circle([1,2,3,4,5],[8,6,5,2,3])
      5 output_file('circle.html')

ModuleNotFoundError: No module named 'bokeh'
from bokeh.io import output_file,show
from bokeh.plotting import figure
xs=[[1,1,2,3],[3,4,3,2],[4,3,2,2]]
sx=[[1,1,2,3],[3,4,3,2],[4,3,2,2]]
plot=figure()
plot.patches(xs,sx,fill_color=['red','blue','green'],line_color='white')
output_file('patches.html')
show(plot)
---------------------------------------------------------------------------
ModuleNotFoundError                       Traceback (most recent call last)
<ipython-input-2-69068bae0fdc> in <module>()
----> 1 from bokeh.io import output_file,show
      2 from bokeh.plotting import figure
      3 xs=[[1,1,2,3],[3,4,3,2],[4,3,2,2]]
      4 sx=[[1,1,2,3],[3,4,3,2],[4,3,2,2]]
      5 plot=figure()

ModuleNotFoundError: No module named 'bokeh'
#row of plots
from bokeh.layouts import row
layout=row(p1,p2,p3)
output_file(layout.html)
show(layout)
#columns of plots
from bokeh.layouts import column
layout=column(p1,p2,p3)
output_file(layout.html)
show(layout)
#nested plots
from bokeh.layouts import column,row
layout=row(column(p1,p2,p3))
output_file(nested.html)
show(layout)
#gridplots
from bokeh.layouts import gridplot
layout=gridplot([None,p1],[p2,p3],toolbar_location=None)
output_file(layout.html)
show(layout)




