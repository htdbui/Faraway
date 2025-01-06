---
title: "Airplane"
author: "db"
---

# 1. Beschreibung der Daten

- Monatliche Anzahl der Flugpassagiere aus den frühen Jahren der Luftfahrt.
- 144 Beobachtungen.
- 2 Variablen:
  - pass: Anzahl der Passagiere
  - year: Zeitpunkt im Format (Jan=0.083, Feb=0.167, Mar=0.250, Apr=0.333, May=0.417, Jun=0.500, Jul=0.583, Aug=0.667, Sep=0.750, Oct=0.833, Nov=0.917, Dec=1.000)

# 2. Packages und Daten

```python=
import pandas as pd, numpy as np, scipy as sp
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
import faraway.utils
```

```python=
import faraway.datasets.air
air = faraway.datasets.air.load()
air = air.rename(columns={'pass': 'passengers'}) # because pass is a key word
air.head()
```

<table class="dataframe">
  <thead>
<tr style="text-align: right;">
  <th></th>
  <th>passengers</th>
  <th>year</th>
</tr>
  </thead>
  <tbody>
<tr>
  <th>0</th>
  <td>112</td>
  <td>49.083333</td>
</tr>
<tr>
  <th>1</th>
  <td>118</td>
  <td>49.166667</td>
</tr>
<tr>
  <th>2</th>
  <td>132</td>
  <td>49.250000</td>
</tr>
<tr>
  <th>3</th>
  <td>129</td>
  <td>49.333333</td>
</tr>
<tr>
  <th>4</th>
  <td>121</td>
  <td>49.416667</td>
</tr>
  </tbody>
</table>

```python=
plt.plot(air['year'], air['passengers']);
```

![png](Figures\airplane_4_0.png)

# 3. Lineare Regression

```python=
# Create elements for running regression
X = pd.DataFrame({'Intercept':1, 'year':air['year']})
y = np.log(air['passengers'])
lmod = sm.OLS(y,X).fit()
lmod.sumary()
```

               coefs stderr tvalues pvalues
    Intercept -1.095  0.184   -5.93  0.0000
        year   0.121  0.003   36.05  0.0000
    n=144 p=2 Residual SD=0.139 R-squared=0.90

```python=
lmod2 = smf.ols(formula='np.log(passengers) ~ year', data=air).fit()
lmod2.sumary()
```

               coefs stderr tvalues pvalues
    Intercept -1.095  0.184   -5.93  0.0000
        year   0.121  0.003   36.05  0.0000
    n=144 p=2 Residual SD=0.139 R-squared=0.90

- Beide, `statsmodels.api` und `statsmodels.formula.api`, stammen aus dem selben Paket `statsmodels`.

- Sie bieten verschiedene Schnittstellen für statistische Modelle:

- `statsmodels.api`:
  
  - Traditionelle Schnittstelle für statistisches Modellieren.
  - Manuelle Erstellung von Designmatrizen (z.B. mit pandas DataFrames) erforderlich.

- `statsmodels.formula.api`:
  
  - Formelbasierte Schnittstelle ähnlich zu R.
  - Modelle mit Formeln angeben, inklusive Transformationen und Interaktionen.
  - Automatische Erstellung von Designmatrizen.

```python=
plt.plot(air['year'], air['passengers'])
plt.plot(air['year'],np.exp(lmod.predict()));
```

![png](Figures\airplane_12_0.png)

## 3.1 Autoregression

$y_t = \beta_0 + \beta_1 y_{t-1} + \beta_{12} y_{t-12} + \beta_{13} y_{t-13} + \epsilon_t$

```python=
# Create columns of lags
air['lag1'] = np.log(air['passengers']).shift(1)
air['lag12'] = np.log(air['passengers']).shift(12)
air['lag13'] = np.log(air['passengers']).shift(13)
airlag = air.dropna(); airlag
```

<table class="dataframe">
  <thead>
<tr style="text-align: right;">
  <th></th>
  <th>passengers</th>
  <th>year</th>
  <th>lag1</th>
  <th>lag12</th>
  <th>lag13</th>
</tr>
  </thead>
  <tbody>
<tr>
  <th>13</th>
  <td>126</td>
  <td>50.166667</td>
  <td>4.744932</td>
  <td>4.770685</td>
  <td>4.718499</td>
</tr>
<tr>
  <th>14</th>
  <td>141</td>
  <td>50.250000</td>
  <td>4.836282</td>
  <td>4.882802</td>
  <td>4.770685</td>
</tr>
<tr>
  <th>15</th>
  <td>135</td>
  <td>50.333333</td>
  <td>4.948760</td>
  <td>4.859812</td>
  <td>4.882802</td>
</tr>
<tr>
  <th>16</th>
  <td>125</td>
  <td>50.416667</td>
  <td>4.905275</td>
  <td>4.795791</td>
  <td>4.859812</td>
</tr>
<tr>
  <th>17</th>
  <td>149</td>
  <td>50.500000</td>
  <td>4.828314</td>
  <td>4.905275</td>
  <td>4.795791</td>
</tr>
<tr>
  <th>...</th>
  <td>...</td>
  <td>...</td>
  <td>...</td>
  <td>...</td>
  <td>...</td>
</tr>
<tr>
  <th>139</th>
  <td>606</td>
  <td>60.666667</td>
  <td>6.432940</td>
  <td>6.326149</td>
  <td>6.306275</td>
</tr>
<tr>
  <th>140</th>
  <td>508</td>
  <td>60.750000</td>
  <td>6.406880</td>
  <td>6.137727</td>
  <td>6.326149</td>
</tr>
<tr>
  <th>141</th>
  <td>461</td>
  <td>60.833333</td>
  <td>6.230481</td>
  <td>6.008813</td>
  <td>6.137727</td>
</tr>
<tr>
  <th>142</th>
  <td>390</td>
  <td>60.916667</td>
  <td>6.133398</td>
  <td>5.891644</td>
  <td>6.008813</td>
</tr>
<tr>
  <th>143</th>
  <td>432</td>
  <td>61.000000</td>
  <td>5.966147</td>
  <td>6.003887</td>
  <td>5.891644</td>
</tr>
  </tbody>
</table>
<p>131 rows × 5 columns</p>

```python=
# Create elements for running regression
X = airlag.loc[:,('lag1', 'lag12', 'lag13')]
X.insert(0, 'Intercept', 1)
y = np.log(airlag['passengers'])
# Run regression
lmod = sm.OLS(y,X).fit()
lmod.sumary()
```

               coefs stderr tvalues pvalues
    Intercept  0.138  0.054    2.58  0.0109
        lag1   0.692  0.062   11.19  0.0000
        lag12  0.922  0.035   26.53  0.0000
        lag13 -0.632  0.068   -9.34  0.0000
    n=131 p=4 Residual SD=0.042 R-squared=0.99

```python=
plt.plot(air['year'], air['passengers'])
plt.xlabel("Year")
plt.ylabel("Passengers")
plt.plot(airlag['year'],np.exp(lmod.predict()),linestyle='dashed');
```

![png](Figures\airplane_17_0.png)

# 4. Vorhersage

- Vorhersage der Passagierzahl für den nächsten Monat

```python=
# Get the values of lags
z = np.log(air['passengers'].iloc[[-1,-12,-13]]).values; z
```

    array([6.06842559, 6.03308622, 6.00388707])

```python=
# Create element for running prediction
x0 = pd.DataFrame([{"const":1,"lag1": z[0], "lag12": z[1], "lag13": z[2]}])
# Run prediction
lmod.get_prediction(x0).summary_frame()
```

<table class="dataframe">
  <thead>
<tr style="text-align: right;">
  <th></th>
  <th>mean</th>
  <th>mean_se</th>
  <th>mean_ci_lower</th>
  <th>mean_ci_upper</th>
  <th>obs_ci_lower</th>
  <th>obs_ci_upper</th>
</tr>
  </thead>
  <tbody>
<tr>
  <th>0</th>
  <td>6.103985</td>
  <td>0.006375</td>
  <td>6.09137</td>
  <td>6.116601</td>
  <td>6.020619</td>
  <td>6.187351</td>
</tr>
  </tbody>
</table>
