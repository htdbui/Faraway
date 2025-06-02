---
title: " Savings"
author: "db"
---

# Datenbeschreibung

- Beobachtungen: 50
- Variablen:
  - sr: Sparquote (Prozentsatz des verfügbaren Einkommens, das gespart wird)
  - pop15: Prozentanteil der Bevölkerung unter 15 Jahren
  - pop75: Prozentanteil der Bevölkerung über 75 Jahren
  - dpi: Reales Pro-Kopf-Verfügbares Einkommen
  - ddpi: Wachstumsrate des realen Pro-Kopf-Verfügbaren Einkommens

# Packages und Daten

```python=
import pandas as pd, numpy as np
import scipy as sp, seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
import faraway.utils
```

```python=
import faraway.datasets.savings
savings = faraway.datasets.savings.load()
savings.head()
```

<table class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>sr</th>
      <th>pop15</th>
      <th>pop75</th>
      <th>dpi</th>
      <th>ddpi</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Australia</th>
      <td>11.43</td>
      <td>29.35</td>
      <td>2.87</td>
      <td>2329.68</td>
      <td>2.87</td>
    </tr>
    <tr>
      <th>Austria</th>
      <td>12.07</td>
      <td>23.32</td>
      <td>4.41</td>
      <td>1507.99</td>
      <td>3.93</td>
    </tr>
    <tr>
      <th>Belgium</th>
      <td>13.17</td>
      <td>23.80</td>
      <td>4.43</td>
      <td>2108.47</td>
      <td>3.82</td>
    </tr>
    <tr>
      <th>Bolivia</th>
      <td>5.75</td>
      <td>41.89</td>
      <td>1.67</td>
      <td>189.13</td>
      <td>0.22</td>
    </tr>
    <tr>
      <th>Brazil</th>
      <td>12.88</td>
      <td>42.19</td>
      <td>0.83</td>
      <td>728.47</td>
      <td>4.56</td>
    </tr>
  </tbody>
</table>

# Lineare Regression

```python=
lmod = smf.ols('sr ~ pop15 + pop75 + dpi + ddpi', savings).fit()
```

# Diagnosis

- Fehler: $\varepsilon \sim N(0,  \sigma^2 I)$
- Annahme des strukturellen Teils $\mathbb{E}(y) = X \beta$
- Ungewöhnliche Beobachtungen passen nicht ins Modell und können Modellwahl und -anpassung beeinflussen.
- Diagnose-Techniken:
  - Grafische Techniken:
    - Flexibel, aber schwerer zu interpretieren.
  - Numerische Techniken:
    - Präzisere und objektivere Ergebnisse durch Berechnungen.
- Modellfindung:
  - Erste Modellversuche oft unzureichend.
  - Regressionsdiagnosen identifizieren Verbesserungsbereiche.
  - Iterativer und interaktiver Prozess.

## Diagnosis.Fehleranalyse

- Fehler und Residuen:
  - Fehler $\varepsilon$ nicht beobachtbar, Residuen $\hat{\varepsilon}$ sind beobachtbar und können untersucht werden.
  - Residuen und Fehler sind verwandt, aber unterschiedlich.
  - Fehler $\varepsilon$ haben gleiche Varianz und sind unkorreliert.
  - Residuen $\hat{\varepsilon}$ haben nicht die gleiche Varianz und Korrelation.
  - Der Einfluss ist jedoch gering.
  - Diagnosen können auf $\hat{\varepsilon}$ angewendet werden, um Annahmen über $\varepsilon$ zu überprüfen.
  - $\hat{\varepsilon} = y-\hat{y} = (I-H)y = (I-H)\varepsilon$
  - $var(\hat{\varepsilon}) = var[(I-H)\varepsilon] = (I-H) \sigma^2$
- Annahme: $var(\varepsilon) = \sigma^2 I$
  - Unabhängigkeit der Fehler:
    - Fehler sollten sich nicht gegenseitig beeinflussen.
    - Ein Fehler sollte keinen anderen Fehler beeinflussen.
  - Konstante Varianz:
    - Fehler sollten über alle Werte der unabhängigen Variablen gleich variabel sein.
  - Normalverteilung der Fehler:
    - Fehler sollten einer Normalverteilung folgen.
    - Diese Annahme ist wichtig für viele statistische Tests und Modelle.
    - Nicht normal verteilte Fehler können die Analyse beeinträchtigen.

### Diagnosis.Fehleranalyse.Konstante Varianz

- Konstante Varianz:
  
  - Residuen allein reichen nicht aus, um sie zu überprüfen.
  - Prüfen, ob Variabilität der Residuen mit einem anderen Faktor zusammenhängt.

- Plot von $\hat{\varepsilon}$ gegen $\hat{y}$.
  
  - Bei konstanter Varianz: konsistentes, symmetrisches Muster (Homoskedastizität).
  - Bei inkonsistentem Muster: Heteroscedastizität.
  - Gekrümmtes Muster deutet auf Nichtlinearität im Modell hin.

```python=
plt.scatter(lmod.fittedvalues, lmod.resid)
plt.ylabel("Residuals"); plt.xlabel("Fitted values"); plt.axhline(0);
```

![](Figures/savings_9_0.png)

- Für genauere Prüfung: $\sqrt{|\hat{\varepsilon}|}$ gegen $\hat{y}$ plotten.

```python=
plt.scatter(lmod.fittedvalues, np.sqrt(abs(lmod.resid)))
plt.ylabel(r'$\sqrt{|\hat{ε}|}$'); plt.xlabel("Fitted values");
```

![](Figures/savings_11_0.png)

- Numerischer Test auf nicht konstante Varianz: Prüfen, ob $\sqrt{|\hat{\varepsilon}|}$ sich mit den angepassten Werten ändert.

```python=
ddf = pd.DataFrame({'x':lmod.fittedvalues, 'y':np.sqrt(abs(lmod.resid))})
dmod = smf.ols('y ~ x', ddf).fit()
dmod.sumary()
```

               coefs stderr tvalues pvalues
    Intercept  2.162  0.348    6.22  0.0000
    x         -0.061  0.035   -1.77  0.0838    
    n=50 p=2 Residual SD=0.634 R-squared=0.06

- Test auf nicht konstante Varianz:
  - Gewichtung und Anpassung der Freiheitsgrade erforderlich.
  - Keine signifikante Steigung, daher kein klares Problem.
  - Test erfasst nur lineare Trends und kann andere Arten übersehen.
- Präzision von Tests:
  - Tests erkennen möglicherweise nicht alle Probleme.
  - Grafische Techniken sind oft effektiver.
  - Plots liefern oft eine bessere Absicherung der Annahmen.
  - Grafische Ansätze werden bevorzugt, Tests zur weiteren Untersuchung genutzt.
- Residuen-gegen-Prädiktor-Plots für Spar-Daten betrachten:

```python=
plt.scatter(savings.pop75, lmod.resid)
plt.xlabel("%pop over 75"); plt.ylabel("Residuals"); plt.axhline(0);
```

![](Figures/savings_15_0.png)

```python=
plt.scatter(savings.pop15, lmod.resid)
plt.xlabel("%pop under 15"); plt.ylabel("Residuals"); plt.axhline(0);
```

![](Figures/savings_16_0.png)

- Zwei Gruppen sichtbar: unterentwickelten Ländern (pop15>35%) und entwickelten Ländern (pop15<35%).
- Varianzen in diesen Gruppen vergleichen und testen.
- Varianztest:
  - Zwei unabhängige Stichproben aus Normalverteilungen.
  - Teststatistik: Verhältnis der beiden Varianzen prüfen.

```python=
numres = lmod.resid[savings.pop15 > 35]
denres = lmod.resid[savings.pop15 < 35]
fstat = np.var(numres, ddof=1)/np.var(denres, ddof=1)
# ddof in the np.var is "Delta Degrees of Freedom."
# By default, ddof = 0, which means the divisor is N
# When ddof=1, the divisor becomes N-1. 
2*(1 - sp.stats.f.cdf(fstat, len(numres)-1, len(denres)-1))
```

    0.013575950424160377

- Modellanpassungen bei Problemen:
  
  - Nichtlinearität und nicht konstante Varianz: Variablen transformieren.
  - Nur nicht konstante Varianz: gewichtete kleinste Quadrate oder Antwortvariable transformieren.

- Transformation der Antwortvariable:
  
  - Ziel: konstante Varianz von $h(y)$.
  - Expansion:
    - $h(y) = h(E(y)) + [y-E(y)] \cdot h'(E(y)) + \cdots$
    - $var[h(y)] = h'(E(y))^2 \cdot var(y) + \cdots$
    - Für konstante Varianz: $h'(E(y)) \propto \sqrt{var(y)}$
    - Das führt zu $h(y) = \int \frac{dy}{\sqrt{var(y)}}$
    - Beispiele:
      - $var(y) \propto (E(y))^2$, dann $h(y) = \log(y)$
      - $var(y) \propto E(y)$, dann $h(y) = \sqrt{y}$
  - Alternative bei Werten y ≤ 0: $\log(y + \delta)$, aber komplizierte Interpretation.

### Diagnosis.Fehleranalyse.Normalität

- Tests und Konfidenzintervalle basieren auf normalverteilten Fehlern.
- Residuen können mit einem Q-Q-Plot auf Normalität geprüft werden.
- Im Q-Q-Plot werden die sortierten Residuen gegen $\Phi^{-1} (\frac{i}{n+1})$ für $i = 1, \ldots, n$ geplottet. Normale Residuen sollten der Linie folgen.
  - $\Phi^{-1}$ ist die Quantilfunktion der Standardnormalverteilung.
  - $\frac{i}{n+1}$ sind gleichmäßig zwischen 0 und 1 verteilte Wahrscheinlichkeiten für die sortierten Residuen.
  - Jeder sortierte Residuum wird dem entsprechenden theoretischen Quantilwert der Normalverteilung zugeordnet.
  - Bei Rechtsschiefe zeigt der Q-Q-Plot eine S-Form, bei Linksschiefe eine umgekehrte S-Form (links oberhalb und rechts unterhalb der Diagonalen).
  - Bei schweren Tails wölben sich die Punkte an beiden Enden nach außen (U-Form).
  - Extreme Fälle könnten auf langschwänzige Fehler (z.B. Cauchy-Verteilung) oder Ausreißer hinweisen.
  - Werden diese Beobachtungen entfernt und andere Punkte werden auffälliger, liegt wahrscheinlich ein langschwänziger Fehler vor.

```python=
sm.qqplot(lmod.resid, line="q");
```

![](Figures/savings_21_0.png)

- Histogramme und Boxplots sind wenig hilfreich.
- Histogramme zeigen oft keine Glockenkurve wegen der Gruppierung in Bins.
- Breite und Platzierung der Bins können das Ergebnis verfälschen.

```python=
plt.hist(lmod.resid); plt.xlabel("Residuals");
```

![](Figures/savings_23_0.png)

- Shapiro-Wilk-Test:
  
  - Formeller Test zur Überprüfung der Normalität.
  - Nullhypothese: Residuen sind normal. Wird die Nullhypothese nicht abgelehnt, können die Residuen normal sein.
  - Am besten zusammen mit einem Q-Q-Plot verwenden.
  - Große Datensätze: Kleine Abweichungen von der Normalität werden erkannt.
  - Große Stichproben: Methode der kleinsten Quadrate bleibt meist geeignet, da Nicht-Normalitätseffekte minimiert werden.
  - Kleine Stichproben: Formale Tests erkennen möglicherweise keine Nicht-Normalität gut und übersehen Abweichungen.

```python=
sp.stats.shapiro(lmod.resid)
```

    ShapiroResult(statistic=0.986984385973169, pvalue=0.8523961877568906)

- Fehler im statistischen Modell:
  
  - Nicht normalverteilte Fehler: Kleinste-Quadrate-Schätzungen sind nicht optimal, aber beste lineare unverzerrte Schätzungen.
  - Robuste Schätzer könnten besser sein.
  - Tests und Konfidenzintervalle: Annahme der Normalverteilung könnte ungenau sein, aber bei großen Stichproben durch das zentrale Grenzwerttheorem genauer.

- Umgang mit Nicht-Normalität:
  
  - Lösung hängt von der Art der Nicht-Normalität ab:
    - Kurzschwänzige Verteilung: Minimale Auswirkungen, kann ignoriert werden.
    - Schiefe Fehler: Transformation der Antwortvariablen könnte helfen.
    - Langschwänzige Fehler: Akzeptieren der Nicht-Normalität und Inferenz auf Basis einer anderen Verteilung.
  - Alternativen:
    - Resampling-Methoden wie Bootstrap oder Permutationstests.
    - Robuste Methoden: Weniger Gewicht auf Ausreißer, erfordern möglicherweise Resampling für genaue Inferenz.

- Modelländerungen und Diagnosetests:
  
  - Andere diagnostische Tests könnten Modelländerungen erfordern.
  - Nichtlinearität und nicht konstante Varianz zuerst beheben, um das Problem der nicht normalverteilten Fehler zu vermeiden.

### Diagnosis.Fehleranalyse.Korrigierte Fehler

- Durbin-Watson-Test:
  - Test zur Bewertung der Korrelation von Fehlern.
  - Nullhypothese: Fehler sind unkorreliert.
  - Nullverteilung folgt einer linearen Kombination von χ²-Verteilungen.
    - $DW = \frac{\sum_{i=2}^{n}{(\hat{\varepsilon}_i - \hat{\varepsilon}_{i-1})^2} }{\sum_{i=1}^{n}{\hat{\varepsilon}_i^2}}$
    - Bei Nullhypothese: Teststatistikwert von 2 erwartet. Werte unter 1 deuten auf ein Problem hin.
- Serielle Korrelation:
  - Kann durch fehlende Kovariaten im Modell entstehen.
  - Beispiel: Quadratische Beziehung zwischen Prädiktor und Antwort, aber nur linearer Term im Modell.
  - Diagnosen zeigen serielle Korrelation in Residuen.
  - Lösung: Fehlenden quadratischen Term im Modell hinzufügen.

```python=
sm.stats.stattools.durbin_watson(lmod.resid)
```

    1.9341

## Diagnosis.Ungewöhnliche Beobachtungen

- Ausreißer:
  - Passen nicht gut zum Modell.
- Einflussreiche Beobachtungen:
  - Beeinflussen die Anpassung des Modells erheblich.
  - Ein Punkt kann sowohl Ausreißer als auch einflussreiche Beobachtung sein.
- Hebelpunkte:
  - Extrem im Prädiktorraum.
  - Potenziell beeinflussen sie die Anpassung, müssen es aber nicht.
  - Identifikation ist wichtig, Umgang damit kann schwierig sein.

### Diagnosis.Ungewöhnliche Beobachtungen.Hebelwirkung

- Hebelwerte:
  
  - $H = X{(X^TX)}^{-1}X$. Hebelwerte sind $h_i = H_{ii}$.
  - Varianz der Fehler: $var(\hat{\varepsilon}_i) = \sigma^2 (1-h_i)$
    - Große  $h_i$ machen $var(\hat{\varepsilon}_i)$ klein. Der $\hat{y}_i$ ist näher am $y_i$.
    - Große Hebelwerte entstehen durch extreme Werte in den Prädiktorvariablen.
    - $h_i$ hängt mit der quadrierten Mahalanobis-Distanz zusammen.
      - Die Mahalanobis-Distanz berücksichtigt Korrelationen und Varianzen und ist geeignet für die Ausreißererkennung in mehrdimensionalen Daten (im Gegensatz zur euklidischen Distanz).
      - $D_{M,i}^2 = (n - 1)h_{ii}$
    - $h_i$ hängt nur von $X$ ab, nicht von $y$. Hebelwerte enthalten nur teilweise Informationen über einen Fall.
    - $\sum_i h_i = p$, der Durchschnittswert für $h_i$ ist p/n. Hebelwerte größer als 2p/n sollten genauer untersucht werden.

- Verteilung der Hebelwerte:
  
  - Nicht annehmbar. Halb-Normal-Plot zur Identifikation großer Hebelwerte.
    - Daten gegen positive Normalquantile plotten.
    - Halb-Normal-Plots für $|\hat{\varepsilon}|$ verwenden.
    - Daten sortieren: $h_{[1]} \leq \cdots \leq h_{[n]}$.
    - Positive Normalquantile berechnen: $u_i=\Phi^{-1}(\frac{n+i}{2n+1})$.
    - $h_{[i]}$ gegen $u_i$ plotten.
    - Gerade Linie nicht erwartet, da Hebelwerte nicht normalverteilt sind. Ausreißer als abweichende Punkte erkennbar.

```python=
diagv = lmod.get_influence()
hatv = pd.Series(diagv.hat_matrix_diag, savings.index)
hatv.sort_values().tail()
```

    South Rhodesia    0.160809
    Ireland           0.212236
    Japan             0.223310
    United States     0.333688
    Libya             0.531457
    dtype: float64

```python=
print(sum(hatv)) # 4.999
print(2*5/50) # p = 5, n = 50, 0.2
# Draw half-normal plot
n=50
ix = np.arange(1, n+1)
halfq = sp.stats.norm.ppf((n+ix)/(2*n+1)),
plt.scatter(halfq, np.sort(hatv))
plt.annotate("Libya",(2.1,0.53)); plt.annotate("USA", (1.9,0.33));
```

![](Figures/savings_33_0.png)

- Residuenvarianz:
  - $var(\hat{\varepsilon}_i) = \sigma^2 (1-h_i)$
  - Formel für standardisierte Residuen: $r_i=\frac{{\hat{\varepsilon}}_i}{\hat{\sigma}\sqrt{1-h_i}}$
- Standardisierte Residuen (Pearson-Residuen):
  - Bevorzugt in Residuenplots, da sie angepasste gleiche Varianz haben.
  - Bei korrekten Modellannahmen: $var(r_i) = 1$ und geringe Korrelation $corr(r_i,r_j)$.
- Standardisierung:
  - Korrigiert nicht-konstante Varianz nur bei konstanter Fehler-Varianz.
  - Bei Heteroskedastizität keine Korrektur durch Standardisierung möglich.

```python=
# lmod.resid_pearson: Divides only by σ^, no leverage component
# Internally Studentized Residuals: Obtainable via `get_influence()` function
rstandard = diagv.resid_studentized_internal
sm.qqplot(rstandard);
```

![](Figures/savings_35_0.png)

- Standardisierte Residuen:
  - Erwartung: Punkte folgen ungefähr der Linie $y = x$, wenn Normalität vorliegt.
  - Vorteil: Größe der Residuen leichter zu beurteilen.
    - Absolutwert von 2: Groß, aber nicht außergewöhnlich.
    - Absolutwert von 4: Sehr ungewöhnlich in der Standardnormalverteilung.
  - Empfehlung:
    - Einige Autoren empfehlen standardisierte Residuen in allen Diagnoseplots.
    - Oft ähnlich wie rohe Residuen, unterscheiden sich nur im Maßstab.
    - Unterschiede im Plot erkennbar bei ungewöhnlich großen Hebelwerten.

### Diagnosis.Ungewöhnliche Beobachtungen.Ausreißer

- Ausreißer:
  
  - Ein Ausreißer ist ein Datenpunkt, der sich von den anderen deutlich unterscheidet.
  - Ausreißertests helfen, außergewöhnliche Datenpunkte von großen Residuen zu unterscheiden.

- Einflussreiche Punkte:
  
  - Ein Punkt mit großem Residuum, der auch die Residuen anderer Punkte erhöht, ist sowohl ein Ausreißer als auch ein einflussreicher Punkt.
  - Solche Punkte sind wichtig zu identifizieren, da sie die Analyse stark beeinflussen können.

![](Figures/savings_1.png)

- Erkennung einflussreicher Punkte:
  
  - Ausschließen des Punktes (i) und Neuberechnung der Schätzungen: ${\hat{\beta}}_{(i)}$ und ${\hat{\sigma}}_{(i)}^2$.
  - Berechnung des neuen Wertes: ${\hat{y}}_{(i)}=x_i^\prime{\hat{\beta}}_{(i)}$
  - Große Differenz ${\hat{y}}_{(i)}-y_i$ zeigt, dass Fall $i$ ein Ausreißer ist.
  - Studentisierte Residuen zur Bewertung: $t_i=\frac{y_i-{\hat{y}}_{(i)}}{{\hat{\sigma}}_{(i)}\sqrt{1+x_i^\prime(X_{(i)}^\prime X_{(i)})^{-1}x_i}}$
  - Alternativ: $t_i=\frac{{\hat{\varepsilon}}_i}{{\hat{\sigma}}_{(i)}\sqrt{1-h_i}}=r_i\ (\frac{n-p-1}{n-p-r_i^2})^\frac{1}{2}\ \sim t_{n-p-1}$

- Testen auf Ausreißer:
  
  - $t_i \sim t_{n-p-1}$ ermöglicht die Berechnung eines p-Werts.
  - Bei Tests aller Fälle (n=100) bei 5% Signifikanzniveau erwartet man etwa fünf Ausreißer.
  - Anpassung des Testniveaus nötig, um zu viele Ausreißer zu vermeiden.

    - Bonferroni-Korrektur:

      - Anpassung des Signifikanzniveaus bei mehreren Tests: $\alpha/n$ für jeden Test.
      - Methode ist konservativ und findet weniger Ausreißer als das nominelle Konfidenzniveau.
      - Reduziert die Wahrscheinlichkeit von Fehlalarmen (falschen positiven Ergebnissen).

```python=
# In statsmodels, studentized residual is externally studentized residual and got by the get_influence()
stud = pd.Series(diagv.resid_studentized_external, savings.index)
(pd.Series.idxmax(abs(stud)), np.max(abs(stud)))
# ('Zambia', 2.854)
```

```python=
# Calculate the critical value: α=0.05, divided by 2 for 2 sided test, n = 50 -> n-p-1=44
abs(sp.stats.t.ppf(0.05/(2*50), 44))
# 3.5256
```

- Da 2,85 kleiner als 3,53 ist:
  - Schlussfolgerung: Sambia ist kein Ausreißer.
- Bei einfacher Regression:
  - Kritischer Minimalwert ist 3,51 bei $n = 23$.
  - p-Wert für Ausreißertests nur berechnen, wenn studentisierter Residuum absolut größer als 3,5 ist.

**Bermerkungen**

- Ausreißer
  - Zwei oder mehr Ausreißer nebeneinander können sich gegenseitig verstecken.
  - Ein Ausreißer in einem Modell ist möglicherweise kein Ausreißer in einem anderen Modell.
    - Bei Änderung oder Transformation der Variablen muss die Frage der Ausreißer neu untersucht werden.
- Größere Datensätze
  - Einzelne Ausreißer haben weniger Einfluss auf die Gesamtanpassung.
  - Wichtig, Ausreißer zu identifizieren, um wertvolle Erkenntnisse zu gewinnen.
    - Fokus auf Cluster von Ausreißern
      - Weniger wahrscheinlich zufällig, eher bedeutungsvolle Muster.
      - Identifizierung dieser Cluster kann schwierig sein.
- Fehlerverteilung
  - Fehlerverteilung ist nicht immer normal, größere Fehler können gelegentlich auftreten.
    - Beispiel: Aktienkurse ändern sich meist geringfügig, können aber auch signifikant und unerwartet schwanken.

**Aufgaben bei Ausreißern**

- Datenfehler prüfen:
  - Bei sicherem Fehler, Punkt verwerfen.
- Wissenschaftliche Entdeckungen:
  - Ausreißer können neue Erkenntnisse bringen, z.B. bei Kreditkartenbetrug.
- Punkt ausschließen:
  - Später bei Modelländerung wieder einbeziehen.
- Natürliche Ausreißer:
  - Robuste Regression verwenden.
  - Ausreißer nicht einfach entfernen, sondern Vorhersagemethoden anpassen.
- Automatisches Ausschließen:
  - Gefährlich, sollte vermieden werden.

### Diagnosis.Ungewöhnliche Beobachtungen.Einflussreiche Beobachtungen

- Einflussreicher Punkt:
  - Entfernen würde die Modellanpassung stark ändern.
  - Kann ein Ausreißer oder hoher Hebel sein.
- Einflussmessung:
  - Subskript (i) zeigt Anpassung ohne Fall i.
  - Änderung der Anpassung messen: ${X}^\prime(\hat{\beta} - \hat{\beta}_{(i)}) = \hat{y} - \hat{y}_{(i)}$
  - Änderung im Koeffizienten $\hat{\beta} - \hat{\beta}_{(i)}$ betrachten.
  - Cook-Statistik:
    - Beliebte Einflussdiagnostik.
    - Reduziert Information auf einen Wert pro Fall.
    - Formel: $D_i = \frac{(\hat{y} - \hat{y}_{(i)})^\prime (\hat{y} - \hat{y}_{(i)})}{p \hat{\sigma}^2} = \frac{1}{p} r_i^2 \frac{h_i}{1 - h_i}$
      - $r_i^2$: Residualeffekt.
      - $\frac{h_i}{1 - h_i}$: Hebel.
      - Kombination führt zu Einfluss.
    - Halb-Normal-Plot von $D_i$ zur Identifikation einflussreicher Beobachtungen.

```python=
cooks = pd.Series(diagv.cooks_distance[0], savings.index)
n=50
ix = np.arange(1,n+1)
halfq = sp.stats.norm.ppf((n+ix)/(2*n+1)),
plt.scatter(halfq, np.sort(cooks));
```

![](Figures/savings_44_0.png)

```python=
# The largest five values of Cook statistics
cooks.sort_values().iloc[-5:]
```

    Philippines    0.045
    Ireland        0.054
    Zambia         0.097
    Japan          0.143
    Libya          0.268
    dtype: float64

```python=
# Exclude Libya and see how the fit changes
lmodi = smf.ols('sr ~ pop15 + pop75 + dpi + ddpi', savings[cooks < 0.2]).fit()
pd.DataFrame({'with':lmod.params, 'without':lmodi.params})
```

<table class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>with</th>
      <th>without</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Intercept</th>
      <td>28.566087</td>
      <td>24.524046</td>
    </tr>
    <tr>
      <th>pop15</th>
      <td>-0.461193</td>
      <td>-0.391440</td>
    </tr>
    <tr>
      <th>pop75</th>
      <td>-1.691498</td>
      <td>-1.280867</td>
    </tr>
    <tr>
      <th>dpi</th>
      <td>-0.000337</td>
      <td>-0.000319</td>
    </tr>
    <tr>
      <th>ddpi</th>
      <td>0.409695</td>
      <td>0.610279</td>
    </tr>
  </tbody>
</table>

- Koeffizient für ddpi änderte sich um ca. 50%.
- Sensitivität: Schätzungen sollen nicht so empfindlich auf ein Land reagieren.

```python=
# Extract DFBETA for pop15. DFBETA stands for "Difference in Betas."
# Nimm die erste Beobachtung, berechne den geschätzten Koeffizienten.
# Berechne die Differenz der Koeffizienten mit und ohne eine Beobachtung.
# Notiere den Unterschied für die Variable pop15.
# Wiederhole das für alle Beob, so erhält man einen Vektor mit 50 Werten.
# diagv.dfbetas zeigt eine Matrix, wobei jede Spalte einer Variable entspricht.
# Jeder Wert zeigt, wie stark sich der pop15-Koeffizient beim Entfernen einer Beob.  
# Große DFBETA-Werte bedeuten einen starken Einfluss.
p15d = diagv.dfbetas[:, 1]
```

```python=
# Plot dfbetas for pop15
plt.scatter(np.arange(1,51), p15d)
plt.axhline(0)
# Annotate Japan and Libya
ix = 22; plt.annotate(savings.index[ix], (ix, p15d[ix]))
ix = 48; plt.annotate(savings.index[ix], (ix, p15d[ix]));
```

![](Figures/savings_49_0.png)

- Änderung des zweiten Parameters (~pop15) bei Ausschluss eines Falls:
  - Japan fällt besonders auf.
  - Prozess für andere Variablen wiederholen.
  - Wirkung des Ausschlusses von Japan untersuchen.

```python=
lmodj = smf.ols('sr ~ pop15+pop75+dpi+ddpi',savings.drop(['Japan'])).fit()
lmodj.sumary()
```

               coefs stderr tvalues pvalues
    Intercept 23.940  7.784    3.08  0.0036
    pop15     -0.368  0.154   -2.39  0.0210
    pop75     -0.974  1.155   -0.84  0.4040
    dpi       -0.000  0.001   -0.51  0.6112
    ddpi       0.335  0.198    1.69  0.0987
    n=49 p=5 Residual SD=3.738 R-squared=0.28

- Vergleich mit vollständiger Datenanpassung:
  - Mehrere Änderungen sichtbar.
  - ddpi-Term nicht mehr signifikant.
  - R²-Wert stark gesunken.

## Diagnosis.Modellstruktur

- Plots zur Diagnose von Regressionsmodellen
  - Plots der Residuen $\hat{\varepsilon}$ gegen:
    - Vorhergesagte Werte $\hat{y}$
    - Prädiktorvariablen $x_i$
  - Nutzen dieser Plots:
    - Überprüfung der Annahmen über Fehler im Modell
    - Hinweise auf mögliche Transformationen der Variablen zur Verbesserung der Modellstruktur
- Plots von $y$ gegen jedes $x_i$
  - Die Beziehung zwischen einem Prädiktor und der Antwortvariable kann von anderen Prädiktoren beeinflusst werden.
- **Partielle Regressionsdiagramme (Added Variable Plots)**
  - Ziel: Den Effekt eines Prädiktors ($x_i$) auf die Antwortvariable ($y$) isolieren.
  - Vorgehen:
    - Zuerst wird $y$ mit allen Prädiktoren außer $x_i$ vorhergesagt, um die Residuen ($\hat{\delta}$) zu erhalten.
    - Dann wird $x_i$ auf alle anderen Prädiktoren regrediert, um die Residuen ($\hat{\gamma}$) zu erhalten.
    - Das partielle Regressionsdiagramm entsteht durch das Plotten von $\hat{\delta}$ gegen $\hat{\gamma}$.
      - Damit lassen sich nichtlineare Beziehungen, Ausreißer und einflussreiche Beobachtungen erkennen.
  - Ein partielles Regressionsdiagramm zeigt den individuellen Beitrag eines Prädiktors zur Antwortvariable, nachdem die Effekte der anderen Prädiktoren herausgerechnet wurden.
  - So lässt sich der Einfluss eines einzelnen Prädiktors in einer multiplen Regression anschaulich darstellen.

```python=
# Examine the variable pop15
d = smf.ols('sr ~ pop75 + dpi + ddpi', savings).fit().resid
m = smf.ols('pop15 ~ pop75 + dpi + ddpi', savings).fit().resid
plt.scatter(m, d)
plt.xlabel("pop15 residuals"); plt.ylabel("sr residuals")
# Line from the point (-10, -10*β_pop15) to the point (8, 8*β_pop15)
plt.plot([-10, 8], [-10*lmod.params.iloc[1], 8*lmod.params.iloc[1]]);
```

![](Figures/savings_54_0.png)

- Mit np.polyfit(m, d, deg=1) wird eine lineare Regression auf die Datenpunkte (m, d) durchgeführt.
  - m: Array der x-Werte
  - d: Array der y-Werte
  - deg=1: Grad des Polynoms (hier eine Gerade)
- Die Funktion gibt die Koeffizienten des Polynoms zurück, beginnend mit dem höchsten Grad.
- Für eine lineare Anpassung liefert sie zwei Werte: Steigung und Achsenabschnitt.

```python=
np.polyfit(m,d,deg=1) # (array([-4.612e-01, -3.809e-13])
lmod.params.iat[1] # -0.4611
```

- **Partielle Residuenplots** sind eine Alternative zu Added Variable Plots.
  - $y-\sum_{j\neq i}{x_j{\hat{\beta}}_j}=\hat{y}+\hat{\varepsilon}-\sum_{j\neq i}{x_j{\hat{\beta}}_j}=x_i{\hat{\beta}}_i+\hat{\varepsilon}$
  - Der partielle Residuenplot ist $x_i{\hat{\beta}}_i+\hat{\varepsilon}$ gegen $x_i$.
  - Die Steigung des Plots ist ${\hat{\beta}}_i$.
  - Partielle Residuenplots eignen sich besser zur Erkennung von Nichtlinearitäten, Added Variable Plots besser zur Erkennung von Ausreißern/Einflüssen.

```python
pr = lmod.resid + savings.pop15*lmod.params.iat[1]
plt.scatter(savings.pop15, pr)
plt.xlabel("pop15"); plt.ylabel("partial residuals")
# Line from the point (20, 20*β_pop15) to the point (50, 50*β_pop15)
plt.plot([20,50], [20*lmod.params.iat[1], 50*lmod.params.iat[1]]);
```

![](Figures/savings_58_0.png)

- Das Diagramm zeigt zwei Gruppen.
- Dies deutet auf unterschiedliche Beziehungen in diesen Gruppen hin.

```python=
smf.ols('sr ~ pop15 + pop75 + dpi + ddpi', savings[savings.pop15 > 35]).fit().sumary()
```

               coefs stderr tvalues pvalues
    Intercept -2.434 21.155   -0.12  0.9097
    pop15      0.274  0.439    0.62  0.5408
    pop75     -3.548  3.033   -1.17  0.2573
    dpi        0.000  0.005    0.08  0.9339
    ddpi       0.395  0.290    1.36  0.1896
    n=23 p=5 Residual SD=4.454 R-squared=0.16

```python=
smf.ols('sr ~ pop15 + pop75 + dpi + ddpi', savings[savings.pop15 < 35]).fit().sumary()
```

               coefs stderr tvalues pvalues
    Intercept 23.962  8.084    2.96  0.0072
    pop15     -0.386  0.195   -1.98  0.0609
    pop75     -1.328  0.926   -1.43  0.1657
    dpi       -0.000  0.001   -0.63  0.5326
    ddpi       0.884  0.295    2.99  0.0067
    
    n=27 p=5 Residual SD=2.772 R-squared=0.51

- Unterentwickelte Länder (pop15 > 35%): kein erkennbarer Zusammenhang.
- Entwickelte Länder (pop15 < 35%): klarer Zusammenhang, besonders mit „Wachstum“ und „pop15“.
    - Der Effekt ist weniger deutlich, weil der Wertebereich eingeschränkt wurde.
- Bedeutung von grafischer Analyse
  - Grafiken zeigen Zusammenhänge, die Zahlen allein oft nicht erkennen lassen.
  - Farben, Symbole, Größen oder mehrere Teilplots helfen, weitere Muster zu erkennen.

```python=
savings.head(3)
```

<table class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>sr</th>
      <th>pop15</th>
      <th>pop75</th>
      <th>dpi</th>
      <th>ddpi</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Australia</th>
      <td>11.43</td>
      <td>29.35</td>
      <td>2.87</td>
      <td>2329.68</td>
      <td>2.87</td>
    </tr>
    <tr>
      <th>Austria</th>
      <td>12.07</td>
      <td>23.32</td>
      <td>4.41</td>
      <td>1507.99</td>
      <td>3.93</td>
    </tr>
    <tr>
      <th>Belgium</th>
      <td>13.17</td>
      <td>23.80</td>
      <td>4.43</td>
      <td>2108.47</td>
      <td>3.82</td>
    </tr>
  </tbody>
</table>

```python=
savings['age'] = np.where(savings.pop15 > 35, 'young', 'old')
sns.lmplot(x='ddpi', y='sr', data=savings, hue='age', facet_kws={"legend_out": False});
```

![](Figures/savings_64_0.png)

- sns.lmplot: Lineare Regression mit Seaborn
  - x='ddpi': x-Achse ist ‘ddpi’
  - y='sr': y-Achse ist ‘sr’
  - data=savings: DataFrame ‘savings’
  - hue='age': Punkte nach ‘age’ gefärbt
  - facet_kws={"legend_out": False}: Legende im Plot

```python=
sns.lmplot(x='ddpi', y='sr', data=savings, col='age');
```

![](Figures/savings_66_0.png)

- Möglichkeiten zur Unterscheidung der Statusvariable nach Bevölkerungsanteil unter 15
  - Zwei Plotvarianten; die zweite ist effektiver.
  - Regressionslinie mit 95%-Konfidenzintervallen zeigt Gruppenunterschiede.
- Höherdimensionale Plots
  - Helfen, verborgene Strukturen zu erkennen.
  - Sind meist interaktiv und müssen ausprobiert werden.
  - 3D-Plots: durch Farbe, Punktgröße, Rotation möglich.
  - Mehrere Plots lassen sich verlinken, um Punkte hervorzuheben.
  - Praktisch oft schwer bedienbar und nicht gut druckbar.

## Diagnosis.Diskussion

- Sortierung nach Wichtigkeit der Effekte:
  - Modellform
    - Korrekte Definition der Beziehungen zwischen den Variablen ist am wichtigsten.
    - Fehlerhafte Modellform führt zu falschen Vorhersagen und irreführenden Interpretationen.
  - Fehlerabhängigkeit
    - Starke Abhängigkeit der Fehler verringert die Informationsmenge der Daten.
    - Kann dazu führen, dass unnötige systematische Komponenten ins Modell aufgenommen werden.
    - Schwer zu erkennen, meist nur bei Zeitreihendaten auffällig.
  - Nichtkonstante Varianz
    - Führt zu ungenauen Schlussfolgerungen, besonders bei der Unsicherheitsschätzung.
    - Bei nur leichter Verletzung ist die Auswirkung meist gering.
  - Normalverteilung
    - Am wenigsten wichtig.
    - Bei großen Datensätzen sind die Ergebnisse meist robust, auch ohne Normalverteilung (zentraler Grenzwertsatz).
    - Kritischer bei kleinen Stichproben oder sehr ungewöhnlichen Fehlerverteilungen.

# Skalierungsänderungen

- Warum Variablen skalieren?
  - Um Einheiten zu konvertieren oder extreme Werte (sehr groß/klein) anzupassen
  - Verbesserung der numerischen Stabilität und Vermeidung von Berechnungsfehlern
- Auswirkungen der Reskalierung
  - Statistische Tests wie t-Tests, F-Tests, Varianz ($\sigma^2$) und $R^2$ bleiben unverändert
    - Gilt sowohl für Reskalierung von $x_i$ als auch von $y$
  - Reskalierung von $x_i$ als $\frac{x_i + a}{b}$ skaliert  $\hat{\beta}_i$
  - Reskalierung von $y$ skaliert $\hat{\beta}$ und $\hat{\sigma}$
- Vorteil der Standardisierung
  - Alle Variablen haben die gleiche Skala, was Vergleiche erleichtert
  - Regressionskoeffizienten liegen zwischen -1 und 1 und zeigen die Stärke und Richtung der Beziehungen (Teilkorrelation)
  - Numerische Stabilität, da zentrierte Variablen Berechnungsprobleme vermeiden
  - Einfache Interpretation: Koeffizienten geben den Effekt einer Standardabweichung des Prädiktors auf die Antwortvariable an

```python=
# Reload the data again
savings_sclDF = faraway.datasets.savings.load()
lmod = smf.ols('sr ~pop15+pop75+dpi+ddpi',savings_sclDF).fit()
lmod.sumary()
```

               coefs stderr tvalues pvalues
    Intercept 28.566  7.355    3.88  0.0003
    pop15     -0.461  0.145   -3.19  0.0026
    pop75     -1.691  1.084   -1.56  0.1255
    dpi       -0.000  0.001   -0.36  0.7192
    ddpi       0.410  0.196    2.09  0.0425
    n=50 p=5 Residual SD=3.803 R-squared=0.34

```python=
# Change the scale
lmod = smf.ols('sr ~pop15+pop75+ I(dpi/1000) +ddpi', savings_sclDF).fit()
lmod.sumary()
```

                   coefs stderr tvalues pvalues
    Intercept     28.566  7.355    3.88  0.0003
    pop15         -0.461  0.145   -3.19  0.0026
    pop75         -1.691  1.084   -1.56  0.1255
    I(dpi / 1000) -0.337  0.931   -0.36  0.7192
    ddpi           0.410  0.196    2.09  0.0425
    n=50 p=5 Residual SD=3.803 R-squared=0.34

```python=
# Standardization
scsav = savings_sclDF.apply(sp.stats.zscore)
lmod = smf.ols('sr ~ pop15 + pop75 + dpi + ddpi', scsav).fit()
lmod.sumary()
```

               coefs stderr tvalues pvalues
    Intercept  0.000  0.121    0.00  1.0000
    pop15     -0.942  0.295   -3.19  0.0026
    pop75     -0.487  0.312   -1.56  0.1255
    dpi       -0.075  0.206   -0.36  0.7192
    ddpi       0.262  0.126    2.09  0.0425
    
    n=50 p=5 Residual SD=0.857 R-squared=0.34

- Wenn Prädiktoren auf ähnlichen Skalen liegen:
  - Plot der geschätzten Koeffizienten mit Konfidenzintervallen ist hilfreich.

```python=
edf = pd.concat([lmod.params, lmod.conf_int()],axis=1).iloc[1:,]
edf.columns = ['estimate','lb','ub']
npreds = edf.shape[0]
fig, ax = plt.subplots()
ax.scatter(edf.estimate,np.arange(npreds))
for i in range(npreds):
    ax.plot([edf.lb.iat[i], edf.ub.iat[i]], [i, i])
ax.set_yticks(np.arange(npreds))
ax.set_yticklabels(edf.index)
ax.axvline(0);
```

![](Figures/savings_74_0.png)

- Unterschiedliche Skalierung für binäre und kontinuierliche Prädiktoren
  - Binäre Prädiktoren
    - Bei Kodierung mit 0 und 1 (gleiche Wahrscheinlichkeit) beträgt die Standardabweichung 0,5.
    - Alternativ kann eine -1/+1-Kodierung verwendet werden; dann ist die Standardabweichung 1.
  - Kontinuierliche Prädiktoren
    - Um die Vergleichbarkeit mit binären Prädiktoren (0/1-Skalierung) herzustellen, sollte man kontinuierliche Variablen durch zwei Standardabweichungen teilen.(*)
-(*): Vergleichbarkeit von Prädiktoren
  - Binär (0/1, gleich verteilt)
    - Sprung von 0 auf 1 ergibt eine Differenz von 1.
    - Standardabweichung ist 0,5.
    - Der Sprung von 0 auf 1 entspricht 2 Standardabweichungen.
  - Kontinuierlich
    - Teilt man eine kontinuierliche Variable durch 2 Standardabweichungen, entspricht eine Änderung von −1 auf +1 bei der kontinuierlichen Variable dem Unterschied zwischen 0 und 1 beim binären Prädiktor.
    - So sind die Effekte direkt vergleichbar.

```python=
# Create a age column
savings_sclDF['age'] = np.where(savings_sclDF.pop15 > 35, 0, 1)
# younger countries are coded as 0 and older countries as 1
savings_sclDF['dpis'] = sp.stats.zscore(savings_sclDF.dpi)/2
savings_sclDF['ddpis'] = sp.stats.zscore(savings_sclDF.ddpi)/2
smf.ols('sr ~ age + dpis + ddpis', savings_sclDF).fit().sumary()
```

               coefs stderr tvalues pvalues
    Intercept  6.818  1.011    6.75  0.0000
    age        5.284  1.585    3.33  0.0017
    dpis      -1.549  1.593   -0.97  0.3361
    ddpis      2.443  1.097    2.23  0.0309
    n=50 p=4 Residual SD=3.800 R-squared=0.32

- Interpretation der Koeffizienten:
  - Ältere Länder (age = 1) haben eine um 5,28 % höhere Sparquote als jüngere (age = 0).
  - Ein Anstieg von ddpi um zwei Standardabweichungen (also eine Einheit auf der Skala) erhöht die Sparquote um 2,44 %.
  - Ein Anstieg von dpi um zwei Standardabweichungen senkt die Sparquote um 1,55 % (nicht signifikant).
  
# Transformation

- Log-Transformation der Antwortvariable:
  
  - $\log{\hat{y}}={\hat{\beta}}_0+{\hat{\beta}}_1x_1+\ldots+{\hat{\beta}}_px_p$
  - $\hat{y}=e^{{\hat{\beta}}_0}e^{{\hat{\beta}}_1x_1}\ldots e^{{\hat{\beta}}_px_p}$
  - Erhöhung von $x_1$ um eins multipliziert die vorhergesagte Antwort um $e^{\hat{\beta}_1}$.
  - Bei kleinen $x$ gilt $\log(1 + x) \approx x$.
  - Beispiel: $\beta_1 = 0.09$ bedeutet, Erhöhung von $x_1$ um eins erhöht $\log(y)$ um 0.09, oder y um 9%.

- Box-Cox-Methode transformiert positive Antworten $y$ zu $g_\lambda(y)$:
  
  - $g_\lambda(y) = \begin{cases} \frac{y^\lambda - 1}{\lambda}, & \lambda \neq 0 \\ \log{y}, & \lambda = 0 \end{cases}$
  - Für $y > 0$ ändert sich $g_\lambda(y)$ gleichmäßig mit $\lambda$.
  - Optimales $\lambda$ wird durch Maximum-Likelihood bestimmt, bei dem Modellfehler normalverteilt sind:
    - $L(\lambda) = -\frac{n}{2} \log{(\frac{RSS_\lambda}{n})} + (\lambda - 1) \sum \log{y_i}$
      - RSS$_\lambda$ ist die Residuenquadratsumme für $g_\lambda(y)$.
      - Beste $\hat{\lambda}$ wird numerisch maximiert.
  - Für Vorhersagen kann $y^\lambda$ als Antwort verwendet werden.
  - $\frac{y^\lambda - 1}{\lambda}$ sorgt für glatten Übergang zu $\log{y}$ bei $\lambda \to 0$.
  - Für Verständlichkeit kann $\lambda$ gerundet werden, z.B. $\hat{\lambda} = 0.46$ zu $\lambda = 0.5$ für $\sqrt{y}$.

- Transformation der Antwortvariable sollte nur bei Notwendigkeit erfolgen.

- Überprüfung durch Konstruktion eines Konfidenzintervalls für $\lambda$:
  
  - $100(1-\alpha)\%$ Konfidenzintervall für $\lambda$:
    - $\{\lambda:L(\lambda)>L(\hat{\lambda})-1/2\chi_{1,1-\alpha}^2\}$
  - Intervall basiert auf Likelihood-Ratio-Test für $H_0: \lambda = \lambda_0$.
  - Teststatistik: $2(L(\hat{\lambda}) - L(\lambda_0))$ folgt $\chi_1^2$-Verteilung.
  - Konfidenzintervall zeigt, wie stark $\lambda$ für bessere Interpretierbarkeit gerundet werden kann.

```python=
# Extracts the exogenous variables matrix X
X = lmod.model.wexog
# Setting Up Variables for Box-Cox Transformation
n = savings.shape[0] # number of observations in the dataset
sumlogy = np.sum(np.log(savings.sr)) # sum of log of sr
lam = np.linspace(0.5, 1.5, 100) # array of 100 lambda values ranging from 0.5 to 1.5
llk = np.empty(100) # empty array to store log-likelihood values
# Calculating Log-Likelihood for Each Lambda
for i in range(0, 100):
    lmod = sm.OLS(sp.stats.boxcox(savings.sr, lam[i]), X).fit()
    llk[i] = -(n/2)*np.log(lmod.ssr/n) + (lam[i]-1)*sumlogy
# Plotting the Log-Likelihood Values
fig, ax = plt.subplots()
ax.plot(lam, llk)
ax.set_xlabel('$\lambda$'); ax.set_ylabel('log likelihood')
# Highlighting the Maximum Log-Likelihood
maxi = llk.argmax()
ax.vlines(lam[maxi], ymin=min(llk), ymax=max(llk), linestyle='dashed')
# Calculates the cutoff for the 95% confidence interval
cicut = max(llk) - sp.stats.chi2.ppf(0.95, 1) / 2
# Identifies the range of lambda values within this confidence interval
rlam = lam[llk > cicut]
# Draws horizontal and vertical dashed lines to indicate the confidence interval on the plot.
ax.hlines(cicut, xmin=rlam[0], xmax=rlam[-1], linestyle='dashed')
ax.vlines([rlam[0], rlam[-1]], ymin=min(llk), ymax=cicut, linestyle='dashed');
```

![](Figures/savings_82_0.png)

- Berechnungsbereich: [0.5, 1.5]
- Ein größerer Bereich könnte notwendig sein, um das Maximum und das Konfidenzintervall zu erfassen.
- Bereich basierte auf vorherigen Tests.
- Konfidenzintervall für $\lambda$: ungefähr 0.6 bis 1.4, daher keine starke Notwendigkeit zur Transformation.

# Knickpunkt-Regression

- Manchmal brauchen Daten verschiedene lineare Regressionsmodelle.
- Beispiel: Spardaten zeigen zwei Gruppen.
- Mit dem Prädiktor ‘pop15’ können wir zwei Modelle erstellen:
  - ‘pop15’ über 35%
  - ‘pop15’ unter 35%

```python=
# Fit linear models for two segments of the data
lmod1 = smf.ols('sr ~ pop15', savings[savings.pop15 < 35]).fit()
lmod2 = smf.ols('sr ~ pop15', savings[savings.pop15 > 35]).fit()
# Create scatter plot of the data
plt.scatter(savings.pop15, savings.sr)
plt.xlabel('Population under 15'); plt.ylabel('Savings rate')
plt.axvline(35, linestyle='dashed')
# Plot the first segment of the regression line
plt.plot([20, 35], [lmod1.params.iat[0] + lmod1.params.iat[1] * 20,
                    lmod1.params.iat[0] + lmod1.params.iat[1] * 35], 'k-')
# Plot the second segment of the regression line
plt.plot([35, 48], [lmod2.params.iat[0] + lmod2.params.iat[1] * 35,
                    lmod2.params.iat[0] + lmod2.params.iat[1] * 48], 'k-');
```

![](Figures/savings_86_0.png)

- Problem bei separaten Modellen: keine Verbindung am Trennpunkt.

- Für einen fließenden Übergang verwenden wir Knickpunkt-Regression:
  
  - Zwei neue Variablen:
    - $B_l(x) = \begin{cases} 
        c - x & \text{wenn } x < c \\ 
        0 & \text{ansonsten} 
      \end{cases}$    
    - $B_r(x) = \begin{cases} 
        x - c & \text{wenn } x > c \\ 
        0 & \text{ansonsten} 
      \end{cases}$

- $B_l$ und $B_r$ werden oft als Hockeyschläger-Funktionen bezeichnet.
- $B_l$ und $B_r$ bilden eine erste Ordnung Spline-Basis mit einem Knoten bei ‘c’, was eine bessere Anpassung ermöglicht.
- Das Modell $y=\beta_0+\beta_1B_l(x)+\beta_2B_r(x)+\varepsilon$ wird durch reguläre Regressionsmethoden angepasst.
- Die beiden Segmente treffen sich bei Punkt ‘c’, was Kontinuität gewährleistet.
- Im Gegensatz zum vorherigen Ansatz verwendet dieses Modell nur drei Parameter, indem es die Glätte bei Punkt ‘c’ sicherstellt.
- Der Achsenabschnitt dieses Modells ist der Wert des Ergebnisses, wo die beiden Teile sich treffen.

```python=
# Define helper functions for piecewise linear terms
def lhs(x, c):
    return np.where(x < c, c - x, 0)
def rhs(x, c):
    return np.where(x < c, 0, x - c)
# Fit the linear model using the piecewise terms
lmod = smf.ols('sr ~ lhs(pop15,35) + rhs(pop15,35)', savings).fit()
# Generate predictions
x = np.arange(20, 49) # array of values from 20 to 48
# Computes predicted savings rate (py) using the fitted model parameters
# and the lhs and rhs functions.
py = lmod.params.iat[0] + lmod.params.iat[1] * lhs(x, 35) + lmod.params.iat[2] * rhs(x, 35)
# Plot the piecewise linear regression line
plt.plot(x, py, linestyle='dotted')
# Create scatter plot of the data
plt.scatter(savings.pop15, savings.sr)
plt.xlabel('Population under 15'); plt.ylabel('Savings rate')
# Add a vertical dashed line at pop15 = 35
plt.axvline(35, linestyle='dashed');
```

![](Figures/savings_88_0.png)

- Welches Modell ist besser?
  - Bei hohen ‘pop15’-Werten ändert sich die Steigung durch Glättung.
  - Aufgrund der Unterschiede zwischen den Gruppen und wenigen Ländern in der Mitte ist Glättung vielleicht nicht nötig.
- Flexibleres Modell durch Hinzufügen weiterer Knickpunkte (Knotenpunkte).
