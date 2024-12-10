## Lernhilfe zur Linearen Regression mit dem Galapagos-Datensatz

### 1. Beschreibung der Daten

Der Datensatz enthält Informationen über 30 Galapagos-Inseln.  Es gibt sechs Variablen:

*   **Species:** Die Anzahl der auf der Insel gefundenen Arten.
*   **Area:** Die Fläche der Insel (km²).
*   **Elevation:** Die höchste Erhebung der Insel (m).
*   **Nearest:** Die Entfernung zur nächsten Insel (km).
*   **Scruz:** Die Entfernung zur Insel Santa Cruz (km).
*   **Adjacent:** Die Fläche der benachbarten Insel (km²).

### 2. Laden von Paketen und Daten

```python
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import scipy as sp 
import statsmodels.api as sm 
import statsmodels.formula.api as smf 
import seaborn as sns 
import faraway.utils 
np.set_printoptions(suppress=True)

import faraway.datasets.galapagos 
galapagos = faraway.datasets.galapagos.load() 
galapagos.head(3)
print("Area: km², Elevation: m, Nearest: km, Scruz: km, Adjacent: km²") 
galapagos.describe().iloc[1:,:].round(3)
```

### 3. Lineare Regression

```python
lmod = smf.ols(formula='Species ~ Area + Elevation + Nearest + Scruz  + Adjacent', data=galapagos).fit() 
lmod.summary()
### Kürzere Version der Zusammenfassung aus dem Paket faraway
lmod.sumary()
```

#### 3.1. Extrahieren der Regressionsgrößen

##### 3.1.1. Grundlagen

*   Schätzungen der Koeffizienten: `lmod.params`
*   Beta-Standardfehler: `lmod.bse`
*   p-Werte: `lmod.pvalues`
*   T-Statistiken für die Koeffizienten: `lmod.tvalues`

##### 3.1.2. F-Werte

*   F-Statistik und ihr p-Wert: `lmod.fvalue, lmod.f_pvalue`

##### 3.1.3. Konfidenzintervalle

*   Konfidenzintervalle der Koeffizienten: `lmod.conf_int()`
*   Ob die t-Verteilung für die Inferenz verwendet werden soll: `lmod.use_t`
    *   Wahr: Die t-Verteilung wird für die Inferenz verwendet.
    *   Falsch: Die Normalverteilung wird für die Inferenz verwendet.

##### 3.1.4. Güte der Anpassung

*   R-Quadrat: `lmod.rsquared`
*   Angepasstes R-Quadrat: `lmod.rsquared_adj`
*   Akaike Information Criterion AIC: `lmod.aic`
*   Bayesian Information Criterion BIC: `lmod.bic`

##### 3.1.5. Quadratsummen

| Quelle        | Freiheitsgrade | Quadratsummen | Mittleres Quadrat     |
| ------------- | --------------- | ------------- | --------------------- |
| Regression    | (p-1)          | ESS           | $\frac{\text{ESS}}{p-1}$ |
| Residuen      | (n-p)          | RSS           | $\frac{\text{RSS}}{n-p}$ |
| Gesamt        | (n-1)          | TSS           | $\frac{\text{TSS}}{n-1}$ |

*   Residuenquadratsumme RSS: `lmod.ssr`
*   Quadratsumme der Residuen SSR: `lmod.ssr`
*   Erklärte Quadratsumme ESS: `lmod.ess`
*   Die zentrierte TSS = Summe(beobachteter Wert - Mittelwert(beobachtete Variable))²: `lmod.centered_tss, lmod.ssr + lmod.ess`
*   Die nicht zentrierte TSS = Summe(beobachteter Wert²): `lmod.uncentered_tss, sum(lmod.model.endog**2)`
*   Freiheitsgrade des Modells: `lmod.df_model`
*   Freiheitsgrade der Residuen: `lmod.df_resid`
*   Anzahl der Beobachtungen: `lmod.nobs`
*   Mittlerer quadratischer Fehler des Modells: `lmod.mse_model, lmod.ess / lmod.df_model`
*   Mittlerer quadratischer Fehler der Residuen: `lmod.mse_resid, lmod.scale`
*   Gesamter mittlerer quadratischer Fehler: `lmod.mse_total, lmod.centered_tss / (lmod.df_model + lmod.df_resid)`

##### 3.1.6. Angepasste Werte und Residuen

*   Angepasste Werte: `lmod.fittedvalues.head(3)`
*   Residuen: `lmod.resid.head(3)`
*   Selbstausgerechnete Residuen: `(lmod.model.endog - lmod.fittedvalues).head(3)`
*   Pearson-Residuen: `lmod.resid_pearson.round(3)`
    *   Pearson-Residuen = Rohresiduen / ihre Standardabweichung: `( lmod.resid / np.sqrt(lmod.mse_resid) ).head(3)`

##### 3.1.7. Kovarianzmatrix

*   Kovarianzmatrix der Koeffizienten: `lmod.cov_params()`
*   Typ der Kovarianzmatrix: `lmod.cov_type`
*   Normalisierte Kovarianzmatrix = Kovarianzmatrix / Residuenvarianz: `lmod.normalized_cov_params`
*   HC0 ist die grundlegende heteroskedastizitätskonsistente Kovarianzmatrix. Sie wendet keine zusätzlichen Skalierungsfaktoren an: `lmod.cov_HC0.round(3); lmod.HC0_se`
*   HC1 wendet eine Freiheitsgradkorrektur auf die HC0-Matrix an. Sie skaliert die Residuen mit n/(n-k), wobei n die Anzahl der Beobachtungen und k die Anzahl der Parameter ist: `lmod.cov_HC1.round(3); lmod.HC1_se`
*   HC2 passt die Residuen an, indem es durch (1 - h\_i) dividiert, wobei h\_i die Hebelwerte (Diagonalelemente der Hutmatrix) sind: `lmod.cov_HC2.round(3); lmod.HC2_se`
*   HC3 passt die Residuen an, indem es durch (1 - h\_i)² dividiert, wobei h\_i die Hebelwerte (Diagonalelemente der Hutmatrix) sind: `lmod.cov_HC3.round(3); lmod.HC3_se`

##### 3.1.8. Ausreisser-Test

*   Führt einen Ausreisser-Test an dem angepassten Modell durch: `lmod.outlier_test()`
    *   student\_resid: Die studentisierten Residuen.
    *   unadj\_p: Die nicht angepassten p-Werte für den Test.
    *   bonf(p): Die Bonferroni-angepassten p-Werte.

#### 3.2. Schrittweise Berechnung der Schätzungen für Beta

*   X-Matrix: `X = galapagos.iloc[:,1:]; X.head()`
*   X'X: `X.T @ X`
*   (X'X)^-1: `XtXi = np.linalg.inv(X.T @ X); XtXi`
*   (X'X)^-1 X'y: `(XtXi @ X.T) @ galapagos.Species`
*   Eine andere Möglichkeit zur Berechnung der Schätzungen mit (X'X)^-1 Beta = X'y: `np.linalg.solve(X.T @ X, X.T @ galapagos.Species)`

##### 3.2.1. Verwenden der Moore-Penrose-Inversen zur Berechnung der Schätzungen

*   Moore-Penrose-Inverse X^- = (X'X)^-1 X': `Xmp = np.linalg.pinv(X); Xmp.shape`
*   Berechnen Sie die Schätzungen: `Xmp @ galapagos.Species`

##### 3.2.2. Verwenden der QR-Zerlegung zur Berechnung der Schätzungen

*   q, r: `q, r = np.linalg.qr(X)`
*   f: `f = q.T @ galapagos.Species; f`
*   Berechnen Sie die Schätzungen: `sp.linalg.solve_triangular(r, f)`
*   Alternativ: `lmod_qr = smf.ols('Species ~ Area + Elevation + Nearest + Scruz  + Adjacent', galapagos).fit(method="qr") lmod_qr.params`

##### 3.2.3. Verwenden des allgemeinen Lösers für das Problem der kleinsten Quadrate

*   res: Die Summe der quadrierten Residuen der Lösung. Wenn der Rang von X kleiner als die Anzahl der Spalten in X ist, ist dies ein leeres Array.
*   rnk: Der effektive Rang der Matrix X. Dies ist die Anzahl der Singulärwerte von X, die größer als eine bestimmte Toleranz sind.
*   s: Die Singulärwerte von X.

```python
params, res, rnk, s = sp.linalg.lstsq(X, galapagos['Species'])
```

#### 3.3. Identifizierbarkeit

##### 3.3.1. Vollständige Nicht-Identifizierbarkeit

```python
galapagos['Adiff'] = galapagos.Area - galapagos.Adjacent 
lmod_ide = smf.ols('Species ~ Area+Elevation+Nearest+Scruz+Adjacent+Adiff', galapagos).fit() 
lmod_ide.sumary()

### Zeige den kleinsten Eigenwert
lmod_ide.eigenvals[-1]

### Verwendung der QR-Zerlegung
lmod_ide_qr = smf.ols('Species ~ Area+Elevation+Nearest+Scruz+Adjacent+Adiff', galapagos).fit(method="qr") 
lmod_ide_qr.sumary()
```

##### 3.3.2. Experiment für nahe Nicht-Identifizierbarkeit

```python
np.random.seed(123) 
galapagos['Adiffe'] = galapagos.Adiff + (np.random.rand(30)-0.5)*0.001 
lmod_ide_ex = smf.ols('Species ~ Area+Elevation+Nearest+Scruz+Adjacent+Adiffe', galapagos).fit() 
lmod_ide_ex.sumary()

lmod_ide_ex_qr = smf.ols('Species ~ Area+Elevation+Nearest+Scruz+Adjacent+Adiffe', galapagos).fit(method="qr") 
lmod_ide_ex_qr.sumary()
```

#### 3.4. Erklärung

Um die Wirkung der Erhebung zu verstehen, können wir das **vollständige Modell** (`lmod`) mit einem **reduzierten Modell** (`lmodr`), das nur die Erhebung als Prädiktor enthält, vergleichen.

*   Ein **Effektplot** kann verwendet werden, um die Bedeutung des Modells für einen bestimmten Prädiktor, in diesem Fall die Erhebung, zu verstehen. Der Plot zeigt die vorhergesagte Anzahl von Arten in Abhängigkeit von der Erhebung für beide Modelle.

**Wichtige Punkte:**

*   Das Konzept, Variablen konstant zu halten, ist im Kontext der Galapagos-Daten nicht sinnvoll, da es sich um Beobachtungsdaten handelt.
*   Wir behaupten **keine Kausalität** in unserer Erklärung.
*   Vergleiche zwischen Modellen können uns Einblicke geben, aber die Informationen sind **nicht absolut** und können sich ändern.

### 4. Hypothesentests

In der linearen Regression verwenden wir Hypothesentests, um die Signifikanz der Prädiktoren im Modell zu beurteilen. Die Quellen beschreiben verschiedene Arten von Hypothesentests, die in diesem Kontext durchgeführt werden können.

#### 4.1. Test aller Prädiktoren

Dieser Test prüft die Hypothese, dass **alle** Prädiktoren im Modell keinen Einfluss auf die Antwortvariable haben.

*   **Nullhypothese:** $H_0: \beta_1 = \cdots = \beta_{p-1}=0$
*   **Alternativhypothese:** Mindestens ein $\beta_j$ ist ungleich Null.

Die **F-Statistik** wird verwendet, um diese Hypothese zu testen. Sie vergleicht die Anpassung des vollständigen Modells (mit allen Prädiktoren) mit der Anpassung eines reduzierten Modells (nur mit dem Intercept). 

**Schritt-für-Schritt Berechnung der F-Statistik und des p-Werts:**

1.  **RSS des reduzierten Modells:** Dies ist die gesamte Quadratsumme (TSS), die in mit `lmod.centered_tss` berechnet werden kann.
2.  **RSS des vollständigen Modells:** `lmod.ssr`
3.  **Freiheitsgrade des vollständigen Modells:** `lmod.df_resid`
4.  **F-Statistik:**  `lmod.mse_model / lmod.mse_resid`. Dies entspricht der Formel $\frac{(TSS-RSS)/(p-1)}{RSS/(n-p)}$.
5.  **p-Wert der F-Statistik:** `1-sp.stats.f.cdf(lmod.fvalue, lmod.df_model, lmod.df_resid)`.

Die Quellen zeigen auch, wie man diesen Test mit `lmod.fvalue`, `lmod.f_pvalue` oder `lmod.compare_f_test(lmodr)` durchführen kann.

#### 4.2. Testen eines Prädiktors

Dieser Test bewertet die Signifikanz eines **einzelnen** Prädiktors im Modell. Es wird geprüft, ob das Entfernen dieses Prädiktors die Modellanpassung signifikant verschlechtert. 

*   Die F-Statistik kann auch hier verwendet werden, wobei das reduzierte Modell alle Prädiktoren außer dem zu testenden enthält. 
*   Alternativ kann die **t-Statistik** verwendet werden. Diese bewertet, wie viele Standardfehler der geschätzte Koeffizient von Null entfernt ist. 
*   **Wichtig:** Die Quellen weisen darauf hin, dass man sich nicht nur auf die p-Werte verlassen sollte, um die praktische Bedeutung eines Prädiktors zu bestimmen. Ein kleiner p-Wert bedeutet lediglich statistische Signifikanz, nicht unbedingt praktische Relevanz.

#### 4.3. Test eines Paares von Prädiktoren

Analog zu 4.2 kann man auch die Signifikanz von **zwei** Prädiktoren gleichzeitig testen. Die Quellen erwähnen jedoch, dass die Interpretation der p-Werte der einzelnen t-Tests in diesem Fall problematisch ist. Es wird empfohlen, einen einzigen F-Test zu verwenden, um mehrere Prädiktoren gleichzeitig zu testen.

#### 4.4. Test eines Unterraums

Man kann auch Hypothesen über **lineare Kombinationen** von Prädiktoren testen. Die Quellen geben Beispiele wie $H_0: β_{Area}=β_{Adjacent}$ und $H_0: β_{Elevation}=0.5$. Diese Tests werden ebenfalls mit der F-Statistik durchgeführt.

#### 4.5. Einschränkungen des Tests

Die F-Tests sind nicht universell einsetzbar. 

*   Sie können keine nichtlinearen Hypothesen testen. 
*   Sie können keine Modelle vergleichen, die nicht verschachtelt sind oder unterschiedliche Prädiktoren haben. 
*   Sie sind nicht direkt anwendbar, wenn die Modelle unterschiedliche Datensätze verwenden oder fehlende Werte haben.

#### 4.6. Permutationstests

Permutationstests sind eine **Alternative** zu den F-Tests, die **keine Normalverteilungsannahme** für die Fehler benötigen. Sie basieren auf der Idee, dass die beobachteten Daten zufällig permutiert werden, wenn die Nullhypothese (kein Zusammenhang zwischen Prädiktoren und Antwort) zutrifft.  Die Quellen beschreiben detailliert, wie Permutationstests für den Test aller Prädiktoren und für den Test eines Prädiktors durchgeführt werden können.

### 5. Konfidenzintervalle für $\beta$

Konfidenzintervalle geben einen Bereich an, in dem der wahre Wert eines Parameters (hier: die Regressionskoeffizienten $\beta$) mit einer bestimmten Wahrscheinlichkeit liegt.

*   **Berechnung:** $\hat{β} *i \pm t* {\alpha/2, t-n} se(\hat{β})$
    *   $\hat{β} *i$: Geschätzter Koeffizient
    *   $t* {\alpha/2, t-n}$: Kritischer Wert der t-Verteilung
    *   $se(\hat{β})$: Standardfehler des Koeffizienten
*   **Interpretation:** Wenn wir ein 95%-Konfidenzintervall berechnen, bedeutet dies, dass wir zu 95% sicher sind, dass der wahre Wert des Parameters innerhalb dieses Intervalls liegt.
*   **Zusammenhang mit Hypothesentests:** Die Quellen weisen darauf hin, dass wir, wenn wir ein Konfidenzintervall von (1-α)% wählen, nur Tests auf dem Signifikanzniveau von α% durchführen können.
*   **Vorteile von Konfidenzintervallen:** Sie geben uns mehr Informationen über die Größe des Effekts eines Prädiktors und sind daher informativer als reine p-Werte.

**Beispiel im Code:**

*   Kritische Werte der t-Verteilung: `qt = np.array(sp.stats.t.interval(0.95,24))`
*   Berechnung des Konfidenzintervalls: `lmod.params["Area"] + lmod.bse["Area"]*qt`
*   Alle Konfidenzintervalle: `lmod.conf_int()`

#### Bootstrap-Konfidenzintervalle

Bootstrap-Methoden sind eine **nichtparametrische Methode**, um Konfidenzintervalle zu konstruieren, die **keine Normalverteilungsannahme** erfordern. 

*   **Grundprinzip:** Anstatt die wahre Verteilung der Daten zu kennen, wird wiederholt aus den beobachteten Daten resampelt (mit Zurücklegen). 
*   **Vorteil:** Die Methode kann auch dann angewendet werden, wenn theoretische Berechnungen schwierig oder unmöglich sind.
*   **Anwendung:** Die Quellen beschreiben detailliert, wie Bootstrap-Konfidenzintervalle für die Regressionskoeffizienten berechnet werden können.

### 6. Diagnose

Nach der Anpassung eines Regressionsmodells ist es wichtig, die **Modellannahmen zu überprüfen**. Die Quellen konzentrieren sich hier auf die Annahme der **konstanten Varianz** der Fehler.

#### 6.1. Konstante Varianz

*   **Überprüfung:** Man kann die Residuen gegen die angepassten Werte plotten. Wenn die Varianz konstant ist, sollten die Punkte zufällig um die Nulllinie streuen.
*   **Beispiel im Code:** `plt.scatter(lmod.fittedvalues, lmod.resid)`
*   **Transformation:** Wenn die Varianz nicht konstant ist, kann eine Transformation der Antwortvariable hilfreich sein. Die Quellen geben das Beispiel einer Quadratwurzeltransformation für Zähldaten.

### 7. Robuste Regression

Robuste Regressionsmethoden sind weniger empfindlich gegenüber **Ausreißern** als die Methode der kleinsten Quadrate.

*   **M-Schätzung:** Ein allgemeiner Ansatz für robuste Regression, bei dem die Koeffizienten so gewählt werden, dass eine bestimmte Funktion der Residuen minimiert wird.
*   **Gewichtete kleinste Quadrate:** M-Schätzung kann als eine Form der gewichteten kleinsten Quadrate interpretiert werden, wobei die Gewichte von den Residuen abhängen.
*   **Vergleich mit OLS:** Die Quellen vergleichen die Ergebnisse von OLS mit robusten Regressionsmethoden und stellen fest, dass sich die Koeffizientenwerte leicht verschieben und die Standardfehler im Allgemeinen reduziert werden.
*   **Gewichte:** Die Analyse der Gewichte aus der robusten Anpassung kann Aufschluss darüber geben, welche Beobachtungen die Anpassung am stärksten beeinflussen.
*   **Einschränkungen:** Robuste Regression ist kein Allheilmittel. Sie löst nicht das Problem von Punkten mit großem Einfluss und hilft nicht bei der Auswahl von Prädiktoren oder der Transformation von Variablen.

### 8. Transformation

Manchmal kann eine **Transformation der Antwortvariable** die Modellanpassung verbessern und die Interpretation erleichtern. 

*   **Log-Transformation:** Bei einer Log-Transformation der Antwortvariable haben die Regressionskoeffizienten eine multiplikative Interpretation.
*   **Box-Cox-Methode:** Eine allgemeine Methode, um die optimale Transformation für die Antwortvariable zu finden.
*   **Auswahl von λ:** Der optimale Wert von λ kann durch Maximierung der Profil-Log-Likelihood bestimmt werden.
*   **Konfidenzintervall für λ:** Ein Konfidenzintervall für λ kann verwendet werden, um zu beurteilen, wie viel Rundung von λ für die Interpretierbarkeit sinnvoll ist.

**Beispiel im Code:**

Die Quellen zeigen, wie die Box-Cox-Methode mit Python implementiert werden kann, um die optimale Transformation für die Galapagos-Daten zu finden.