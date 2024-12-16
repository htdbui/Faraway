---
title: "Galapagos"
author: "db"
---


# 1. Datenbeschreibung

- Die Galapagos-Daten haben **30 Beobachtungen und 6 Variablen**.
	* Species: die Anzahl der auf der Insel gefundenen Arten
	* Area: die Fl�che der Insel (km�)
	* Elevation: die h�chste Erhebung der Insel (m)
	* Nearest: die Entfernung von der n�chsten Insel (km)
	* Scruz: die Entfernung von der Insel Santa Cruz (km)
	* Adjacent: die Fl�che der angrenzenden Insel (km�)

# 2. Pakete und Daten laden

# 3. Lineare Regression

- Die `lineare Regression` ist ein statistisches Verfahren.

* **Modell erstellen:** Definieren Sie die Beziehung zwischen den Variablen mit einer Formel (z. B. `Species ~ Area + Elevation + Nearest + Scruz + Adjacent`).
* **Modell anpassen:** `lmod = smf.ols(formula='Species ~ Area + Elevation + Nearest + Scruz  + Adjacent', data=galapagos).fit()`

# 3.1. Regressionsgr��en extrahieren

- Nachdem Sie ein `lineares Regressionsmodell` angepasst haben, k�nnen Sie verschiedene `Gr��en` extrahieren.
  - Diese helfen Ihnen, das `Modell` zu verstehen.
  - Und seine `Anpassungsg�te` zu bewerten.

## 3.1.1. Grundlagen

- **Koeffizientensch�tzungen:**
  - Die `Koeffizienten` der `linearen Gleichung`.
  - Sie geben an, wie sich die `abh�ngige Variable` in Abh�ngigkeit von den `unabh�ngigen Variablen` �ndert.
  - Beispiel: `lmod.params`.

- **Standardfehler:**
  - Ein Ma� f�r die `Unsicherheit` der `Koeffizientensch�tzungen`.
  - Beispiel: `lmod.bse`.

- **p-Werte:**
  - Ein Ma� f�r die `statistische Signifikanz` der `Koeffizienten`.
  - Beispiel: `lmod.pvalues`.

- **t-Statistiken:**
  - Ein Ma� f�r die `Gr��e der Koeffizienten` im Verh�ltnis zu ihren `Standardfehlern`.
  - Beispiel: `lmod.tvalues`.

## 3.1.2. F-Werte

- **F-Statistik:**
  - Ein Ma� f�r die `Gesamtsignifikanz` des `Modells`.
  - Vergleicht die `Anpassungsg�te` des Modells mit der eines Modells ohne `Pr�diktoren`.
  - Beispiel: `lmod.fvalue`.

- **p-Wert der F-Statistik:**
  - Ein Ma� f�r die `Wahrscheinlichkeit`, die beobachtete `F-Statistik` zu erhalten.
  - Dies gilt, wenn es keinen Zusammenhang zwischen den `Pr�diktoren` und der `Antwortvariablen` gibt.
  - Beispiel: `lmod.f_pvalue`.

## 3.1.3. Konfidenzintervalle

- **Konfidenzintervalle der Koeffizienten:**
  - Ein Bereich von Werten, der die `wahren Werte` der `Koeffizienten` mit einer bestimmten Wahrscheinlichkeit (z. B. 95%) enth�lt.
  - Beispiel: `lmod.conf_int()`.

- **Bootstrap Konfidenzintervalle:**
  - Erm�glichen die Konstruktion von `Konfidenzaussagen`, ohne anzunehmen, dass die `Daten` einer `Normalverteilung` folgen.

## 3.1.4. Anpassungsg�te

- **R-Quadrat:**
  - Ein Ma� f�r den Anteil der `Varianz` der `Antwortvariablen`.
  - Dieser Anteil wird durch die `Pr�diktoren` erkl�rt.
  - Beispiel: `lmod.rsquared`.

- **Angepasstes R-Quadrat:**
  - Ein Ma� f�r den Anteil der `Varianz` der `Antwortvariablen`.
  - Dieser Anteil wird durch die `Pr�diktoren` erkl�rt.
  - Ber�cksichtigt die Anzahl der `Pr�diktoren` im Modell.
  - Beispiel: `lmod.rsquared_adj`.

- **Akaike-Informationskriterium (AIC):**
  - Ein Ma� f�r die `Anpassungsg�te` des Modells.
  - Ber�cksichtigt die `Komplexit�t` des Modells.
  - Beispiel: `lmod.aic`.

- **Bayessches Informationskriterium (BIC):**
  - Ein Ma� f�r die `Anpassungsg�te` des Modells.
  - Ber�cksichtigt die `Komplexit�t` des Modells und die `Stichprobengr��e`.
  - Beispiel: `lmod.bic`.

## 3.1.5. Quadratsummen

- **Residuenquadratsumme (RSS):**
  - Die Summe der `quadrierten Differenzen` zwischen den `beobachteten Werten` der `Antwortvariablen` und den vom `Modell` vorhergesagten Werten.
  - Beispiel: `lmod.ssr`.

- **Erkl�rte Quadratsumme (ESS):**
  - Die Summe der `quadrierten Differenzen` zwischen den vom `Modell` vorhergesagten Werten und dem `Mittelwert` der `Antwortvariablen`.
  - Beispiel: `lmod.ess`.

- **Totale Quadratsumme (TSS):**
  - Die Summe der `quadrierten Differenzen` zwischen den `beobachteten Werten` der `Antwortvariablen` und dem `Mittelwert` der `Antwortvariablen`.
  - Beispiel: `lmod.centered_tss`.

## 3.1.6. Angepasste Werte und Residuen

- **Angepasste Werte:**
  - Die vom Modell `vorhergesagten Werte` der `Antwortvariablen`.
  - Beispiel: `lmod.fittedvalues`.

- **Residuen:**
  - Die Differenzen zwischen den `beobachteten Werten` der `Antwortvariablen` und den vom Modell `vorhergesagten Werten`.
  - Beispiel: `lmod.resid`.

- **Pearson-Residuen:**
  - Eine Art von `standardisierten Residuen`.

## 3.1.7. Kovarianzmatrix

- **Kovarianzmatrix der Koeffizienten:**
  - Eine Matrix, die die `Varianzen` und `Kovarianzen` der `Koeffizientensch�tzungen` enth�lt.
  - Beispiel: `lmod.cov_params()`.

- **Normalisierte Kovarianzmatrix:**
  - Die `Kovarianzmatrix` dividiert durch die `Restvarianz`.

## 3.1.8. Ausrei�ertest

* **Ausrei�ertest:** Ein Test, um festzustellen, ob es Beobachtungen gibt, die nicht zum Modell passen (z. B. `lmod.outlier_test()`).

# 3.2. Schrittweise Berechnung der Sch�tzungen f�r Beta

**Die Koeffizienten werden mit der Formel** $\hat{\beta} = (X'X)^{-1}X'y$  berechnet.

* **X:** die Matrix der Pr�diktorvariablen.
* **y:** der Vektor der Antwortvariablen.

## 3.2.1. Verwendung der Moore-Penrose-Inversen zur Berechnung der Sch�tzungen

- **Moore-Penrose-Inverse:**
  - Eine verallgemeinerte `Inverse` einer `Matrix`.
  - Kann verwendet werden, um die `Koeffizienten` eines `linearen Regressionsmodells` zu berechnen.
  - Beispiel: `Xmp = np.linalg.pinv(X)`.

## 3.2.2. Verwendung der QR-Zerlegung zur Berechnung der Sch�tzungen

- **QR-Zerlegung:**
  - Die `QR-Zerlegung` ist eine Methode.
  - Sie zerlegt eine `Matrix` in zwei Teile.
    - Ein Teil ist eine `orthogonale Matrix`.
    - Der andere Teil ist eine `obere Dreiecksmatrix`.
  - Diese Zerlegung hilft bei der Berechnung von `Koeffizienten` eines `linearen Regressionsmodells`.
    - Beispiel: `q, r = np.linalg.qr(X)`.

## 3.2.3. Verwendung des allgemeinen L�sers f�r das Problem der kleinsten Quadrate

Der allgemeine L�ser f�r das Problem der kleinsten Quadrate ist eine numerische Methode, um die Koeffizienten eines linearen Regressionsmodells zu berechnen (z. B. `params, res, rnk, s = sp.linalg.lstsq(X, galapagos['Species'])`).

# 3.3. Identifizierbarkeit

- **Identifizierbarkeit in der linearen Regression:**
  - Bezieht sich auf die F�higkeit, `eindeutige Sch�tzungen` f�r die `Modellparameter` zu erhalten.
  
- **Problem bei Linearkombination der Pr�diktoren:**
  - Wenn `eine Linearkombination der Pr�diktoren` eine `Konstante` ergibt, sind die `Parameter` des Modells nicht identifizierbar.
  
- **Folgen:**
  - Dies f�hrt zu `nicht eindeutigen Koeffizientensch�tzungen`.

# 3.4. Erkl�rung

- **Interpretation der Ergebnisse eines linearen Regressionsmodells:**
  - Sobald ein `lineares Regressionsmodell` angepasst ist, ist es wichtig, die Ergebnisse zu interpretieren.
  - Die Interpretation hilft zu verstehen, was die Ergebnisse �ber die `Beziehung zwischen den Variablen` aussagen.

- **Effekt einer Variablen verstehen:**
  - Um den `Effekt einer Variablen` zu verstehen, w�hrend andere `Variablen konstant` gehalten werden, kann ein `Effektdiagramm` hilfreich sein.

- **Besonderheit der Galapagos-Daten:**
  - Bei den `Galapagos-Daten` ist es nicht m�glich, `Variablen konstant zu halten`, da es sich um `Beobachtungsdaten` handelt.
  - Daher sollten die `Ergebnisse` mit Vorsicht interpretiert werden.


# 4. Hypothesentests

- **Hypothesentests in der linearen Regression:**
  - Werden verwendet, um die `Signifikanz` der `Pr�diktoren` im Modell zu beurteilen.

- **Nullhypothese ($H_0$):**
  - Besagt, dass es keinen `Zusammenhang` zwischen den `Pr�diktoren` und der `Antwortvariablen` gibt.

- **Alternativhypothese ($H_a$):**
  - Besagt, dass es einen `Zusammenhang` gibt.

## 4.1. Test aller Pr�diktoren

- **Gesamtsignifikanz des Modells:**
  - Dieser Test beurteilt die `Gesamtsignifikanz` des Modells.
  - Vergleicht die `Anpassungsg�te` eines Modells mit allen `Pr�diktoren` mit einem Modell ohne Pr�diktoren (`nur der Achsenabschnitt`).

- **Hypothesen:**
  - **Nullhypothese ($H_0$):**
    - Alle `Regressionskoeffizienten` sind gleich `Null` $(\beta_1 = \beta_2 = \cdots = \beta_p = 0)$.
  - **Alternativhypothese ($H_a$):**
    - Mindestens ein `Regressionskoeffizient` ist ungleich `Null`.

- **F-Statistik:**
  - Verwendet, um diese Hypothesen zu testen.
  - Vergleicht die `erkl�rte Varianz` des Modells mit der `unerkl�rten Varianz`.

- **Berechnung der F-Statistik und des p-Werts:**
  1. **RSS (Residuenquadratsumme) f�r das reduzierte Modell berechnen:**
     - Dies ist gleich der `TSS (Gesamtquadratsumme)`.
  2. **RSS f�r das vollst�ndige Modell berechnen:**
     - Inklusive aller `Pr�diktoren`.
  3. **Freiheitsgrade f�r das vollst�ndige Modell berechnen:**
     - Dies ist `n - p`, wobei `n` die Anzahl der Beobachtungen und `p` die Anzahl der Parameter im Modell ist.
  4. **F-Statistik berechnen:**
     - Formel: $F = \frac{(\text{RSS}_{\text{reduziert}} - \text{RSS}_{\text{vollst�ndig}}) / (p - 1)}{\text{RSS}_{\text{vollst�ndig}} / (n - p)}$
  5. **p-Wert der F-Statistik berechnen:**
     - Dies ist die Wahrscheinlichkeit, eine `F-Statistik` zu beobachten, die so gro� oder gr��er als die berechnete ist, wenn die `Nullhypothese` wahr ist.

- **Python-Implementierung:**
  - `F-Statistik` und `p-Wert` k�nnen mit den Attributen `fvalue` und `f_pvalue` des angepassten Modells abgerufen werden.

## 4.2. Testen eines Pr�diktors

- **Signifikanz eines einzelnen Pr�diktors:**
  - Dieser Test beurteilt die `Signifikanz` eines `einzelnen Pr�diktors` im Modell.

- **Hypothesen:**
  - **Nullhypothese ($H_0$):**
    - Der `Regressionskoeffizient` des getesteten Pr�diktors ist gleich `Null`.
  - **Alternativhypothese ($H_a$):**
    - Der `Regressionskoeffizient` des getesteten Pr�diktors ist ungleich `Null`.

- **Durchf�hrung des Tests:**
  - Sie k�nnen diesen Test mit einem `t-Test` oder einem `F-Test` durchf�hren.
  - Die `quadrierte t-Statistik` ist gleich der `F-Statistik`.

- **Interpretation des p-Werts:**
  - Ein `niedriger p-Wert` (typischerweise < 0,05) deutet darauf hin, dass der Pr�diktor signifikant zur Vorhersage der `Antwortvariablen` beitr�gt.
  - Beachten Sie, dass `statistische Signifikanz` nicht unbedingt `praktische Signifikanz` bedeutet.

## 4.3. Testen eines Pr�diktorpaares

- **Gemeinsame Signifikanz von zwei Pr�diktoren:**
  - Dieser Test beurteilt die `gemeinsame Signifikanz` von `zwei Pr�diktoren` im Modell.

- **Hypothesen:**
  - **Nullhypothese ($H_0$):**
    - Die `Regressionskoeffizienten` beider Pr�diktoren sind gleich `Null`.
  - **Alternativhypothese ($H_a$):**
    - Mindestens einer der `Regressionskoeffizienten` ist ungleich `Null`.

- **Verwendung eines F-Tests:**
  - Verwenden Sie einen `F-Test`, um diese Hypothese zu testen.
  - Vermeiden Sie die Verwendung mehrerer `t-Tests`, da diese nicht einfach zu kombinieren sind und zu falschen Schlussfolgerungen f�hren k�nnen.

## 4.4. Testen eines Unterraums

- **Signifikanz einer linearen Kombination von Pr�diktoren:**
  - Dieser Test beurteilt die `Signifikanz` einer `linearen Kombination von Pr�diktoren` im Modell.

- **Hypothesen:**
  - **Nullhypothese ($H_0$):**
    - Die `lineare Kombination` der `Regressionskoeffizienten` ist gleich einem bestimmten Wert.
  - **Alternativhypothese ($H_a$):**
    - Die `lineare Kombination` der `Regressionskoeffizienten` ist ungleich dem bestimmten Wert.

- **Verwendung eines F-Tests:**
  - Verwenden Sie einen `F-Test`, um diese Hypothese zu testen.

## 4.5. Grenzen des Tests

- **Einschr�nkungen von F-Tests in der linearen Regression:**
  - Nichtlineare Hypothesen.
  - Nicht verschachtelte Modelle. 
  - Fehlende Werte und unterschiedliche Datens�tze:

## 4.6. Permutationstests

- **Permutation Tests:**
  - Eine alternative Testmethode, die keine Annahme von normalen Fehlern ben�tigt.

- **Interpretation:**
  - Wenn die Antwort nicht mit den Pr�diktoren zusammenh�ngt, w�ren die beobachteten Antwortwerte zuf�llig �ber die F�lle verteilt, ohne Bezug zu den Pr�diktoren.

- **F-Statistik:**
  - Ein Ma� f�r die Assoziation zwischen den Pr�diktoren und der Antwort.
  - Gr��ere Werte der F-Statistik weisen auf st�rkere Assoziationen hin.

- **Wahrscheinlichkeit bei keiner Beziehung:**
  - Was ist die Chance unter der Annahme, dass keine Beziehung zwischen Pr�diktoren und Antwort besteht, eine F-Statistik so gro� oder gr��er als die beobachtete zu sehen?

- **Durchf�hrung des Permutationstests:**
  - Berechnung der F-Statistik f�r alle m�glichen Permutationen der Antwortvariablen.
  - Bestimmung des Anteils dieser Permutationen, die die beobachtete F-Statistik �berschreiten.
  - Ein kleiner Anteil deutet darauf hin, dass die Antwort mit den Pr�diktoren zusammenh�ngt, und wir verwerfen die Annahme keiner Beziehung.

- **Sch�tzung des Anteils:**
  - Dieser Anteil wird durch einen p-Wert gesch�tzt, der auf der Annahme normaler Fehler basiert.

- **Permutationstests f�r einen Pr�diktor:**
  - Anstatt die Antwortvariable zu permutieren, wird eine der Pr�diktorvariablen permutiert.

# 5. Konfidenzintervalle f�r $\beta$

- **Konfidenzintervalle:**
  - Liefern einen Bereich `plausibler Werte` f�r die `Regressionskoeffizienten` ($\beta$s).
  - Geben die `Unsicherheit` �ber die gesch�tzten Koeffizienten an.
  - Basieren auf den `Daten` und dem gew�hlten `Konfidenzniveau`.

- **95%-Konfidenzintervall:**
  - Bedeutet, dass bei einer wiederholten `Stichprobenziehung` aus derselben `Grundgesamtheit` 95% der berechneten `Konfidenzintervalle` den `wahren Wert` des Parameters enthalten w�rden.

- **Berechnung des Konfidenzintervalls:**
  - Konfidenzintervall: $\hat{\beta}_i \pm t_{\alpha/2, t-n} se(\hat{\beta})$

  - Bestandteile:
    - $\hat{\beta}_i$: Gesch�tzter `Regressionskoeffizient` f�r den `i-ten Pr�diktor`.
    - $t_{\alpha/2, t-n}$: Kritischer Wert aus der `t-Verteilung` f�r das gew�hlte `Konfidenzniveau` (1-$\alpha$) und die `Freiheitsgrade` (n-p). 
      - n: `Stichprobengr��e`
      - p: Anzahl der `Parameter` im Modell.
    - $se(\hat{\beta})$: `Standardfehler` des gesch�tzten Koeffizienten.
- **Interpretation des Konfidenzintervalls:**
  - Wenn das `Konfidenzintervall` den Wert `Null` **nicht** enth�lt, ist der entsprechende `Pr�diktor` statistisch `signifikant` auf dem gew�hlten `Signifikanzniveau`.
  - Die `Breite` des Konfidenzintervalls gibt die `Pr�zision` der Sch�tzung an.
    - Ein `breiteres Intervall` weist auf eine gr��ere `Unsicherheit` hin.
- **Vorteile der Verwendung von Konfidenzintervallen:**
  - Konfidenzintervalle liefern mehr `Informationen` als nur die `Punktsch�tzung` des Koeffizienten.
  - Sie erm�glichen eine bessere Beurteilung der `praktischen Signifikanz` eines Pr�diktors.

## Bootstrap-Konfidenzintervalle

- **Bootstrap-Methode:**
  - Eine `Resampling-Technik`, die `Konfidenzintervalle` ohne Annahme einer `Normalverteilung` erm�glicht.

- **Schritte zur Berechnung von Bootstrap-Konfidenzintervallen:**
  1. Neue `Residuen` ziehen:
     - Aus den beobachteten Residuen $\hat{e}_1, \cdots, \hat{e}_n$ werden mit Zur�cklegen neue Residuen $\mathbf{e}^*$ gezogen.
  2. Neue `Antwortvariable` erstellen:
     - Neue Residuen und urspr�ngliche Pr�diktoren nutzen, um $\mathbf{y}^* = \mathbf{X\hat{\beta}} + \mathbf{e}^*$ zu erzeugen.
  3. Neues `Modell` sch�tzen:
     - Mit $\mathbf{X}$ und $\mathbf{y}^*$ wird ein neues Modell gesch�tzt, und der Koeffizient $\mathbf{\hat{\beta}}$ berechnet.
  4. Schritte 1-3 `wiederholen`:
     - Diese Schritte werden viele Male (z.B. 4000 Mal) wiederholt.
  5. `Konfidenzintervall` bestimmen:
     - Das `95%-Konfidenzintervall` ergibt sich aus den `2.5%-` und `97.5%-Quantilen` der Bootstrap-Verteilung der Koeffizienten.

- **Vorteile der Bootstrap-Methode:**
  - Keine `Normalverteilungsannahme` erforderlich.
  - Geeignet f�r `komplexe Modelle`, bei denen die theoretische Berechnung von Konfidenzintervallen schwierig ist.

# 6. Diagnose

- **Diagnose von Regressionsmodellen:**
  - �berpr�fung der Modellannahmen und Identifizierung potenzieller Probleme.

- **6.1. Konstante Varianz (Homoskedastizit�t):**
  - Eine wichtige Annahme der linearen Regression ist die `Homoskedastizit�t`.
  - Die Varianz der Residuen sollte �ber den gesamten Wertebereich der Pr�diktoren konstant sein.

- **�berpr�fung der Homoskedastizit�t:**
  - **Streudiagramm der Residuen gegen die angepassten Werte:**
    - Wenn die Punkte zuf�llig um die horizontale Linie bei Null streuen und keine systematischen Muster zeigen, deutet dies auf konstante Varianz hin.

- **Verletzung der Homoskedastizit�t:**
  - Wenn die Varianz der Residuen nicht konstant ist (`Heteroskedastizit�t`):
    - Die Standardfehler der Koeffizienten sind verzerrt.
    - Die Hypothesentests sind ung�ltig.
  - Eine Transformation der Antwortvariablen (z.B. Wurzel- oder Logarithmustransformation) kann helfen, die Homoskedastizit�t herzustellen.

- **Beispiel:**
  - Streudiagramm der Residuen gegen die angepassten Werte f�r ein Modell zur Vorhersage der Artenvielfalt auf den Galapagos-Inseln.
  - Das Diagramm zeigt eine `zunehmende Varianz der Residuen` mit steigenden angepassten Werten.
  - Eine `Wurzeltransformation der Antwortvariablen` verbessert die Homoskedastizit�t.

# 7. Robuste Regression

- **Least-Squares-Regression:**
  - Funktioniert am besten mit normalverteilten Fehlern.
  - Langschw�nzige Fehler k�nnen problematisch sein.

- **Umgang mit extremen Werten:**
  - Ursache verstehen: Fehler entfernen, echte Beobachtungen behalten.
  - Robuste Regression sch�tzt $\mathbb{E}(y) = X\beta$, unabh�ngig von Ausrei�ern.

- **Ausrei�ererkennung:**
  - Methoden zur Erkennung und Entfernung vor der Regression.
  - Bei mehreren Ausrei�ern ist robuste Regression besser.

- **M-Sch�tzung:**
  - Minimiert $\sum_{i=1}^{n}\rho(y_i - x_i^\prime \beta)$.
  - Beispiele:
    - $\rho(x) = x^2$: Least-Squares.
    - $\rho(x) = |x|$: Least Absolute Deviation (LAD).
    - $\rho(x)$ nach Huber: Kompromiss zwischen Least-Squares und LAD.

- **Gewichtete Least-Squares:**
  - Normalgleichung: $X^\prime(y - X\beta) = 0$.
  - Mit Gewichten: $\sum_{i=1}^{n} w_i x_{ij}(y_i - \sum_{j=1}^{p} x_{ij} \beta_j) = 0$.
  - Gewichtsfunktion $w(u)$:
    - Least-Squares: konstant.
    - LAD: $w(u) = 1/|u|$.
    - Huber: $w(u) = \begin{cases} 1 & \text{wenn } |u| \le c \\ \frac{c}{|u|} & \text{sonst} \end{cases}$.

- **Iterative Berechnung:**
  - M-Sch�tzungen erfordern iterative Schritte.
  - Wechsel zwischen Weighted Least-Squares und Neuberechnung der Gewichte, bis Konvergenz erreicht ist.
  - Standardfehler durch WLS mit $\widehat{\text{var}}(\hat{\beta}) = \hat{\sigma}^2 (X^\prime W X)^{-1}$ und einer robusten Sch�tzung von $\sigma^2$.

# 8. Transformation

- **Log Transformation Interpretation:**
  - Regression equation: $\log{\hat{y}}={\hat{\beta}}_0+{\hat{\beta}}_1x_1+\ldots+{\hat{\beta}}_px_p$
  - Equivalent in original scale: $\hat{y}=e^{{\hat{\beta}}_0}e^{{\hat{\beta}}_1x_1}\ldots e^{{\hat{\beta}}_px_p}$
  - Increasing $x_1$ by one multiplies the predicted response by $e^{\hat{\beta}_1}$.
  - For small $x$, $\log(1 + x) \approx x$. For example, if $\beta_1 = 0.09$, increasing $x_1$ by one raises $\log(y)$ by 0.09 or $y$ by 9%.

- **Box-Cox Transformation:**
  - Transforms positive responses $y$ to $g_\lambda(y)$, based on $\lambda$:
    - $g_\lambda(y) = \begin{cases}\frac{y^\lambda-1}{\lambda} & \lambda \neq 0 \\ \log{y} & \lambda = 0 \end{cases}$
  - $g_\lambda(y)$ changes smoothly with $\lambda$ for fixed $y > 0$.
  - Optimal $\lambda$ is chosen using maximum likelihood, assuming normal distribution of errors.
  - Profile log-likelihood for fitting the model:
    - $L(\lambda) = -\frac{n}{2}\log\left(\frac{RSS_\lambda}{n}\right) + (\lambda-1)\sum\log{y_i}$
  - Find best $\hat{\lambda}$ by maximizing $L(\lambda)$ numerically.
  - For prediction, use $y^\lambda$ directly; $\frac{y^\lambda - 1}{\lambda}$ ensures smooth transition to $\log{y}$ as $\lambda \to 0$.
  - Round $\lambda$ for interpretability (e.g., $\sqrt{y}$ if $\hat{\lambda} \approx 0.5$).

- **Checking Necessity of Transformation:**
  - Construct confidence interval for $\lambda$.
  - $100(1-\alpha)\%$ confidence interval: $\{\lambda: L(\lambda) > L(\hat{\lambda}) - \frac{1}{2}\chi_{1,1-\alpha}^2\}$
  - Based on inverting likelihood ratio test for $H_0: \lambda = \lambda_0$.
  - Uses statistic $2[L(\hat{\lambda}) - L(\lambda_0)] \sim \chi_1^2$.
  - Confidence interval helps determine reasonable rounding of $\lambda$ for interpretability.





