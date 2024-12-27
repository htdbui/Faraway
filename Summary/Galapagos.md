---
title: "Galapagos"
author: "db"
---


# 1. Datenbeschreibung

- Die Galapagos-Daten haben **30 Beobachtungen und 6 Variablen**.
	* Species: die Anzahl der auf der Insel gefundenen Arten
	* Area: die Fläche der Insel (km²)
	* Elevation: die höchste Erhebung der Insel (m)
	* Nearest: die Entfernung von der nächsten Insel (km)
	* Scruz: die Entfernung von der Insel Santa Cruz (km)
	* Adjacent: die Fläche der angrenzenden Insel (km²)


# 3. Lineare Regression

- Die `lineare Regression` ist ein statistisches Verfahren.

* **Modell erstellen:** Definieren Sie die Beziehung zwischen den Variablen mit einer Formel (z. B. `Species ~ Area + Elevation + Nearest + Scruz + Adjacent`).
* **Modell anpassen:** `lmod = smf.ols(formula='Species ~ Area + Elevation + Nearest + Scruz  + Adjacent', data=galapagos).fit()`

# 3.1. Regressionsgrößen extrahieren

- Nachdem Sie ein `lineares Regressionsmodell` angepasst haben, können Sie verschiedene `Größen` extrahieren.
  - Diese helfen Ihnen, das `Modell` zu verstehen.
  - Und seine `Anpassungsgüte` zu bewerten.

## 3.1.1. Grundlagen

- **Koeffizientenschätzungen:**
  - Die `Koeffizienten` der `linearen Gleichung`.
  - Sie geben an, wie sich die `abhängige Variable` in Abhängigkeit von den `unabhängigen Variablen` ändert.
  - `lmod.params`.

- **Standardfehler:**
  - Ein Maß für die `Unsicherheit` der `Koeffizientenschätzungen`.
  - `lmod.bse`.

- **p-Werte:**
  - Ein Maß für die `statistische Signifikanz` der `Koeffizienten`.
  - `lmod.pvalues`.

- **t-Statistiken:**
  - Ein Maß für die `Größe der Koeffizienten` im Verhältnis zu ihren `Standardfehlern`.
  - `lmod.tvalues`.

## 3.1.2. F-Werte

- **F-Statistik:**
  - Ein Maß für die `Gesamtsignifikanz` des `Modells`.
  - Vergleicht die `Anpassungsgüte` des Modells mit der eines Modells ohne `Prädiktoren`.
  - `lmod.fvalue`.

- **p-Wert der F-Statistik:**
  - Ein Maß für die `Wahrscheinlichkeit`, die beobachtete `F-Statistik` zu erhalten.
  - Dies gilt, wenn es keinen Zusammenhang zwischen den `Prädiktoren` und der `Antwortvariablen` gibt.
  - `lmod.f_pvalue`.

## 3.1.3. Konfidenzintervalle

- **Konfidenzintervalle der Koeffizienten:**
  - Ein Bereich von Werten, der die `wahren Werte` der `Koeffizienten` mit einer bestimmten Wahrscheinlichkeit (z. B. 95%) enthält.
  - `lmod.conf_int()`.

- **Bootstrap Konfidenzintervalle:**
  - Ermöglichen die Konstruktion von `Konfidenzaussagen`, ohne anzunehmen, dass die `Daten` einer `Normalverteilung` folgen.

## 3.1.4. Anpassungsgüte

- **R-Quadrat:**
  - Ein Maß für den Anteil der `Varianz` der `Antwortvariablen`.
  - Dieser Anteil wird durch die `Prädiktoren` erklärt.
  - `lmod.rsquared`.

- **Angepasstes R-Quadrat:**
  - Ein Maß für den Anteil der `Varianz` der `Antwortvariablen`.
  - Dieser Anteil wird durch die `Prädiktoren` erklärt.
  - Berücksichtigt die Anzahl der `Prädiktoren` im Modell.
  - `lmod.rsquared_adj`.

- **Akaike-Informationskriterium (AIC):**
  - Ein Maß für die `Anpassungsgüte` des Modells.
  - Berücksichtigt die `Komplexität` des Modells.
  - `lmod.aic`.

- **Bayessches Informationskriterium (BIC):**
  - Ein Maß für die `Anpassungsgüte` des Modells.
  - Berücksichtigt die `Komplexität` des Modells und die `Stichprobengröße`.
  - `lmod.bic`.

## 3.1.5. Quadratsummen

- **Residuenquadratsumme (RSS):**
  - Die Summe der `quadrierten Differenzen` zwischen den `beobachteten Werten` der `Antwortvariablen` und den vom `Modell` vorhergesagten Werten.
  - `lmod.ssr`.

- **Erklärte Quadratsumme (ESS):**
  - Die Summe der `quadrierten Differenzen` zwischen den vom `Modell` vorhergesagten Werten und dem `Mittelwert` der `Antwortvariablen`.
  - `lmod.ess`.

- **Totale Quadratsumme (TSS):**
  - Die Summe der `quadrierten Differenzen` zwischen den `beobachteten Werten` der `Antwortvariablen` und dem `Mittelwert` der `Antwortvariablen`.
  - `lmod.centered_tss`.

## 3.1.6. Angepasste Werte und Residuen

- **Angepasste Werte:**
  - Die vom Modell `vorhergesagten Werte` der `Antwortvariablen`.
  - `lmod.fittedvalues`.

- **Residuen:**
  - Die Differenzen zwischen den `beobachteten Werten` der `Antwortvariablen` und den vom Modell `vorhergesagten Werten`.
  - `lmod.resid`.

- **Pearson-Residuen:**
  - Eine Art von `standardisierten Residuen`.

## 3.1.7. Kovarianzmatrix

- **Kovarianzmatrix der Koeffizienten:**
  - Eine Matrix, die die `Varianzen` und `Kovarianzen` der `Koeffizientenschätzungen` enthält.
  - `lmod.cov_params()`.

- **Normalisierte Kovarianzmatrix:**
  - Die `Kovarianzmatrix` dividiert durch die `Restvarianz`.

## 3.1.8. Ausreißertest

* **Ausreißertest:** Ein Test, um festzustellen, ob es Beobachtungen gibt, die nicht zum Modell passen (z. B. `lmod.outlier_test()`).

# 3.2. Schrittweise Berechnung der Schätzungen für Beta

**Die Koeffizienten werden mit der Formel** $\hat{\beta} = (X'X)^{-1}X'y$  berechnet.

* **X:** die Matrix der Prädiktorvariablen.
* **y:** der Vektor der Antwortvariablen.

## 3.2.1. Verwendung der Moore-Penrose-Inversen zur Berechnung der Schätzungen

- **Moore-Penrose-Inverse:**
  - Eine verallgemeinerte `Inverse` einer `Matrix`.
  - Kann verwendet werden, um die `Koeffizienten` eines `linearen Regressionsmodells` zu berechnen.
  - `Xmp = np.linalg.pinv(X)`.

## 3.2.2. Verwendung der QR-Zerlegung zur Berechnung der Schätzungen

- **QR-Zerlegung:**
  - Die `QR-Zerlegung` ist eine Methode.
  - Sie zerlegt eine `Matrix` in zwei Teile.
    - Ein Teil ist eine `orthogonale Matrix`.
    - Der andere Teil ist eine `obere Dreiecksmatrix`.
  - Diese Zerlegung hilft bei der Berechnung von `Koeffizienten` eines `linearen Regressionsmodells`.
    - `q, r = np.linalg.qr(X)`.

## 3.2.3. Verwendung des allgemeinen Lösers für das Problem der kleinsten Quadrate

Der allgemeine Löser für das Problem der kleinsten Quadrate ist eine numerische Methode, um die Koeffizienten eines linearen Regressionsmodells zu berechnen (z. B. `params, res, rnk, s = sp.linalg.lstsq(X, galapagos['Species'])`).

# 3.3. Identifizierbarkeit

- **Identifizierbarkeit in der linearen Regression:**
  - Bezieht sich auf die Fähigkeit, `eindeutige Schätzungen` für die `Modellparameter` zu erhalten.
  
- **Problem bei Linearkombination der Prädiktoren:**
  - Wenn `eine Linearkombination der Prädiktoren` eine `Konstante` ergibt, sind die `Parameter` des Modells nicht identifizierbar.
  
- **Folgen:**
  - Dies führt zu `nicht eindeutigen Koeffizientenschätzungen`.

# 3.4. Erklärung

- **Interpretation der Ergebnisse eines linearen Regressionsmodells:**
  - Sobald ein `lineares Regressionsmodell` angepasst ist, ist es wichtig, die Ergebnisse zu interpretieren.
  - Die Interpretation hilft zu verstehen, was die Ergebnisse über die `Beziehung zwischen den Variablen` aussagen.

- **Effekt einer Variablen verstehen:**
  - Um den `Effekt einer Variablen` zu verstehen, während andere `Variablen konstant` gehalten werden, kann ein `Effektdiagramm` hilfreich sein.

- **Besonderheit der Galapagos-Daten:**
  - Bei den `Galapagos-Daten` ist es nicht möglich, `Variablen konstant zu halten`, da es sich um `Beobachtungsdaten` handelt.
  - Daher sollten die `Ergebnisse` mit Vorsicht interpretiert werden.


# 4. Hypothesentests

- **Hypothesentests in der linearen Regression:**
  - Werden verwendet, um die `Signifikanz` der `Prädiktoren` im Modell zu beurteilen.

- **Nullhypothese ($H_0$):**
  - Besagt, dass es keinen `Zusammenhang` zwischen den `Prädiktoren` und der `Antwortvariablen` gibt.

- **Alternativhypothese ($H_a$):**
  - Besagt, dass es einen `Zusammenhang` gibt.

## 4.1. Test aller Prädiktoren

- **Gesamtsignifikanz des Modells:**
  - Dieser Test beurteilt die `Gesamtsignifikanz` des Modells.
  - Vergleicht die `Anpassungsgüte` eines Modells mit allen `Prädiktoren` mit einem Modell ohne Prädiktoren (`nur der Achsenabschnitt`).

- **Hypothesen:**
  - **Nullhypothese ($H_0$):**
    - Alle `Regressionskoeffizienten` sind gleich `Null` $(\beta_1 = \beta_2 = \cdots = \beta_p = 0)$.
  - **Alternativhypothese ($H_a$):**
    - Mindestens ein `Regressionskoeffizient` ist ungleich `Null`.

- **F-Statistik:**
  - Verwendet, um diese Hypothesen zu testen.
  - Vergleicht die `erklärte Varianz` des Modells mit der `unerklärten Varianz`.

- **Berechnung der F-Statistik und des p-Werts:**
  1. **RSS (Residuenquadratsumme) für das reduzierte Modell berechnen:**
     - Dies ist gleich der `TSS (Gesamtquadratsumme)`.
  2. **RSS für das vollständige Modell berechnen:**
     - Inklusive aller `Prädiktoren`.
  3. **Freiheitsgrade für das vollständige Modell berechnen:**
     - Dies ist `n - p`, wobei `n` die Anzahl der Beobachtungen und `p` die Anzahl der Parameter im Modell ist.
  4. **F-Statistik berechnen:**
     - Formel: $F = \frac{(\text{RSS}_{\text{reduziert}} - \text{RSS}_{\text{vollständig}}) / (p - 1)}{\text{RSS}_{\text{vollständig}} / (n - p)}$
  5. **p-Wert der F-Statistik berechnen:**
     - Dies ist die Wahrscheinlichkeit, eine `F-Statistik` zu beobachten, die so groß oder größer als die berechnete ist, wenn die `Nullhypothese` wahr ist.

- **Python-Implementierung:**
  - `F-Statistik` und `p-Wert` können mit den Attributen `fvalue` und `f_pvalue` des angepassten Modells abgerufen werden.

## 4.2. Testen eines Prädiktors

- **Signifikanz eines einzelnen Prädiktors:**
  - Dieser Test beurteilt die `Signifikanz` eines `einzelnen Prädiktors` im Modell.

- **Hypothesen:**
  - **Nullhypothese ($H_0$):**
    - Der `Regressionskoeffizient` des getesteten Prädiktors ist gleich `Null`.
  - **Alternativhypothese ($H_a$):**
    - Der `Regressionskoeffizient` des getesteten Prädiktors ist ungleich `Null`.

- **Durchführung des Tests:**
  - Sie können diesen Test mit einem `t-Test` oder einem `F-Test` durchführen.
  - Die `quadrierte t-Statistik` ist gleich der `F-Statistik`.

- **Interpretation des p-Werts:**
  - Ein `niedriger p-Wert` (typischerweise < 0,05) deutet darauf hin, dass der Prädiktor signifikant zur Vorhersage der `Antwortvariablen` beiträgt.
  - Beachten Sie, dass `statistische Signifikanz` nicht unbedingt `praktische Signifikanz` bedeutet.

## 4.3. Testen eines Prädiktorpaares

- **Gemeinsame Signifikanz von zwei Prädiktoren:**
  - Dieser Test beurteilt die `gemeinsame Signifikanz` von `zwei Prädiktoren` im Modell.

- **Hypothesen:**
  - **Nullhypothese ($H_0$):**
    - Die `Regressionskoeffizienten` beider Prädiktoren sind gleich `Null`.
  - **Alternativhypothese ($H_a$):**
    - Mindestens einer der `Regressionskoeffizienten` ist ungleich `Null`.

- **Verwendung eines F-Tests:**
  - Verwenden Sie einen `F-Test`, um diese Hypothese zu testen.
  - Vermeiden Sie die Verwendung mehrerer `t-Tests`, da diese nicht einfach zu kombinieren sind und zu falschen Schlussfolgerungen führen können.

## 4.4. Testen eines Unterraums

- **Signifikanz einer linearen Kombination von Prädiktoren:**
  - Dieser Test beurteilt die `Signifikanz` einer `linearen Kombination von Prädiktoren` im Modell.

- **Hypothesen:**
  - **Nullhypothese ($H_0$):**
    - Die `lineare Kombination` der `Regressionskoeffizienten` ist gleich einem bestimmten Wert.
  - **Alternativhypothese ($H_a$):**
    - Die `lineare Kombination` der `Regressionskoeffizienten` ist ungleich dem bestimmten Wert.

- **Verwendung eines F-Tests:**
  - Verwenden Sie einen `F-Test`, um diese Hypothese zu testen.

## 4.5. Grenzen des Tests

- **Einschränkungen von F-Tests in der linearen Regression:**
  - Nichtlineare Hypothesen.
  - Nicht verschachtelte Modelle. 
  - Fehlende Werte und unterschiedliche Datensätze:

## 4.6. Permutationstests

- **Permutation Tests:**
  - Eine alternative Testmethode, die keine Annahme von normalen Fehlern benötigt.

- **Interpretation:**
  - Wenn die Antwort nicht mit den Prädiktoren zusammenhängt, wären die beobachteten Antwortwerte zufällig über die Fälle verteilt, ohne Bezug zu den Prädiktoren.

- **F-Statistik:**
  - Ein Maß für die Assoziation zwischen den Prädiktoren und der Antwort.
  - Größere Werte der F-Statistik weisen auf stärkere Assoziationen hin.

- **Wahrscheinlichkeit bei keiner Beziehung:**
  - Was ist die Chance unter der Annahme, dass keine Beziehung zwischen Prädiktoren und Antwort besteht, eine F-Statistik so groß oder größer als die beobachtete zu sehen?

- **Durchführung des Permutationstests:**
  - Berechnung der F-Statistik für alle möglichen Permutationen der Antwortvariablen.
  - Bestimmung des Anteils dieser Permutationen, die die beobachtete F-Statistik überschreiten.
  - Ein kleiner Anteil deutet darauf hin, dass die Antwort mit den Prädiktoren zusammenhängt, und wir verwerfen die Annahme keiner Beziehung.

- **Schätzung des Anteils:**
  - Dieser Anteil wird durch einen p-Wert geschätzt, der auf der Annahme normaler Fehler basiert.

- **Permutationstests für einen Prädiktor:**
  - Anstatt die Antwortvariable zu permutieren, wird eine der Prädiktorvariablen permutiert.

# 5. Konfidenzintervalle für $\beta$

- **Konfidenzintervalle:**
  - Liefern einen Bereich `plausibler Werte` für die `Regressionskoeffizienten` ($\beta$s).
  - Geben die `Unsicherheit` über die geschätzten Koeffizienten an.
  - Basieren auf den `Daten` und dem gewählten `Konfidenzniveau`.

- **95%-Konfidenzintervall:**
  - Bedeutet, dass bei einer wiederholten `Stichprobenziehung` aus derselben `Grundgesamtheit` 95% der berechneten `Konfidenzintervalle` den `wahren Wert` des Parameters enthalten würden.

- **Berechnung des Konfidenzintervalls:**
  - Konfidenzintervall: $\hat{\beta}_i \pm t_{\alpha/2, t-n} se(\hat{\beta})$

  - Bestandteile:
    - $\hat{\beta}_i$: Geschätzter `Regressionskoeffizient` für den `i-ten Prädiktor`.
    - $t_{\alpha/2, t-n}$: Kritischer Wert aus der `t-Verteilung` für das gewählte `Konfidenzniveau` (1-$\alpha$) und die `Freiheitsgrade` (n-p). 
      - n: `Stichprobengröße`
      - p: Anzahl der `Parameter` im Modell.
    - $se(\hat{\beta})$: `Standardfehler` des geschätzten Koeffizienten.
- **Interpretation des Konfidenzintervalls:**
  - Wenn das `Konfidenzintervall` den Wert `Null` **nicht** enthält, ist der entsprechende `Prädiktor` statistisch `signifikant` auf dem gewählten `Signifikanzniveau`.
  - Die `Breite` des Konfidenzintervalls gibt die `Präzision` der Schätzung an.
    - Ein `breiteres Intervall` weist auf eine größere `Unsicherheit` hin.
- **Vorteile der Verwendung von Konfidenzintervallen:**
  - Konfidenzintervalle liefern mehr `Informationen` als nur die `Punktschätzung` des Koeffizienten.
  - Sie ermöglichen eine bessere Beurteilung der `praktischen Signifikanz` eines Prädiktors.

## Bootstrap-Konfidenzintervalle

- **Bootstrap-Methode:**
  - Eine `Resampling-Technik`, die `Konfidenzintervalle` ohne Annahme einer `Normalverteilung` ermöglicht.

- **Schritte zur Berechnung von Bootstrap-Konfidenzintervallen:**
  1. Neue `Residuen` ziehen:
     - Aus den beobachteten Residuen $\hat{e}_1, \cdots, \hat{e}_n$ werden mit Zurücklegen neue Residuen $\mathbf{e}^*$ gezogen.
  2. Neue `Antwortvariable` erstellen:
     - Neue Residuen und ursprüngliche Prädiktoren nutzen, um $\mathbf{y}^* = \mathbf{X\hat{\beta}} + \mathbf{e}^*$ zu erzeugen.
  3. Neues `Modell` schätzen:
     - Mit $\mathbf{X}$ und $\mathbf{y}^*$ wird ein neues Modell geschätzt, und der Koeffizient $\mathbf{\hat{\beta}}$ berechnet.
  4. Schritte 1-3 `wiederholen`:
     - Diese Schritte werden viele Male (z.B. 4000 Mal) wiederholt.
  5. `Konfidenzintervall` bestimmen:
     - Das `95%-Konfidenzintervall` ergibt sich aus den `2.5%-` und `97.5%-Quantilen` der Bootstrap-Verteilung der Koeffizienten.

- **Vorteile der Bootstrap-Methode:**
  - Keine `Normalverteilungsannahme` erforderlich.
  - Geeignet für `komplexe Modelle`, bei denen die theoretische Berechnung von Konfidenzintervallen schwierig ist.

# 6. Diagnose

- **Diagnose von Regressionsmodellen:**
  - Überprüfung der Modellannahmen und Identifizierung potenzieller Probleme.

- **6.1. Konstante Varianz (Homoskedastizität):**
  - Eine wichtige Annahme der linearen Regression ist die `Homoskedastizität`.
  - Die Varianz der Residuen sollte über den gesamten Wertebereich der Prädiktoren konstant sein.

- **Überprüfung der Homoskedastizität:**
  - **Streudiagramm der Residuen gegen die angepassten Werte:**
    - Wenn die Punkte zufällig um die horizontale Linie bei Null streuen und keine systematischen Muster zeigen, deutet dies auf konstante Varianz hin.

- **Verletzung der Homoskedastizität:**
  - Wenn die Varianz der Residuen nicht konstant ist (`Heteroskedastizität`):
    - Die Standardfehler der Koeffizienten sind verzerrt.
    - Die Hypothesentests sind ungültig.
  - Eine Transformation der Antwortvariablen (z.B. Wurzel- oder Logarithmustransformation) kann helfen, die Homoskedastizität herzustellen.

- **Beispiel:**
  - Streudiagramm der Residuen gegen die angepassten Werte für ein Modell zur Vorhersage der Artenvielfalt auf den Galapagos-Inseln.
  - Das Diagramm zeigt eine `zunehmende Varianz der Residuen` mit steigenden angepassten Werten.
  - Eine `Wurzeltransformation der Antwortvariablen` verbessert die Homoskedastizität.

# 7. Robuste Regression

- **Least-Squares-Regression:**
  - Funktioniert am besten mit normalverteilten Fehlern.
  - Langschwänzige Fehler können problematisch sein.

- **Umgang mit extremen Werten:**
  - Ursache verstehen: Fehler entfernen, echte Beobachtungen behalten.
  - Robuste Regression schätzt $\mathbb{E}(y) = X\beta$, unabhängig von Ausreißern.

- **Ausreißererkennung:**
  - Methoden zur Erkennung und Entfernung vor der Regression.
  - Bei mehreren Ausreißern ist robuste Regression besser.

- **M-Schätzung:**
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
  - M-Schätzungen erfordern iterative Schritte.
  - Wechsel zwischen Weighted Least-Squares und Neuberechnung der Gewichte, bis Konvergenz erreicht ist.
  - Standardfehler durch WLS mit $\widehat{\text{var}}(\hat{\beta}) = \hat{\sigma}^2 (X^\prime W X)^{-1}$ und einer robusten Schätzung von $\sigma^2$.

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





