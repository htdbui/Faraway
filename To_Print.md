## Lernhilfe zur Vorhersage des Fettgehalts von Fleisch anhand von Absorbanzspektren

### 1. Beschreibung der Daten

Der Datensatz enthält 215 Beobachtungen mit 101 Variablen. Die Variablen sind:

*   Fettgehalt
*   100-Kanal-Absorbanzspektrum

### 2. Pakete und Daten laden

```python
import pandas as pd
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

import statsmodels.api as sm
import statsmodels.formula.api as smf
import faraway.utils
import faraway.datasets.meatspec

meatspec = faraway.datasets.meatspec.load()
meatspec.head()
meatspec.describe().round(1)
```

### 3. Lineare Regression

*   Um die Leistung eines Modells zu beurteilen, muss es an neuen Daten getestet werden. Daher wird der Datensatz in zwei Teile aufgeteilt: eine Trainingsstichprobe mit den ersten 172 Beobachtungen zum Erstellen der Modelle und eine Teststichprobe mit den restlichen 43 Beobachtungen zur Bewertung. Die Teststichprobe wird nur zur Bewertung verwendet, nicht zur Modellauswahl.
*   Ein guter Ansatz ist es, mit einem linearen Modell zu beginnen, das alle Prädiktoren verwendet. Die `LinearRegression`-Funktion aus dem Modul `linear_model` in scikit-learn wird verwendet. Obwohl sie weniger Funktionen als statsmodels hat, lässt sie sich gut in andere scikit-learn-Funktionen integrieren, was sie bequemer macht. Der Score der Funktion ist der R$^2$-Wert.
*   Die Anpassung des Modells ist in Bezug auf R$^2$ sehr gut. Um die Vorhersageleistung der Teststichprobe zu bewerten, wird der Root Mean Square Error (RMSE) verwendet:
    $$ \sqrt{ \sum_{i=1}^n {(\hat{y}_i - y_i)}^2 / n } $$
*   RMSE wird gegenüber MSE bevorzugt, da RMSE in den gleichen Einheiten wie die Antwortvariable vorliegt, was die Interpretation erleichtert.
*   Die Leistung der Teststichprobe ist viel schlechter, was häufig vorkommt. Die Anpassung des Modells an die Trainingsdaten überschätzt oft die zukünftige Leistung. Hier ist der tatsächliche Fehler etwa fünfmal höher als vom Modell angegeben.
*   Es ist unwahrscheinlich, dass alle 100 Prädiktoren für eine gute Vorhersage notwendig sind. Eine Untersuchung der Prädiktoren zeigt, dass benachbarte Frequenzen hoch korreliert sind, was darauf hindeutet, dass viele möglicherweise unnötig sind. Die rekursive Merkmalseliminierung mit Kreuzvalidierung wird verwendet, um die Anzahl der Prädiktoren auszuwählen.

### 4. Rekursive Merkmalseliminierung

*   RFECV mit Kreuzvalidierung wird angewendet, um die beste Merkmalsuntergruppe zu finden.
*   RFECV() hat folgende Parameter:
    *   `redreg` ist das lineare Regressionsmodell, das für die Anpassung verwendet wird.
    *   `step=1` bedeutet, dass in jedem Schritt ein Merkmal entfernt wird.
    *   `cv=10` bedeutet, dass eine 10-fache Kreuzvalidierung durchgeführt wird, um die Leistung des Modells zu bewerten.
*   `selector.support_` ist ein boolesches Array, das angibt, welche Merkmale ausgewählt wurden.
*   Die Modellauswahl hat 81 Variablen entfernt. Während sich die nominale Anpassung leicht verschlechterte, verbesserte sich die tatsächliche Leistung von 3,86 auf 2,52.

### 5. Hauptkomponentenanalyse

*   Die PCA kann als eine bestimmte Rotation der Daten verstanden werden. Die Daten sollen um ihren Mittelwert gedreht werden. Zu diesem Zweck wird die Matrix der Prädiktoren $\mathbf{X}$ zentriert, indem der Mittelwert für jede Variable subtrahiert wird, so dass die Spalten von X den Mittelwert Null haben. Es wird ein $\mathbf{X}$ verwendet, das keine Spalte mit Einsen für einen Intercept-Term enthält. Die PC-Zerlegung kann wie folgt berechnet werden:
    *   Finden Sie $u_1$ so, dass $var(\mathbf{X} u_1)$ maximiert wird, vorbehaltlich $||u_1|| = u_1^T u_1 = 1$.
    *   Finden Sie $u_2$ so, dass $var(\mathbf{X} u_2)$ maximiert wird, vorbehaltlich $||u_2|| = 1$ und $u_2^T u_1 = 0$.
    *   Setzen Sie diesen Prozess fort, um $u_3$, $u_4$, $\ldots$ zu finden.
*   Die Hauptkomponenten $z_i = \mathbf{X} u_i$ sind unkorreliert. Die erste Hauptkomponente hat die größte Varianz. Sie ist die Linearkombination der Prädiktoren, die die meiste Varianz in den Daten erklärt.
*   Die Terme werden in der Matrixform $\mathbf{Z} = \mathbf{X} \mathbf{U}$ zusammengefasst, wobei $\mathbf{U}$ und $\mathbf{Z}$ die Spalten $u_i$ bzw. $z_i$ haben. $\mathbf{U}$ wird als Rotationsmatrix bezeichnet. $\mathbf{Z}$ ist eine Version der Daten, die auf Orthogonalität gedreht wurde.
*   `pca.explained_variance_` enthält die Varianz jeder PC mit ihren Scores.
*   Die erste Hauptkomponente (PC) erklärt etwa zehnmal mehr Variation als die zweite, wobei die Beiträge danach stark abnehmen. Dies zeigt, dass der Großteil der Variation in den Prädiktoren mit nur wenigen Dimensionen erfasst werden kann.
*   Das Attribut `components_` des angepassten PCA-Objekts wird verwendet, um die Linearkombinationen (oder Loadings) anzuzeigen.
*   Diese Vektoren stellen die Linearkombinationen von Prädiktoren dar, die die PCs erzeugen:
    *   Der erste PC ist eine fast konstante Kombination von Frequenzen, die misst, ob die Prädiktoren im Allgemeinen groß oder klein sind.
    *   Der zweite PC stellt höhere und niedrigere Frequenzen gegenüber.
    *   Der dritte PC ist schwieriger zu interpretieren.
*   Manchmal, wie in diesem Beispiel, können PCs intuitiv interpretiert werden, aber manchmal muss man sich mit einer verbesserten Vorhersage ohne klare Interpretation zufrieden geben.
*   Die ersten vier PCs werden verwendet, um die Antwort vorherzusagen.
*   `pca.fit_transform(Xtrain)` passt PCA an Xtrain an, um PCs zu erhalten. Dann transformiert es Xtrain in eine Score-Matrix (Matrix der PCs als Spalten).
*   Obwohl man mit nur vier Variablen im Vergleich zu 100 keine so gute Anpassung erwartet, ist die Anpassung immer noch mit viel größeren Modellen vergleichbar.
*   PCR ist eine Art der Schrumpfungsschätzung. Um den Begriff zu verstehen, werden die 100 Steigungskoeffizienten aus der vollständigen Kleinste-Quadrate-Anpassung dargestellt.
*   Die Koeffizienten reichen bis in die Tausende, und benachbarte Koeffizienten können sich erheblich unterscheiden. Dies ist überraschend, da man erwarten könnte, dass benachbarte Frequenzen ähnliche Auswirkungen auf die Antwort haben.
*   Die Koeffizienten aus dem Vier-Komponenten-Modell werden dargestellt.
    *   Die Auswirkungen der ursprünglichen Merkmale werden durch PC-Loadings und Regressionskoeffizienten berechnet.
    *   `pca.components_[:4,]` ist die Matrix der Größe 100 x 4. Dies sind die Loadings der ersten vier PCs.
    *   `pca.components_[:4,] %*% pc4reg.coef_` sind die Auswirkungen der ursprünglichen Merkmale.
*   Der Bereich dieser Koeffizienten ist viel kleiner als die Tausende, die bei der gewöhnlichen Kleinste-Quadrate-Anpassung beobachtet wurden, was zu stabileren Koeffizienten führt. Dieser Effekt wird als Schrumpfung bezeichnet.
*   Zusätzlich gibt es eine Glätte zwischen benachbarten Frequenzen, was mit dem wissenschaftlichen Verständnis der Daten übereinstimmt. Die Vorhersage verwendet hauptsächlich die untere Hälfte der Frequenzen.

### 6. Partial Least Squares (PLS)

**Partial Least Squares (PLS) ist eine Technik, die verwendet wird, um einen Satz von Input-Variablen  $X_1, ..., X_m$ mit einem Satz von Output-Variablen $Y_1, ..., Y_z$ in Beziehung zu setzen.** Die PLS-Regression ähnelt der PCR insofern, als beide Methoden Antworten mithilfe von Linearkombinationen der Prädiktoren vorhersagen. Der **wesentliche Unterschied besteht jedoch darin, dass die PCR diese Kombinationen bestimmt, ohne die Output-Variablen *Y* zu berücksichtigen, während die PLS-Regression sie explizit auswählt, um die Vorhersage von *Y* zu optimieren**.

Dieser Abschnitt konzentriert sich auf die univariate PLS, bei der z = 1 ist und Y ein Skalar ist. Es werden Modelle der Form gesucht: 

$$ \hat{y} = β_1 T_1 + ... + β_k T_k $$

wobei $T_i$  sich gegenseitig orthogonale Linearkombinationen der Xs sind.

Es gibt mehrere Algorithmen zur Berechnung von PLS, typischerweise durch iterative Bestimmung der $T_i$ zur Vorhersage von y unter Beibehaltung der Orthogonalität. Ein häufiger Kritikpunkt ist, dass PLS kein genau definiertes Modellierungsproblem hat, was es schwierig macht, Algorithmen theoretisch zu unterscheiden.

#### PLS-Modell mit vier Komponenten

In diesem Beispiel wird ein PLS-Modell mit vier Komponenten an die Daten zur Fleischspektroskopie angepasst. Die Prädiktoren werden nicht skaliert, da sie fast die gleiche Varianz aufweisen.

* `plsmod.coef_` gibt die Regressionskoeffizienten an, die die ursprünglichen Xtrain-Variablen mit der Zielvariablen trainmeat.fat verbinden. Dies ist ein Vektor der Größe 100 x 1.
* `plsmod.x_loadings_` zeigt, wie jede ursprüngliche X-Variable zu jeder der latenten Komponenten beiträgt. Es handelt sich um eine Matrix der Größe 100 x 4.
* `plsmod.y_loadings_` zeigt, wie die Y-Variable mit jeder der latenten Komponenten zusammenhängt. Es handelt sich um einen Vektor der Größe 1 x 4.
* `plsmod.x_scores_` stellen die ursprünglichen X-Variablen dar, die in den neuen Raum transformiert wurden, der durch die latenten Komponenten definiert ist. Diese Scores sind die Projektionen der Prädiktorvariablen auf die PLS-Komponenten und werden verwendet, um die Varianz in den Prädiktorvariablen zu erklären, die mit der Antwortvariablen zusammenhängt. Dies ist eine Matrix der Größe 172 x 4.
* `plsmod.y_scores_` stellen die Y-Variable dar, die in den neuen Raum transformiert wurde, der durch die während des PLS-Regressionsprozesses extrahierten latenten Komponenten definiert ist. Diese Scores sind die Projektionen der Antwortvariablen auf die PLS-Komponenten. Sie werden verwendet, um die Varianz in der Antwortvariablen zu erklären, die mit den Prädiktoren zusammenhängt. Es handelt sich um eine Matrix der Größe 172 x 4.
    * Diese Scores werden verwendet, um die Kovarianz zwischen den Projektionen von X und Y während des PLS-Regressionsprozesses zu maximieren.
    * Sie erfassen die relevantesten Informationen in Y, die mit der latenten Struktur von X zusammenhängen.

#### Auswertung der Vorhersagegenauigkeit

Die Vorhersageleistung des PLS-Modells ist geringfügig besser als die des PCA-Regressionsmodells mit vier Komponenten, was zu erwarten ist, da die PLS-Regression darauf ausgelegt ist, die Antwort zu modellieren. 

**Möglicherweise werden mit mehr als vier Komponenten bessere Ergebnisse erzielt.** Um dies zu testen, wird eine 10-fache Kreuzvalidierung auf der Trainingsstichprobe durchgeführt. 

Die Kreuzvalidierung ergibt, dass die beste Leistung mit 14 Komponenten erzielt wird. Mit diesem Modell wird ein RMSE von 1,88 auf der Trainingsstichprobe und ein RMSE von 2,17 auf der Teststichprobe erzielt.

#### Zusammenfassung

PLS ist eine effektive Methode zur Modellierung der Beziehung zwischen einem Satz von Prädiktorvariablen und einer Antwortvariablen. Es ist besonders nützlich, wenn die Anzahl der Prädiktoren groß ist im Verhältnis zur Stichprobengröße oder wenn die Prädiktoren stark korreliert sind. 

**Im Vergleich zur PCR ist PLS oft besser für Vorhersageprobleme geeignet, da es Linearkombinationen konstruiert, die speziell auf die Vorhersage der Antwort abzielen.**

Es ist jedoch wichtig zu beachten, dass PLS und PCR genauso empfindlich auf Annahmen reagieren wie OLS. Daher ist es wichtig, diese Überprüfungen in eine umfassende Analyse einzubeziehen.

### 7. Ridge-Regression

* Die Ridge-Regression geht davon aus, dass normalisierte Regressionskoeffizienten nicht sehr groß sein sollten, was vernünftig ist, wenn man viele Prädiktoren mit potenziellen Auswirkungen auf die Antwort hat.
* Diese Methode beinhaltet eine Schrumpfung und ist besonders effektiv, wenn die Modellmatrix kollinear ist, wodurch die Kleinste-Quadrate-Schätzungen instabil werden.
* Unter der Annahme, dass die Prädiktoren durch ihre Mittelwerte zentriert, durch ihre Standardabweichungen skaliert und die Antwort zentriert ist, minimiert die Ridge-Regressionsschätzung:
    $$ (y - X β)^T (y - X β) + α \sum_j β_j^2 $$
    * für eine Wahl von $α \geq 0 $. Der Strafterm ist  $\sum_j β_j^2$.
    * Ziel ist es, diesen Term klein zu halten. Die Ridge-Regression ist aufgrund dieses Terms eine Art der **penalisierten Regression**, auch bekannt als **Regularisierung**.
* Die Ridge-Regressionsschätzungen von βs:
    $$ \hat{β} = (X^T X + α I)^{-1} X^T y $$
    * Der Term $\alpha I$ führt einen „Ridge“ in die $X^T X$-Matrix ein, was der Methode ihren Namen gibt.
* Ein äquivalenter Ausdruck des Problems ist die Wahl von $\beta$, um zu minimieren:
    $$ (y - X β)^T (y - X β) \text{ ~~subject to~~ } \sum_{j=1}^p β_j^2 \leq t^2 $$
    * Hier spielt t die gleiche Rolle wie α. Man findet die Kleinste-Quadrate-Lösung mit einer Obergrenze für die Größe der Koeffizienten.
* Die Ridge-Regression lässt sich auch aus Bayes'scher Sicht rechtfertigen, wo eine A-priori-Verteilung kleinere Parameterwerte begünstigt.
* Während $α$ oder $t$ automatisch gewählt werden können, ist es auch ratsam, die $\hat{\beta}$-Werte gegen $α$ aufzutragen. Wählen Sie das kleinste $α$, das stabile $\beta$-Schätzungen liefert.

#### Anwendung der Ridge-Regression auf die Fleischspektroskopie-Daten 

* Wenn $\alpha = 0$ ist, entspricht dies den kleinsten Quadraten, und wenn $\alpha \to 0$ geht, geht $\hat{\beta} \to 0$. In der Praxis konzentrieren wir uns auf einen engeren Bereich von $\alpha$. 
* Es wird ein logarithmisch beabstandetes Gitter von $\alpha$-Werten zwischen $10^{-5}$ und $10^{-10}$ verwendet. Für andere Datensätze müssen Sie diesen Bereich möglicherweise anpassen, um den relevanten Bereich zu erfassen.

* Der Wert von α kann mit Kreuzvalidierung ausgewählt werden, indem ein Gitter von α-Werten durchsucht wird.
* Das Ergebnis ist besser als das PLS-Ergebnis.

* Die Ridge-Koeffizienten sind weniger glatt als die Vier-Komponenten-PCA, was für eine gute Vorhersage nicht ausreicht.
* Die optimale Schrumpfung ist moderat, und die Ridge-Koeffizienten sind viel kleiner als die des vollen linearen Modells.
* Ridge-Regressionskoeffizienten sind verzerrt, was unerwünscht, aber nicht der einzige Faktor ist. Der mittlere quadratische Fehler (MSE) misst die Schätzgenauigkeit und kann in das Quadrat der Verzerrung (systematischer Fehler) und die Varianz (Variabilität der Schätzung) aufgeteilt werden:
    $$ \mathbb{E}\left(\hat{\beta}-\beta\right)^2=\left(\mathbb{E}\left(\hat{\beta}-\beta\right)\right)^2+\mathbb{E}\left(\hat{\beta}-\mathbb{E}\left(\hat{\beta}\right)\right)^2 $$
    * Die Reduzierung der Schätzvarianz kann die Verzerrung erhöhen, aber dieser Kompromiss kann den MSE deutlich senken, so dass eine gewisse Verzerrung akzeptabel ist. Dies ist ein häufiges Problem bei der Ridge-Regression.
* Eine Studie von Frank und Friedman aus dem Jahr 1993 ergab, dass die Ridge-Regression die PCR und PLS übertraf, stellte aber fest, dass die beste Methode je nach Datensatz variiert, was es schwierig macht, einen klaren Gewinner zu ermitteln.

### 8. Lasso

* Die Lasso-Methode ähnelt der Ridge-Regression, minimiert aber eine Gleichung mit einem anderen Strafterm, um den optimalen Wert für  $\hat{β}$ auszuwählen:
    $$ \left({y}-{X\beta}\right)^\prime\left({y}-{X\beta}\right) \text{ ~~subject to~~ } \sum_{j}^{p}\left|\beta_j\right|\le t $$
    * oder äquivalent zu minimieren:
    $$ \left({y}-{X\beta}\right)^\prime\left({y}-{X\beta}\right) + \alpha \sum_j \left|\beta_j\right| $$

* Die von Tibshirani 1996 eingeführte Lasso-Methode hat keine explizite Lösung, kann aber mit der Least-Angle-Regression effizient gelöst werden, wie von Efron et al. 2004 beschrieben.
* „Lasso“ steht für „Least Angle Shrinkage and Selection Operator“.
* Der Hauptunterschied zwischen Lasso und Ridge-Regression liegt in ihren Lösungen.
    * Die Lasso-Beschränkung $L_1$  $\sum_{j}^{p}|\beta_j| \le t$ bildet ein Quadrat in zwei Dimensionen und ein Polytop in höheren Dimensionen, was oft dazu führt, dass einige Koeffizienten Null sind. Wenn $t$ zunimmt, werden mehr Variablen einbezogen und ihre Koeffizienten wachsen. Für große $t$ wird die Beschränkung redundant und liefert die Kleinste-Quadrate-Lösung.
    * Bei moderaten $t$-Werten tendieren viele $\hat{β}$ in Lasso dazu, Null zu sein, was es ideal für spärliche Effekte macht, bei denen nur wenige Prädiktoren eine Rolle spielen.
* Lasso fungiert als **Variablenselektionsmethode**, indem es Prädiktoren mit $\hat{β} = 0$ eliminiert. Im Gegensatz dazu reduziert die Ridge-Regression  $\hat{β}$, eliminiert aber keine Variablen.

#### Anwendung der Lasso-Methode auf die Fleischspektroskopie-Daten 

* Der kleine Wert von a zeigt eine sehr geringe Schrumpfung an. Es wird der Anteil der Nicht-Null-Koeffizienten in dieser Anpassung überprüft.
    * Keiner der Lasso-Koeffizienten ist Null.

* Die Ergebnisse sind schlechter als bei PCR, PLS und Ridge-Regression, aber immer noch besser als das volle Kleinste-Quadrate-Modell.
* Konvergenzprobleme könnten durch Skalieren der Daten behoben werden, was jedoch zusätzlichen Aufwand erfordert, um die Teststichprobe in ähnlicher Weise zu skalieren.
* Dieses Beispiel ist aufgrund der starken Kollinearität zwischen den Prädiktoren, die meist für die Vorhersage der Antwort nützlich sind, nicht gut für Lasso geeignet.
* Lasso zeichnet sich durch spärliche Effekte aus, wie z. B. bei Genexpressionsdaten, bei denen nur wenige Gene das Ergebnis beeinflussen.
* Es funktioniert auch, wenn die Anzahl der Prädiktoren die Anzahl der Beobachtungen übersteigt.
* Allerdings ist Lasso in Anwendungen mit nicht-spärlichen Effekten, wie z. B. vielen sozioökonomischen Beispielen, weniger effektiv.
* In diesem Beispiel verursacht die Messung aller Frequenzen wahrscheinlich keine zusätzlichen Kosten. In Anwendungen, bei denen die Erfassung zusätzlicher Prädiktoren jedoch kostspielig ist, ist Lasso als effektive Modellauswahlmethode besonders wertvoll.


### 9. Andere Methoden

* Das scikit-learn-Paket bietet verschiedene Schrumpfungsmethoden.
    * **Elastic-Net** kombiniert Ridge und Lasso, indem es sowohl L1- als auch L2-Strafen verwendet, so dass einige Prädiktoren weggelassen werden können, während die Regularisierungsvorteile von Ridge erhalten bleiben.
    * **Least Angle Regression (LARS)** bevorzugt Modelle mit weniger Prädiktoren, ähnlich wie Lasso.
    * **Orthogonal Matching Pursuit** geht noch weiter, indem es eine maximale Anzahl von Nicht-Null-Koeffizienten vorgibt.
* Das Paket enthält auch die **Bayes'sche Regression**, die schwach informative Prioris auferlegt, ähnlich wie die Ridge-Regression.

* Es gibt viele Methoden, die eine kontinuierliche Antwort aus Prädiktoren vorhersagen können, wie z. B. **neuronale Netze, Support-Vektor-Maschinen und Random Forests**. 
    * Diese Methoden verwenden jedoch keine Linearkombination von Prädiktoren.
    * Obwohl sie effektiv sind, fehlt ihnen die Interpretierbarkeit der Modellkoeffizienten linearer Methoden, die deutlich zeigen, wie neue Vorhersagen generiert werden, wenn mehr Daten gesammelt werden.

