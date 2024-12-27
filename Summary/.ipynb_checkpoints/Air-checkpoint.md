---
title: "Air"
author: "db"
---

# 1. Datenbeschreibung

*   Die Daten enthalten die monatliche Anzahl der Flugpassagiere aus den frühen Jahren des Flugverkehrs.
*   Es gibt 144 Beobachtungen.
*   Es gibt 2 Variablen:
    *   **pass:** Anzahl der Passagiere
    *   **Jahr:** Zeitpunkt in einem bestimmten Format (Januar = 0.083, Februar = 0.167, März = 0.250, April = 0.333, Mai = 0.417, Juni = 0.500, Juli = 0.583, August = 0.667, September = 0.750, Oktober = 0.833, November = 0.917, Dezember = 1.000) 

# 2. Pakete und Daten laden

*   Zuerst müssen die notwendigen Pakete importiert werden, darunter pandas, numpy, scipy, matplotlib.pyplot, statsmodels.api und statsmodels.formula.api.
*   Dann werden die Daten mit `faraway.datasets.air.load()` geladen.
*   Die Spalte "pass" wird in "passengers" umbenannt, da "pass" ein Schlüsselwort ist.
*   Die ersten paar Zeilen der Daten können mit `air.head()` angezeigt werden.
*   Ein Diagramm der Passagierzahlen im Laufe der Zeit kann mit `plt.plot(air['year'], air['passengers'])` erstellt werden.

# 3. Lineare Regression

*   **Ziel:** Eine lineare Beziehung zwischen der Zeit und dem Logarithmus der Passagierzahlen modellieren.
*   **Elemente für die Regression erstellen:**
    *   **X:** Eine Datenmatrix mit einem Achsenabschnitt und dem Jahr.
    *   **y:** Der Logarithmus der Passagierzahlen.
*   **Regression durchführen:**
    *   Dies kann mit `sm.OLS(y,X).fit()` aus `statsmodels.api` oder mit `smf.ols(formula='np.log(passengers) ~ year', data=air).fit()` aus `statsmodels.formula.api` erfolgen. 
*   **Ergebnisse ausgeben:** Die Zusammenfassung des Modells kann mit `lmod.summary()` oder `lmod2.summary()` angezeigt werden.
*   **Modell visualisieren:** Ein Diagramm der angepassten Linie und der tatsächlichen Daten kann mit `plt.plot(air['year'], air['passengers'])` und `plt.plot(air['year'],np.exp(lmod.predict()))` erstellt werden. 

## 3.1 Autoregression

*   **Ziel:** Die Passagierzahlen mit Verzögerungen der Passagierzahlen der vorherigen Monate modellieren.
*   **Modell:**  
    $y_t = β_0 + β_1 y_{t-1} + β_{12} y_{t-12} + β_{13} y_{t-13} + \epsilon_t$
    *   wobei $y_t$ die Passagierzahl im Monat $t$ ist.
    *   $y_{t-1}$, $y_{t-12}$ und $y_{t-13}$ sind die Passagierzahlen ein, zwölf und dreizehn Monate vor dem Monat $t$.
*   **Spalten mit Verzögerungen erstellen:** Neue Spalten im Datensatz werden für die Verzögerungen erstellt.
*   **Elemente für die Regression erstellen:**
    *   **X:** Eine Datenmatrix mit einem Achsenabschnitt und den Verzögerungen.
    *   **y:** Der Logarithmus der Passagierzahlen.
*   **Regression durchführen:** Die Regression wird mit `sm.OLS(y,X).fit()` durchgeführt.
*   **Ergebnisse ausgeben:** Die Zusammenfassung des Modells kann mit `lmod.summary()` angezeigt werden.
*   **Modell visualisieren:** Ein Diagramm der angepassten Linie und der tatsächlichen Daten kann mit `plt.plot(air['year'], air['passengers'])` und `plt.plot(airlag['year'],np.exp(lmod.predict()),linestyle='dashed')` erstellt werden. 

# 4. Vorhersage

*   **Ziel:** Die Passagierzahl des nächsten Monats vorhersagen.
*   **Werte der Verzögerungen abrufen:** Die Werte der Verzögerungen aus dem letzten Datenpunkt werden abgerufen.
*   **Element für die Vorhersage erstellen:** Ein Datenrahmen mit dem Achsenabschnitt und den Verzögerungen wird erstellt.
*   **Vorhersage durchführen:** Die Vorhersage wird mit `lmod.get_prediction(x0).summary_frame()` durchgeführt. 
