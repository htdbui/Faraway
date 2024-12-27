---
title: "Meat"
author: "db"
---

### 1. Beschreibung der Daten

- Der Datensatz enthält Informationen über Fleischproben.
- Er besteht aus `215` Beobachtungen und `101` Variablen.

* **Fettgehalt:**
  - Repräsentiert den Fettgehalt jeder Fleischprobe.
  - Zielvariable, die wir vorhersagen möchten.

* **100-Kanal-Absorptionsspektrum:**
  - Jede Beobachtung hat `100` Messwerte.
  - Stellt das Absorptionsspektrum des Fleisches bei verschiedenen Wellenlängen dar.
  - Diese `100` Variablen sind Prädiktoren zur Vorhersage des Fettgehalts.

### 3. Lineare Regression

- **Lineare Regression:**
  - Grundlegendes statistisches Verfahren.
  - Modelliert die Beziehung zwischen einer Zielvariablen (`Fettgehalt`) und Prädiktorvariablen (`100 Absorptionswerte`).
  - Ziel: Eine lineare Gleichung zur Vorhersage zukünftiger Werte finden.

- **Modellbewertung:**
  - **Trainingsstichprobe:**
    - Erste `172` Beobachtungen zum Trainieren des Modells.
  - **Teststichprobe:**
    - Restliche `43` Beobachtungen zur Leistungsbewertung.
  - **R²:**
    - Misst, wie gut das Modell die Varianz erklärt.
    - Höherer Wert (0 bis 1) deutet auf bessere Anpassung hin.
  - **Root Mean Squared Error (RMSE):**
    - Quantifiziert den durchschnittlichen Unterschied zwischen vorhergesagten und tatsächlichen Werten.
    - Niedrigerer Wert zeigt bessere Vorhersagegenauigkeit an.

- **Wichtige Punkte:**
  - Leistung des Modells ist im Trainingssatz oft besser.
  - Bewertung am unabhängigen Testsatz wichtig für Generalisierungsfähigkeit.

### 4. Rekursive Merkmalseliminierung

- **Rekursive Merkmalseliminierung (`RFE`):**
  - Technik zur Identifikation wichtiger Prädiktoren.
  - Entfernt iterativ das am wenigsten wichtige Merkmal.
  - Modell wird neu trainiert, bis gewünschte Anzahl von Merkmalen erreicht ist.

- **Fleischspektroskopiedaten:**
  - `RFE` reduziert die Anzahl der Prädiktoren.
  - Benachbarte Frequenzen im Spektrum sind oft hoch korreliert.
  - Eliminierung von `81` Variablen verbessert die Modellleistung in der Teststichprobe.
  - Anpassung im Trainingssatz nimmt leicht ab.

### 5. Hauptkomponentenanalyse (PCA)

- **PCA:**
  - Technik zur Dimensionsreduktion.
  - Reduziert hochdimensionale Daten auf wenige unkorrelierte Variablen (`Hauptkomponenten`).
  - PCs sind lineare Kombinationen der ursprünglichen Prädiktoren.
  - Erklären den größten Teil der Varianz in den Daten.

- **Anwendung auf Fleischspektroskopiedaten:**
  - Erste PC erklärt den größten Teil der Varianz.
  - Wenige Dimensionen erfassen die Variabilität.
  - Erste vier PCs für Regressionsmodell verwendet.
  - Modell liefert vergleichbare Ergebnisse wie mit 100 Prädiktoren.
  - PCA-Koeffizienten (`Loadings`):
    - Erste PC erfasst allgemeinen Pegel der Absorptionswerte.
    - Zweite PC gewichtet höhere und niedrigere Frequenzen gegeneinander.

- **Vorteile der PCA:**
  - Reduziert Dimensionalität und vereinfacht Modellierung.
  - Deckt versteckte Muster auf.
  - Verbessert Modellleistung durch Reduktion der Multikollinearität.

### 6. Partielle kleinste Quadrate (PLS)

- **`PLS`** ist eine Regressionstechnik.
  - Verwandt mit `PCA`.
  - Maximiert die Beziehung zwischen Prädiktoren und Zielvariablen.
  - `PCA` findet Komponenten (PCs), die die meiste Varianz in Prädiktoren erklären.
  - `PLS` findet Komponenten (latente Variablen), die die Kovarianz zwischen Prädiktoren und Zielvariablen maximieren.

- **Anwendung auf die Fleischspektroskopiedaten:**
  - Ein `PLS`-Modell mit vier Komponenten wurde erstellt.
  - `PLS` lieferte bessere Vorhersageleistung als `PCA` mit vier Komponenten.

- **Vorteile der `PLS`:**
  - Nützlich bei großer Anzahl an Prädiktoren oder starker Korrelation zwischen Prädiktoren.
  - Kann für Dimensionsreduktion und Vorhersage verwendet werden.

### 7. Ridge-Regression

- **`Ridge-Regression`** ist eine Regularisierungstechnik.
  - Verwendet, um Koeffizienten eines linearen Regressionsmodells zu schrumpfen.
  - Fügt der Verlustfunktion einen Strafterm hinzu.
  - Dieser Strafterm ist proportional zur Summe der quadrierten Koeffizienten.
  - Verhindert, dass Koeffizienten zu groß werden.
  - Hilft, Überanpassung zu vermeiden.

- **Wichtige Punkte:**
  - `Ridge-Regression` reduziert die Variabilität der Modellschätzungen durch Schrumpfung.
  - Die Stärke der Schrumpfung wird durch den Parameter `a` gesteuert.
  - Besonders nützlich bei Multikollinearität in den Daten.
  - Die Koeffizienten sind verzerrt.
  - Kann zu einem niedrigeren mittleren quadratischen Fehler (`MSE`) führen.

### 8. Lasso

- **`Lasso` (Least Absolute Shrinkage and Selection Operator)** ist eine Regularisierungstechnik.
  - Verwendet den Absolutwert der Koeffizienten als Strafterm.
  - Kann einige Koeffizienten auf genau Null setzen.
  - Effektiv für die Variablenauswahl.

- **Wichtige Punkte:**
  - `Lasso` führt Schrumpfung und Variablenauswahl durch.
  - Nützlich bei vielen Prädiktoren, von denen nur einige relevant sind.
  - Bessere Leistung als `Ridge-Regression`, wenn die Anzahl der Prädiktoren die Anzahl der Beobachtungen übersteigt.

### 9. Andere Methoden

Zusätzlich zu den oben beschriebenen Methoden gibt es viele andere Regressions- und maschinelle Lerntechniken zur Vorhersage einer kontinuierlichen Zielvariablen.

- **Beispiele:**
  - **`Elastic Net`**: Kombiniert Strafen von `Ridge` und `Lasso`.
  - **`Least Angle Regression (LARS)`**: Effiziente Methode zur Anpassung von `Lasso`-Modellen.
  - **`Orthogonal Matching Pursuit`**: Gieriger Algorithmus, der iterativ Prädiktoren hinzufügt.
  - **`Support Vector Machines (SVMs)`**: Für lineare und nichtlineare Regression.
  - **`Neuronale Netze`**: Erfassen komplexe nichtlineare Beziehungen.
  - **`Random Forests`**: Kombinieren mehrere Entscheidungsbäume.

- Die Wahl der besten Methode hängt von den Eigenschaften des Datensatzes und den Analysezielen ab.