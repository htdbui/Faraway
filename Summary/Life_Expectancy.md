---
title: "Life Expectancy"
author: "db"
---

# 1. Datenübersicht

- **Einführung in den Datensatz:**
  - Der Datensatz enthält Informationen über die 50 Bundesstaaten der USA.

- **Wichtige Variablen:**
  - `Population`: Einwohnerzahl in Tausend
  - `Income`: Pro-Kopf-Einkommen
  - `Illiteracy`: Analphabetenrate
  - `LifeExp`: Durchschnittliche Lebenserwartung
  - `Murder`: Mordrate pro 100.000 Einwohner
  - `HSGrad`: Prozentsatz der Personen mit High-School-Abschluss
  - `Frost`: Durchschnittliche Anzahl der Frosttage pro Jahr
  - `Area`: Landfläche in Quadratmeilen

# 3.0 Lineare Regressionsanalyse

- **Definition und Zweck:**
  - `Lineare Regression` ist eine statistische Methode.
  - Modelliert die Beziehung zwischen einer `abhängigen Variable` und einer oder mehreren `unabhängigen Variablen`.
  - Ziel ist es, eine Gleichung zur Vorhersage der `abhängigen Variable` zu finden.

- **Annahmen:**
  - `Linearität`: Lineare Beziehung zwischen `abhängigen` und `unabhängigen Variablen`.
  - `Unabhängigkeit`: Beobachtungen sind unabhängig voneinander.
  - `Homoskedastizität`: Konstante Varianz der Fehlerterme.
  - `Normalität`: Fehlerterme sind normalverteilt.

- Einige der `Koeffizienten` stimmen mit unseren Erwartungen überein. 
  - Höhere `Mordraten` verringern die `Lebenserwartung`.
- Einige Variablen zeigen keinen signifikanten Einfluss.
  - Zum Beispiel hat `Einkommen` keinen signifikanten Einfluss, was überraschend ist.

# 3.1 Hypothesentests-basierte Eliminierungsverfahren

- **Rückwärts-Elimination:**
  - Wählen von Modellvariablen ohne spezielle Software.
  - Start mit allen `Prädiktoren` im Modell.
  - Entfernen des `Prädiktors` mit dem höchsten `p-Wert` über $α_{crit}$.
  - Wiederholen bis nur signifikante `Prädiktoren` verbleiben.
  - $α_{crit}$ kann flexibel sein (z.B. 15%-20% für bessere Vorhersagen).

- **Vorwärts-Selektion:**
  - Start mit leerem Modell.
  - Hinzufügen von `Prädiktoren` mit niedrigsten `p-Werten` unter $α_{crit}$.
  - Fortsetzen bis keine weiteren `Prädiktoren` hinzugefügt werden können.

- **Schrittweise Regression:**
  - Kombination von Rückwärts-Elimination und Vorwärts-Selektion.
  - Variablen werden je nach Bedarf hinzugefügt oder entfernt.

- **Nachteile der testbasierten Verfahren:**
  - Einzelnes Hinzufügen/Entfernen kann beste Kombination übersehen.
  - Vorsicht bei der Interpretation von `p-Werten`.
  - Variablenauswahl nicht immer zielgerichtet für Vorhersage oder Erklärung.
  - Schrittweise Auswahl führt oft zu kleineren Modellen als optimal.

- **Allgemeine Hinweise:**
  - Testbasierte Auswahl nur in einfachen Fällen oder stark strukturierten Modellen nutzen.
  - Entfernen von Variablen kann Modellinterpretation erleichtern.
  - Geringer Rückgang von `R²` (z.B. von 0.74 auf 0.71) zeigt geringen Einfluss auf Modellgüte.
  - Ausgelassene Variablen könnten dennoch relevant sein.

- **Beispiel:**
  - `Analphabetismus` beeinflusst `Lebenserwartung`.
  - Ersatz durch `High-School-Abschlussrate` verbessert Modell geringfügig.
  - Verfahren unterscheiden nicht immer klar zwischen signifikanten und insignifikanten `Prädiktoren`.

# 3.2 Kriterienbasierte Verfahren

- **Einführung:**
  - `Kriterienbasierte Verfahren` wählen Variablen und Modelle durch Optimierung eines Kriteriums.
  - Diese Methoden berücksichtigen die Güte der Anpassung und die Modellkomplexität.

- **Modellevaluierung:**
  - Wähle ein Modell `g` mit Parametern `θ`, das dem wahren Modell `f` nahekommt.
  - Verwende die Kullback-Leibler-Distanz als Maß.
  - Diese Distanz ist positiv, außer `g` ist genau `f`.

- **Kriterien und Anwendungen:**
  - `AIC`: Bevorzugt Modelle mit guter Anpassung und begrenzter Parameteranzahl. Häufig für `Vorhersagezwecke` genutzt.
    - Berechnung:
      $$ AIC = -2\log{L(\hat{\theta})} + 2p $$
    - `L(θ)` ist die maximale Likelihood, `p` die Anzahl der Parameter.
    - Ein kleinerer `AIC`-Wert zeigt ein besseres Modell.
  - `BIC`: Ähnlich wie AIC, aber bestraft komplexere Modelle stärker. Ideal zur `Identifizierung des "wahren" Modells`.
    - Berechnung:
      $$ BIC = -2\log{L(\hat{\theta})} + p\log{n} $$
    - `BIC` bevorzugt kleinere Modelle mehr als `AIC`.
  - `Angepasstes R²`: Misst die erklärte Varianz, angepasst an die Anzahl der Prädiktoren. Nutzt man zur Beurteilung der `Güte der Anpassung`.
    - Kleinere `RSS` (Residual Sum of Squares) sind bevorzugt.

- **Modellauswahl Schritte:**
  1. Anpassen verschiedener Modelle mit unterschiedlichen Prädiktoren.
  2. Berechnen des gewählten Kriteriums (`AIC`, `BIC`, oder `angepasstes R²`) für jedes Modell.
  3. Wählen des Modells mit dem niedrigsten Kriteriumswert.

- **Bias-Varianz-Trade-off:**
  - Ein zu einfaches Modell führt zu Verzerrung.
  - Ein zu komplexes Modell führt zu Überanpassung.
  - `AIC` hilft, diesen Trade-off zu balancieren.

# 3.3 RFE (Rekursive Merkmalseliminierung)

- **Definition:**
  - `RFE` ist eine Methode zur Merkmalselektion.
  - Bewertet Merkmale anhand ihrer `Koeffizienten` im `linearen Regressionsmodell`.

- **Funktionsweise:**
  1. Anpassen eines `linearen Regressionsmodells` mit allen Merkmalen.
  2. Berechnung der `Koeffizienten`.
  3. Entfernen des Merkmals mit der geringsten Bedeutung.
  4. Wiederholen bis zur gewünschten Anzahl an Merkmalen.

- **Vorteile:**
  - Systematische Merkmalselektion basierend auf `prädiktiver Leistung`.
  - Entfernt irrelevante Merkmale, verbessert Generalisierbarkeit.

- **Modellauswahl:**
  - Kombiniere `RFE` mit `AIC` zur Auswahl des besten Modells.

- **Überlegungen:**
  - `RFE`-Leistung kann durch Ausreißer beeinflusst werden.
  - Untersuche Daten auf Ausreißer vor der Anwendung.

- **Effizientere Methode:**
  - Erstelle Modelle mit unterschiedlicher Anzahl an `Prädiktoren`.
  - Verwende `scikit-learn` zum Standardisieren der Variablen.

- **Einflüsse:**
  - Ausreißer beeinflussen die Variablenauswahl.
  - Transformiere schiefe Variablen durch Log-Transformation.

- **Empfehlung:**
  - Beginne mit einem Modell mit allen `Prädiktoren`.
  - Führe Regressionsdiagnosen durch.
  - Behandle ungewöhnliche Punkte vor der Modellauswahl.
