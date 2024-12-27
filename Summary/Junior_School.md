---
title: "Junior School"
author: "db"
---

# 1. Datenüberblick

- **Einführung:**
  - Daten vom Junior School Project (Mortimore et al., 1988).
  - Mathematiktestergebnisse von Schülern aus 49 Londoner Schulen.

- **Wichtige Variablen:**
  - `school`: Code für 50 Schulen (1-50). **(kategorisch)**
  - `math`: Punktzahl im Mathe-Test. **(kontinuierlich)**
  - `mathcent`: Zentrierte Mathematikpunktzahl (Punktzahl minus Durchschnitt). **(kontinuierlich)**

# 3. False Discovery Rate (FDR)

- **Definition:**
  - `FDR` ist der erwartete Anteil falscher Positiver unter den als signifikant deklarierten Ergebnissen.
  - Wichtig bei mehreren Hypothesentests, da die Wahrscheinlichkeit von Typ-I-Fehlern steigt.

- **T-Tests:**
  - Prüfen Unterschiede vom Durchschnitt.
  - Schulen 1 und 50 sind unterdurchschnittlich.
  - Schule 2 ist nicht signifikant überdurchschnittlich.

- **Signifikante Unterschiede:**
  - Große Stichprobe zeigt erwartete Unterschiede.
  - Interessanter: Welche Schulen über- oder unterdurchschnittlich abschneiden?

- **Vergleiche zwischen Schulen:**
  - Zu viele paarweise Vergleiche.
  - Fokus auf Schulen mit signifikant abweichenden Mittelwerten.

- **Korrekturmethoden:**
  - **FWER (Familywise Error Rate):**
    - Kontrolle der Wahrscheinlichkeit für mindestens einen Fehler.
    - Bonferroni-Korrektur: Multiplikation der p-Werte mit der Anzahl der Vergleiche.
    - Acht Schulen identifiziert, nur Schule 31 nicht überdurchschnittlich.

- **FDR-Kontrolle:**
  - Weniger streng als FWER.
  - **Benjamini-Hochberg-Verfahren:**
    - Sortiere `p-Werte`.
    - Finde größten Index `i`, wo `p(i) <= α * (i/m)`.
    - Tests bis zu diesem Index sind signifikant.
  - Achtzehn Schulen identifiziert, mehr als bei FWER.
  - Häufig genutzt in Bildgebung und Bioinformatik.

- **Code-Snippet (Python):**
  ```python
  from statsmodels.stats.multitest import multipletests
  # FDR-Kontrolle mit Benjamini-Hochberg-Verfahren
  reject, padj, _, _ = multipletests(lmod.pvalues, method="fdr_bh")
  # Anzeigen der signifikanten Modellparameter
  lmod.params[reject]
  ```
  - Verwendet `multipletests` aus `statsmodels` zur FDR-Kontrolle.
  - `lmod` enthält Ergebnisse eines linearen Regressionsmodells.