# BetAI
Eine Website die Fußball Vorhersagen mit hilfe einen RNN macht. 
Eine Website die Fußball Vorhersagen mit hilfe eines Random Forest Models macht. 

## Inhalt Ziele
- Keine Anmeldung
- Startseite mit den alten vorhersagen und abgleich Top 5
- Wettvorhersage nur sortierbar nach Liga und nur für nächsten Spieltag
  - Vielleicht Ausgabe der Vorhersage mit 1/2 begründungen
  - Sieg/ Niederlage/ Unentschieden Über/Unter 2,5
## Checklist Reihenfolge
- Model (Random Forest oder RNN) + Head to Head + Performance letzter 5/10 Spiele
- Download und Predictions Takten
  - Montags 12 Uhr  --> 2024_25.csv
  - Montags 13 Uhr --> Results.csv (Prediction letzter Woche abgleich mit 2024_25)
  - Montags/Dienstag/Mittwoch/Donnerstag 14 Uhr --> prediction.csv
- Flask Screens:
  - Results
  - Prediction
  - Analytics
  - FAQ
- Webiste online bringen