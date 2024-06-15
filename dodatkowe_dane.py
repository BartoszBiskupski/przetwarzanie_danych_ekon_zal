import numpy as np
import pandas as pd

num_records = 40
dane = { 'ID': range(101, 101 + num_records), 'Wiek': np.random.randint(18, 65, size=num_records),
'Płeć': np.random.choice(['M', 'K'], size=num_records), 'Wykształcenie': np.random.choice(['Podstawowe',
'Średnie', 'Wyższe', 'Doktorat'], size=num_records), 'Roczny_dochód': np.random.randint(20000, 120000,
size=num_records), 'Wydatki_mieszkanie': np.random.randint(5000, 25000, size=num_records),
'Wydatki_żywność': np.random.randint(2000, 10000, size=num_records), 'Wydatki_transport':
np.random.randint(1000, 7000, size=num_records), 'Wydatki_edukacja': np.random.randint(500, 5000,
size=num_records), 'Kredyty': np.random.randint(0, 50000, size=num_records), 'Karty_kredytowe':
np.random.randint(0, 10000, size=num_records), 'Osoby_zależne': np.random.randint(0, 4,
size=num_records), 'Oszczędności': np.random.randint(0, 50000, size=num_records), 'Inwestycje_akcje':
np.random.randint(0, 20000, size=num_records), 'Inwestycje_nieruchomości': np.random.randint(0, 60000,
size=num_records), 'Stanowisko': np.random.choice(['Analityk', 'Kierownik', 'Inżynier', 'Specjalista',
'Profesor'], size=num_records), 'Branża': np.random.choice(['Finanse', 'Handel', 'Technologia', 'Marketing',
'Edukacja'], size=num_records), 'Lata_doświadczenia': np.random.randint(0, 40, size=num_records)}


df = pd.DataFrame(dane)
df.to_csv("dodatkowe_dane.csv", index=False)