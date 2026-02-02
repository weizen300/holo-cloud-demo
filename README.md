# HoloPython Demo - Cloud Particle Analyzer

Demo-Anwendung zur Analyse holografischer Aufnahmen von Wolkentröpfchen und Eiskristallen.

## Schnellstart

```bash
# Virtuelle Umgebung aktivieren
source .venv/bin/activate

# Anwendung starten
python holo_analyzer_demo.py
```

## Installation (falls .venv nicht existiert)

```bash
# Python 3.9-3.12 empfohlen (tkinter-Unterstützung)
python3 -m venv .venv
source .venv/bin/activate
pip install numpy scipy matplotlib zarr
```

**Hinweis:** Python 3.14 (Homebrew) hat oft kein tkinter. Falls die GUI nicht startet:
```bash
# Alternative mit Python 3.12
python3.12 -m venv .venv
source .venv/bin/activate
pip install numpy scipy matplotlib zarr
```

## Funktionen

| Button | Funktion |
|--------|----------|
| **Daten laden** | Lädt .npy oder .zarr Dateien |
| **Neu generieren** | Erzeugt neues synthetisches Hologramm |
| **Analysieren** | Führt Partikelerkennung durch |
| **Export (Zarr)** | Speichert Daten + Metadaten |
| **Größenverteilung** | Zeigt Histogramm der Partikelgrößen |

Der **Schwellwert-Slider** passt die Segmentierungsempfindlichkeit an.

## Technische Details

### Verwendete Bibliotheken
- `numpy` - Array-Operationen
- `scipy.ndimage` - Bildverarbeitung (label, find_objects, median_filter)
- `matplotlib` - Visualisierung
- `zarr` - Datenexport
- `tkinter` - GUI

### Analyse-Pipeline
1. **Preprocessing**: Median-Filter zur Hintergrund-Subtraktion
2. **Segmentierung**: Binäres Thresholding + Connected-Component-Labeling
3. **Merkmalsextraktion**: Zentroid, Fläche, Durchmesser, Intensität

### Code-Struktur
```
holo_analyzer_demo.py
├── HologramSimulator    # Synthetische Daten mit Beugungsmustern
├── HologramAnalyzer     # Analyse-Pipeline
└── HoloAnalyzerGUI      # Tkinter-Oberfläche
```

## Features

- **Dunkles Theme** - Modernes, augenschonendes Design
- **Echtzeit-Analyse** - Schwellwert-Änderungen werden sofort angewendet
- **Colorbar** - Intensitätsskala im Plot
- **Statistik-Panel** - Zusammenfassung der Partikelmerkmale
- **Partikel-Highlighting** - Klick auf Tabelle hebt Partikel im Plot hervor
- **Vorverarbeitung anzeigen** - Toggle zwischen Original und preprocessed
- **Größenverteilung** - Histogramm mit Statistik (Mittelwert, Median, Standardabweichung)
