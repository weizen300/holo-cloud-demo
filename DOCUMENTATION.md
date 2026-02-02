# HoloPython Demo - Technische Dokumentation

## Projektübersicht

Diese Demo-Anwendung simuliert und analysiert **holografische Aufnahmen von Wolkenpartikeln** (Tröpfchen und Eiskristalle). Sie wurde entwickelt, um relevante Fähigkeiten für das HoloPython-Projekt an der ETH Zürich zu demonstrieren.

### Wissenschaftlicher Hintergrund

In der Atmosphärenforschung werden holografische Kameras eingesetzt, um Wolkenpartikel zu vermessen. Wenn kohärentes Licht (Laser) auf kleine Partikel trifft, entstehen charakteristische **Beugungsmuster** - konzentrische Ringe, deren Abstände von der Partikelgröße abhängen.

```
Laser → [Wolkenpartikel] → Beugungsmuster → Kamerasensor → Hologramm
```

Die Analyse dieser Hologramme ermöglicht:
- Bestimmung von Tröpfchengrößen-Verteilungen
- Unterscheidung zwischen Eiskristallen und Wassertröpfchen
- Untersuchung von Niederschlagsbildungsprozessen

---

## Architektur

```
┌─────────────────────────────────────────────────────────────────┐
│                        holo_analyzer_demo.py                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────────────┐    ┌──────────────────────┐          │
│  │  HologramSimulator   │    │   HologramAnalyzer   │          │
│  ├──────────────────────┤    ├──────────────────────┤          │
│  │ • generate()         │    │ • analyze()          │          │
│  │ • _add_particle()    │───▶│ • _preprocess()      │          │
│  │                      │    │ • _segment()         │          │
│  │ Erzeugt synthetische │    │ • _extract_features()│          │
│  │ Hologramme mit       │    │                      │          │
│  │ Beugungsmustern      │    │ Erkennt & vermisst   │          │
│  └──────────────────────┘    │ Partikel             │          │
│                              └──────────────────────┘          │
│                                        │                        │
│                                        ▼                        │
│                         ┌──────────────────────┐               │
│                         │   HoloAnalyzerGUI    │               │
│                         ├──────────────────────┤               │
│                         │ • Visualisierung     │               │
│                         │ • Parameter-Slider   │               │
│                         │ • Statistik-Tabelle  │               │
│                         │ • Zarr-Export        │               │
│                         └──────────────────────┘               │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Komponenten im Detail

### 1. HologramSimulator

Generiert realistische synthetische Hologramme zur Demonstration.

**Algorithmus:**

```python
# 1. Hintergrund: Gaußsches Rauschen (simuliert Sensorrauschen)
hologram = np.random.normal(loc=0.5, scale=0.1, size=(256, 256))

# 2. Für jedes Partikel: Beugungsmuster hinzufügen
for particle in particles:
    distance = √((x - cx)² + (y - cy)²)  # Abstand vom Zentrum

    # Beugungsmuster: Oszillierende Intensität mit Gaußscher Einhüllenden
    pattern = intensity * exp(-distance / (radius * 2)) * cos(2π * distance / wavelength)²

    hologram += pattern
```

**Physikalische Grundlage:**

Das Beugungsmuster approximiert ein Airy-Pattern, das entsteht, wenn Licht an einer kreisförmigen Apertur (hier: kugelförmiges Partikel) gebeugt wird. Die exakte Lösung wäre eine Bessel-Funktion, aber die Gaußsche Approximation ist für Demonstrationszwecke ausreichend.

---

### 2. HologramAnalyzer

Implementiert eine robuste Bildverarbeitungs-Pipeline zur Partikelerkennung, die speziell für Beugungsmuster optimiert ist.

**Das Problem mit einfachem Thresholding:**

Beugungsmuster bestehen aus konzentrischen Ringen. Ein einfacher Threshold + Connected-Components-Ansatz würde die Ringe erkennen, nicht die Zentren:

```
Beugungsmuster        Einfaches Threshold      Problem
┌───────────┐         ┌───────────┐
│  ◯◯◯◯◯    │         │  ○○○○○    │           Ringe werden als
│ ◯     ◯   │    →    │ ○     ○   │    →      separate Objekte
│ ◯  ●  ◯   │         │ ○     ○   │           erkannt, nicht
│ ◯     ◯   │         │ ○     ○   │           das Zentrum!
│  ◯◯◯◯◯    │         │  ○○○○○    │
└───────────┘         └───────────┘
```

**Lösung: Lokale Maxima-Detektion**

#### Schritt 1: Preprocessing (Glättung)

```python
# Hintergrund entfernen
background = scipy.ndimage.median_filter(hologram, size=40)
foreground = hologram - background

# Gaußsche Glättung: Verschmilzt Ringe zu einem Peak pro Partikel
sigma = 7.5  # ~ halbe Partikelgröße
smoothed = scipy.ndimage.gaussian_filter(foreground, sigma=sigma)
```

**Visualisierung:**

```
Beugungsmuster        Nach Glättung
┌───────────┐         ┌───────────┐
│  ◯◯◯◯◯    │         │           │
│ ◯     ◯   │         │     ▓     │         Ein Peak pro
│ ◯  ●  ◯   │    →    │    ▓█▓    │    →    Partikel!
│ ◯     ◯   │         │     ▓     │
│  ◯◯◯◯◯    │         │           │
└───────────┘         └───────────┘
```

#### Schritt 2: Lokale Maxima-Detektion

```python
# Maximum-Filter: Jedes Pixel wird durch das Maximum seiner Nachbarschaft ersetzt
neighborhood_size = 25  # ~ Partikelgröße
local_max = scipy.ndimage.maximum_filter(smoothed, size=neighborhood_size)

# Ein Pixel ist lokales Maximum wenn es gleich dem Nachbarschafts-Maximum ist
is_peak = (smoothed == local_max)

# Nur Peaks über dynamischem Schwellwert (basierend auf Perzentil)
percentile = 70 + threshold * 29  # threshold 0.1→70%, 0.9→99%
intensity_threshold = np.percentile(smoothed, percentile)
centers = is_peak & (smoothed > intensity_threshold)
```

#### Schritt 3: Watershed-Segmentierung

```python
# Markiere jeden Peak mit eindeutiger ID
markers, n_particles = scipy.ndimage.label(centers)

# Watershed: "Flutet" von Markern aus, bis Regionen sich treffen
labels = scipy.ndimage.watershed_ift(inverted_image, markers)
```

**Visualisierung:**

```
Peaks (Zentren)       Watershed-Regionen
┌───────────┐         ┌───────────┐
│     1     │         │ 1111111111│
│           │         │ 1111122222│
│  2     3  │    →    │ 2222233333│
│           │         │ 2222233333│
└───────────┘         └───────────┘
```

#### Schritt 4: Merkmalsextraktion

Für jedes erkannte Partikel werden berechnet:

| Merkmal | Berechnung | Bedeutung |
|---------|------------|-----------|
| **Zentroid (x, y)** | `scipy.ndimage.center_of_mass()` | Intensitäts-gewichteter Schwerpunkt |
| **Fläche** | `np.count_nonzero(mask)` | Anzahl Pixel im Partikel |
| **Äquivalenter Durchmesser** | `d = 2 × √(Fläche / π)` | Durchmesser eines Kreises mit gleicher Fläche |
| **Mittlere Intensität** | `np.mean(hologram[mask])` | Durchschnittliche Helligkeit |

**Sensitivitäts-Parameter:**

Der Slider steuert, wie streng die Detektion ist:

| Wert | Perzentil | Effekt |
|------|-----------|--------|
| 0.1 | 73% | Locker - viele Detektionen, auch schwache Signale |
| 0.5 | 85% | Ausgewogen - gute Balance |
| 0.9 | 96% | Streng - nur starke Signale, weniger Fehlalarme |

---

### 3. Datenexport (Zarr)

Das Zarr-Format wird im echten HoloPython-Projekt verwendet und ist optimiert für:

- **Chunked Storage**: Effizientes I/O für große Arrays
- **Kompression**: Reduziert Speicherbedarf
- **Hierarchische Struktur**: Gruppiert zusammengehörige Daten
- **Metadaten**: Speichert Analyse-Parameter

**Exportierte Struktur:**

```
export.zarr/
├── hologram/          # Original-Array (256×256, float64)
├── preprocessed/      # Vorverarbeitetes Array
├── labels/            # Segmentierungsmaske (int32)
├── particles/         # Strukturiertes Array mit Merkmalen
└── .zattrs            # Metadaten (threshold, n_particles, ...)
```

---

## Verwendete Technologien

### NumPy

```python
# Vektorisierte Operationen (schnell, kein Python-Loop)
distance = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)

# Broadcasting: 2D-Array aus 1D-Arrays
y, x = np.ogrid[:256, :256]

# Bedingte Zuweisung
hologram = np.where(mask, hologram + pattern, hologram)

# Clipping
hologram = np.clip(hologram, 0, 1)
```

### SciPy.ndimage

```python
# Median-Filter (Hintergrundschätzung)
scipy.ndimage.median_filter(image, size=40)

# Gaußsche Glättung (Ringe zu Peaks verschmelzen)
scipy.ndimage.gaussian_filter(image, sigma=7.5)

# Maximum-Filter (lokale Maxima finden)
scipy.ndimage.maximum_filter(image, size=25)

# Connected-Component-Labeling
labels, n = scipy.ndimage.label(binary)

# Watershed-Segmentierung (überlappende Regionen trennen)
scipy.ndimage.watershed_ift(inverted, markers)

# Bounding Boxes für jedes Label
slices = scipy.ndimage.find_objects(labels)

# Schwerpunktberechnung
centroids = scipy.ndimage.center_of_mass(image, labels, index=range(1, n+1))
```

---

## GPU-Beschleunigung (Ausblick)

An mehreren Stellen im Code sind Kommentare für potenzielle GPU-Beschleunigung:

```python
# TODO: GPU-Beschleunigung mit CuPy möglich - cupy.random.normal()
# TODO: GPU-Beschleunigung mit CuPy möglich - cupyx.scipy.ndimage.median_filter
# TODO: GPU-Beschleunigung mit CuPy möglich - cucim.skimage.measure.label
```

**CuPy** ist ein Drop-in-Replacement für NumPy, das auf NVIDIA GPUs läuft:

```python
import cupy as cp

# Statt NumPy
hologram = cp.random.normal(0.5, 0.1, (256, 256))
distance = cp.sqrt((x - cx) ** 2 + (y - cy) ** 2)

# Speedup: 10-100x für große Arrays
```

---

## Ausführen

```bash
# Virtuelle Umgebung aktivieren
source .venv/bin/activate

# Anwendung starten
python holo_analyzer_demo.py
```

### Abhängigkeiten

```
numpy>=2.0       # Array-Operationen
scipy>=1.13      # Bildverarbeitung
matplotlib>=3.9  # Visualisierung
zarr>=2.18       # Datenexport
tkinter          # GUI (in Python enthalten)
```

---

## Autor

**Jacob Gorinevski**
Bewerbung als Research Assistant - HoloPython Projekt
ETH Zürich, Januar 2026
