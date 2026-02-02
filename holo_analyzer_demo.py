#!/usr/bin/env python3
"""
Holographic Cloud Particle Analyzer - Demo Application
=======================================================

Eine Demo-Anwendung zur Analyse holografischer Aufnahmen von Wolkentröpfchen
und Eiskristallen. Entwickelt für die Bewerbung als Research Assistant
am HoloPython-Projekt der ETH Zürich.

Author: Jacob Gorinevski
Date: Januar 2026
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
import scipy.ndimage as ndi
import zarr

# =============================================================================
# KONSTANTEN
# =============================================================================

# Hologramm-Parameter
HOLOGRAM_SIZE: int = 256  # Größe des Hologramms in Pixeln
MIN_PARTICLES: int = 5   # Minimale Anzahl Partikel
MAX_PARTICLES: int = 15  # Maximale Anzahl Partikel
MIN_RADIUS: int = 5      # Minimaler Partikelradius in Pixeln
MAX_RADIUS: int = 20     # Maximaler Partikelradius in Pixeln

# Analyse-Parameter
DEFAULT_THRESHOLD: float = 0.5  # Sensitivität: 0.1=locker (viele), 0.9=streng (wenige)
MEDIAN_FILTER_SIZE: int = 5     # Größe des Median-Filters für Preprocessing

# Visualisierungs-Parameter
FIGURE_DPI: int = 100
OVERLAY_ALPHA: float = 0.7
COLORMAP: str = 'viridis'  # Modernes, wissenschaftliches Farbschema

# GUI-Farbschema (Dunkles Theme)
COLORS = {
    'bg_dark': '#1e1e2e',
    'bg_medium': '#2d2d3d',
    'bg_light': '#3d3d4d',
    'accent': '#89b4fa',
    'accent_hover': '#b4befe',
    'text': '#cdd6f4',
    'text_dim': '#6c7086',
    'success': '#a6e3a1',
    'warning': '#f9e2af',
}


# =============================================================================
# DATENKLASSEN
# =============================================================================

@dataclass
class ParticleFeatures:
    """
    Extrahierte Merkmale eines detektierten Partikels.

    Attributes
    ----------
    particle_id : int
        Eindeutige ID des Partikels.
    centroid_x : float
        X-Koordinate des Zentroids.
    centroid_y : float
        Y-Koordinate des Zentroids.
    area : int
        Fläche des Partikels in Pixeln.
    equivalent_diameter : float
        Äquivalenter Durchmesser (aus Fläche berechnet).
    mean_intensity : float
        Mittlere Intensität innerhalb des Partikels.
    """
    particle_id: int
    centroid_x: float
    centroid_y: float
    area: int
    equivalent_diameter: float
    mean_intensity: float


@dataclass
class AnalysisResult:
    """
    Ergebnis einer vollständigen Hologramm-Analyse.

    Attributes
    ----------
    original : NDArray[np.float64]
        Original-Hologramm.
    preprocessed : NDArray[np.float64]
        Vorverarbeitetes Hologramm.
    labels : NDArray[np.int32]
        Segmentierungsmaske mit Partikel-Labels.
    particles : list[ParticleFeatures]
        Liste der extrahierten Partikelmerkmale.
    threshold : float
        Verwendeter Schwellwert.
    """
    original: NDArray[np.float64]
    preprocessed: NDArray[np.float64]
    labels: NDArray[np.int32]
    particles: list[ParticleFeatures] = field(default_factory=list)
    threshold: float = DEFAULT_THRESHOLD


# =============================================================================
# HOLOGRAM SIMULATOR
# =============================================================================

class HologramSimulator:
    """
    Generiert synthetische Hologramm-Daten mit simulierten Beugungsmustern.

    Diese Klasse erzeugt realistische Hologramme von Wolkenpartikeln durch
    Simulation von Beugungsmustern (konzentrische Ringe mit Gaußschem
    Intensitätsprofil).

    Parameters
    ----------
    size : int, optional
        Größe des Hologramms in Pixeln (quadratisch).
        Default: HOLOGRAM_SIZE (256).
    noise_level : float, optional
        Standardabweichung des Gaußschen Hintergrundrauschens.
        Default: 0.1.
    seed : int | None, optional
        Random Seed für Reproduzierbarkeit.
        Default: None.

    Attributes
    ----------
    size : int
        Hologramm-Größe.
    noise_level : float
        Rausch-Level.
    rng : np.random.Generator
        Zufallszahlengenerator.

    Examples
    --------
    >>> simulator = HologramSimulator(size=256, seed=42)
    >>> hologram = simulator.generate()
    >>> hologram.shape
    (256, 256)
    """

    def __init__(
        self,
        size: int = HOLOGRAM_SIZE,
        noise_level: float = 0.1,
        seed: int | None = None
    ) -> None:
        self.size = size
        self.noise_level = noise_level
        self.rng = np.random.default_rng(seed)

    def generate(
        self,
        n_particles: int | None = None
    ) -> NDArray[np.float64]:
        """
        Generiert ein synthetisches Hologramm mit Beugungsmustern.

        Parameters
        ----------
        n_particles : int | None, optional
            Anzahl der zu generierenden Partikel.
            Wenn None, wird zufällig zwischen MIN_PARTICLES und MAX_PARTICLES gewählt.

        Returns
        -------
        NDArray[np.float64]
            2D-Array mit dem simulierten Hologramm.
            Werte sind auf [0, 1] normalisiert.
        """
        if n_particles is None:
            n_particles = self.rng.integers(MIN_PARTICLES, MAX_PARTICLES + 1)

        # Erzeuge Hintergrund mit Gaußschem Rauschen
        # TODO: GPU-Beschleunigung mit CuPy möglich - cupy.random.normal()
        hologram = self.rng.normal(
            loc=0.5,  # Mittlerer Hintergrundwert
            scale=self.noise_level,
            size=(self.size, self.size)
        )

        # Generiere Partikel mit Beugungsmustern
        for _ in range(n_particles):
            hologram = self._add_particle(hologram)

        # Normalisiere auf [0, 1] Bereich
        # clip() begrenzt Werte auf gültigen Intensitätsbereich
        hologram = np.clip(hologram, 0, 1)

        return hologram.astype(np.float64)

    def _add_particle(
        self,
        hologram: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """
        Fügt ein einzelnes Partikel mit Beugungsmuster hinzu.

        Das Beugungsmuster wird als konzentrische Ringe mit Gaußschem
        Intensitätsprofil simuliert (Airy-Pattern-Approximation).

        Parameters
        ----------
        hologram : NDArray[np.float64]
            Aktuelles Hologramm-Array.

        Returns
        -------
        NDArray[np.float64]
            Hologramm mit hinzugefügtem Partikel.
        """
        # Zufällige Position und Größe
        cx = self.rng.integers(MAX_RADIUS, self.size - MAX_RADIUS)
        cy = self.rng.integers(MAX_RADIUS, self.size - MAX_RADIUS)
        radius = self.rng.integers(MIN_RADIUS, MAX_RADIUS + 1)
        intensity = self.rng.uniform(0.3, 0.8)

        # Erstelle Koordinatengitter für Beugungsmuster
        # meshgrid erzeugt 2D-Koordinatenmatrizen aus 1D-Arrays
        y, x = np.ogrid[:self.size, :self.size]

        # Berechne Abstand vom Partikelzentrum
        # TODO: GPU-Beschleunigung mit CuPy möglich - schnellere Distanzberechnung
        distance = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)

        # Simuliere Beugungsmuster (konzentrische Ringe)
        # Bessel-Funktion wäre exakter, aber Gaußsche Approximation reicht hier
        # Die Formel erzeugt oszillierende Intensität mit abklingender Amplitude
        pattern = intensity * np.exp(-distance / (radius * 2)) * \
                  np.cos(2 * np.pi * distance / (radius * 0.8)) ** 2

        # Maske für den Einflussbereich des Partikels
        # where() wählt Werte basierend auf Bedingung aus
        mask = distance < radius * 3

        # Addiere Beugungsmuster zum Hologramm
        # TODO: GPU-Beschleunigung mit CuPy möglich - elementweise Addition
        hologram = np.where(mask, hologram + pattern, hologram)

        return hologram


# =============================================================================
# ANALYSE-PIPELINE
# =============================================================================

class HologramAnalyzer:
    """
    Analysiert Hologramme zur Partikelerkennung und Merkmalsextraktion.

    Verwendet einen verbesserten Algorithmus basierend auf:
    1. Difference of Gaussians (DoG) für Blob-Detektion
    2. Lokale Maxima-Detektion für Partikelzentren
    3. Watershed-Segmentierung für überlappende Partikel

    Parameters
    ----------
    threshold : float, optional
        Sensitivität für die Detektion (0-1). Höher = weniger Partikel.
        Default: DEFAULT_THRESHOLD.
    min_particle_size : int, optional
        Minimale Partikelgröße in Pixeln.
        Default: 10.

    Examples
    --------
    >>> analyzer = HologramAnalyzer(threshold=0.3)
    >>> result = analyzer.analyze(hologram)
    >>> print(f"Gefunden: {len(result.particles)} Partikel")
    """

    def __init__(
        self,
        threshold: float = DEFAULT_THRESHOLD,
        min_particle_size: int = 10
    ) -> None:
        self.threshold = threshold
        self.min_particle_size = min_particle_size

    def analyze(self, hologram: NDArray[np.float64]) -> AnalysisResult:
        """
        Führt die vollständige Analyse-Pipeline aus.

        Parameters
        ----------
        hologram : NDArray[np.float64]
            Eingabe-Hologramm als 2D-Array.

        Returns
        -------
        AnalysisResult
            Vollständiges Analyseergebnis mit allen Zwischenschritten.
        """
        # Schritt 1: Preprocessing mit DoG (Difference of Gaussians)
        preprocessed = self._preprocess_dog(hologram)

        # Schritt 2: Lokale Maxima finden (Partikelzentren)
        centers = self._find_local_maxima(preprocessed)

        # Schritt 3: Watershed-Segmentierung von den Zentren aus
        labels, n_labels = self._watershed_segment(preprocessed, centers)

        # Schritt 4: Merkmalsextraktion
        particles = self._extract_features(hologram, preprocessed, labels, n_labels)

        return AnalysisResult(
            original=hologram,
            preprocessed=preprocessed,
            labels=labels,
            particles=particles,
            threshold=self.threshold
        )

    def _preprocess_dog(
        self,
        hologram: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """
        Preprocessing: Starke Glättung um Beugungsmuster zu einem Peak zu verschmelzen.

        Die Beugungsmuster (konzentrische Ringe) werden durch einen großen
        Gaußfilter zu einem einzelnen Peak pro Partikel zusammengeführt.

        Parameters
        ----------
        hologram : NDArray[np.float64]
            Rohes Hologramm.

        Returns
        -------
        NDArray[np.float64]
            Geglättetes Bild mit einem Peak pro Partikel.
        """
        # TODO: GPU-Beschleunigung mit CuPy möglich - cupyx.scipy.ndimage.gaussian_filter

        # Schritt 1: Hintergrund entfernen (großer Median-Filter)
        background = ndi.median_filter(hologram, size=MAX_RADIUS * 2)
        foreground = hologram - background

        # Schritt 2: Moderate Gaußsche Glättung
        # Zu viel Glättung = überlappende Partikel verschmelzen
        # Zu wenig = Ringe werden als separate Peaks erkannt
        sigma = MIN_RADIUS * 1.5
        smoothed = ndi.gaussian_filter(foreground, sigma=sigma)

        # Schritt 3: Normalisierung auf [0, 1]
        smoothed = smoothed - smoothed.min()
        max_val = smoothed.max()
        if max_val > 0:
            smoothed = smoothed / max_val

        return smoothed

    def _find_local_maxima(
        self,
        preprocessed: NDArray[np.float64]
    ) -> NDArray[np.bool_]:
        """
        Findet lokale Maxima als Kandidaten für Partikelzentren.

        Ein Pixel ist ein lokales Maximum, wenn es größer ist als alle
        seine Nachbarn in einem definierten Radius.

        Parameters
        ----------
        preprocessed : NDArray[np.float64]
            Vorverarbeitetes Bild.

        Returns
        -------
        NDArray[np.bool_]
            Binärmaske mit True an lokalen Maxima.
        """
        # Maximum-Filter findet das Maximum in der Nachbarschaft jedes Pixels
        # TODO: GPU-Beschleunigung mit CuPy möglich - cupyx.scipy.ndimage.maximum_filter
        # Nachbarschaft = erwartete Partikelgröße (mittlerer Radius)
        avg_radius = (MIN_RADIUS + MAX_RADIUS) // 2
        neighborhood_size = 2 * avg_radius + 1

        # Dilatation: Jedes Pixel wird durch das Maximum seiner Nachbarschaft ersetzt
        local_max_values = ndi.maximum_filter(preprocessed, size=neighborhood_size)

        # Ein Pixel ist lokales Maximum wenn es gleich dem Nachbarschafts-Maximum ist
        is_local_max = (preprocessed == local_max_values)

        # Schwellwert basierend auf Perzentil der Intensitätswerte
        # Höherer threshold = höheres Perzentil = weniger, aber stärkere Maxima
        # threshold 0.1 → 70. Perzentil (viele Detektionen)
        # threshold 0.9 → 99. Perzentil (nur stärkste Signale)
        percentile = 70 + self.threshold * 29  # Bereich: 70-99
        intensity_threshold = np.percentile(preprocessed, percentile)

        is_above_threshold = (preprocessed > intensity_threshold)

        # Kombiniere beide Bedingungen
        maxima = is_local_max & is_above_threshold

        # Entferne Maxima am Bildrand (oft Artefakte)
        border = MAX_RADIUS
        maxima[:border, :] = False
        maxima[-border:, :] = False
        maxima[:, :border] = False
        maxima[:, -border:] = False

        return maxima

    def _watershed_segment(
        self,
        preprocessed: NDArray[np.float64],
        centers: NDArray[np.bool_]
    ) -> tuple[NDArray[np.int32], int]:
        """
        Watershed-Segmentierung ausgehend von den detektierten Zentren.

        Watershed behandelt das Bild wie eine topographische Karte und
        "flutet" von den Markern (Zentren) aus - ideal für überlappende Partikel.

        Parameters
        ----------
        preprocessed : NDArray[np.float64]
            Vorverarbeitetes Bild.
        centers : NDArray[np.bool_]
            Binärmaske der Partikelzentren.

        Returns
        -------
        tuple[NDArray[np.int32], int]
            - labels: Segmentierungsmaske
            - n_labels: Anzahl Partikel
        """
        # Markiere jeden gefundenen Peak mit einer eindeutigen ID
        markers, n_markers = ndi.label(centers)

        if n_markers == 0:
            return np.zeros_like(preprocessed, dtype=np.int32), 0

        # Erstelle Maske für den Bereich, der segmentiert werden soll
        # Nur Bereiche über einem niedrigeren Schwellwert werden betrachtet
        mask = preprocessed > (self.threshold * 0.3)

        # Invertiere das Bild für Watershed (Watershed findet Täler, nicht Gipfel)
        # TODO: GPU-Beschleunigung mit CuPy möglich
        inverted = preprocessed.max() - preprocessed

        # Watershed-Segmentierung
        # Jeder Marker "wächst" bis er auf einen anderen trifft oder den Maskenrand erreicht
        labels = ndi.watershed_ift(
            (inverted * 255).astype(np.uint8),
            markers
        )

        # Wende Maske an - Hintergrund wird 0
        labels = labels * mask.astype(np.int32)

        return labels.astype(np.int32), n_markers

    def _extract_features(
        self,
        hologram: NDArray[np.float64],
        preprocessed: NDArray[np.float64],
        labels: NDArray[np.int32],
        n_labels: int
    ) -> list[ParticleFeatures]:
        """
        Merkmalsextraktion für alle detektierten Partikel.

        Berechnet für jedes Partikel:
        - Zentroid (Schwerpunkt, gewichtet nach DoG-Intensität)
        - Fläche in Pixeln
        - Äquivalenter Durchmesser
        - Mittlere Intensität im Original

        Parameters
        ----------
        hologram : NDArray[np.float64]
            Original-Hologramm für Intensitätsmessung.
        preprocessed : NDArray[np.float64]
            DoG-gefiltertes Bild für Zentroid-Berechnung.
        labels : NDArray[np.int32]
            Segmentierungsmaske.
        n_labels : int
            Anzahl gefundener Partikel.

        Returns
        -------
        list[ParticleFeatures]
            Liste mit Merkmalen aller Partikel.
        """
        particles: list[ParticleFeatures] = []

        if n_labels == 0:
            return particles

        # find_objects gibt Bounding-Box-Slices für jedes Label zurück
        # TODO: GPU-Beschleunigung mit CuPy möglich - parallelisierte Merkmalsberechnung
        slices = ndi.find_objects(labels)

        # center_of_mass gewichtet nach DoG-Bild (findet echtes Zentrum besser)
        centroids = ndi.center_of_mass(
            preprocessed,  # Gewichtung nach DoG statt Original
            labels,
            index=range(1, n_labels + 1)
        )

        for i, item in enumerate(zip(slices, centroids), start=1):
            slc, centroid = item

            if slc is None:
                continue

            # Überspringe wenn Zentroid NaN ist (kann bei leeren Regionen passieren)
            if np.isnan(centroid[0]) or np.isnan(centroid[1]):
                continue

            # Maske für aktuelles Partikel
            particle_mask = labels[slc] == i

            # Fläche = Anzahl der Pixel im Partikel
            area = np.count_nonzero(particle_mask)

            # Überspringe zu kleine Partikel (wahrscheinlich Rauschen)
            if area < self.min_particle_size:
                continue

            # Äquivalenter Durchmesser (Durchmesser eines Kreises mit gleicher Fläche)
            equivalent_diameter = 2 * np.sqrt(area / np.pi)

            # Mittlere Intensität im Original-Hologramm
            particle_region = hologram[slc]
            mean_intensity = float(np.mean(particle_region[particle_mask]))

            particles.append(ParticleFeatures(
                particle_id=len(particles) + 1,  # Neu nummerieren nach Filterung
                centroid_x=float(centroid[1]),  # Spalte = X
                centroid_y=float(centroid[0]),  # Zeile = Y
                area=area,
                equivalent_diameter=equivalent_diameter,
                mean_intensity=mean_intensity
            ))

        return particles


# =============================================================================
# GUI-ANWENDUNG
# =============================================================================

class HoloAnalyzerGUI:
    """
    Hauptanwendung mit grafischer Benutzeroberfläche.

    Bietet eine interaktive Oberfläche zur Analyse von Hologrammen mit:
    - Synthetische Datengenerierung
    - Echtzeit-Analyse mit einstellbarem Schwellwert
    - Visualisierung mit Partikel-Overlay
    - Export im Zarr-Format

    Parameters
    ----------
    master : tk.Tk
        Tkinter-Hauptfenster.

    Attributes
    ----------
    simulator : HologramSimulator
        Generator für synthetische Daten.
    analyzer : HologramAnalyzer
        Analyse-Pipeline.
    current_hologram : NDArray[np.float64] | None
        Aktuell geladenes Hologramm.
    current_result : AnalysisResult | None
        Aktuelles Analyseergebnis.
    """

    def __init__(self, master) -> None:
        # Importiere GUI-Module
        import tkinter as tk
        from tkinter import ttk, filedialog, messagebox
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
        from matplotlib.patches import Circle
        from matplotlib.figure import Figure
        import matplotlib.pyplot as plt

        # Speichere Module als Instanz-Attribute für späteren Zugriff
        self._tk = tk
        self._ttk = ttk
        self._filedialog = filedialog
        self._messagebox = messagebox
        self._FigureCanvasTkAgg = FigureCanvasTkAgg
        self._NavigationToolbar2Tk = NavigationToolbar2Tk
        self._Circle = Circle
        self._Figure = Figure
        self._plt = plt

        self.master = master
        self.master.title("HoloPython - Cloud Particle Analyzer")
        self.master.geometry("1200x800")
        self.master.minsize(1000, 700)

        # Dunkles Theme für Matplotlib
        plt.style.use('dark_background')

        # Initialisiere Komponenten
        self.simulator = HologramSimulator()
        self.analyzer = HologramAnalyzer()

        # Zustandsvariablen
        self.current_hologram: NDArray[np.float64] | None = None
        self.current_result: AnalysisResult | None = None
        self.show_overlay = tk.BooleanVar(value=True)
        self.show_preprocessed = tk.BooleanVar(value=False)
        self.threshold_var = tk.DoubleVar(value=DEFAULT_THRESHOLD)

        # Style konfigurieren
        self._configure_style()

        # GUI aufbauen
        self._create_widgets()
        self._create_layout()

        # Initiale Daten generieren
        self._generate_new()

    def _configure_style(self) -> None:
        """Konfiguriert das visuelle Theme der Anwendung."""
        style = self._ttk.Style()

        # Versuche modernes Theme
        available_themes = style.theme_names()
        if 'clam' in available_themes:
            style.theme_use('clam')

        # Konfiguriere Farben
        style.configure('TFrame', background=COLORS['bg_dark'])
        style.configure('TLabelframe', background=COLORS['bg_dark'])
        style.configure('TLabelframe.Label',
                       background=COLORS['bg_dark'],
                       foreground=COLORS['accent'],
                       font=('Helvetica', 10, 'bold'))
        style.configure('TLabel',
                       background=COLORS['bg_dark'],
                       foreground=COLORS['text'])
        style.configure('TButton',
                       padding=8,
                       font=('Helvetica', 10))
        style.configure('Accent.TButton',
                       padding=10,
                       font=('Helvetica', 11, 'bold'))
        style.configure('TCheckbutton',
                       background=COLORS['bg_dark'],
                       foreground=COLORS['text'])
        style.configure('TScale', background=COLORS['bg_dark'])
        style.configure('Treeview',
                       background=COLORS['bg_medium'],
                       foreground=COLORS['text'],
                       fieldbackground=COLORS['bg_medium'],
                       rowheight=25)
        style.configure('Treeview.Heading',
                       background=COLORS['bg_light'],
                       foreground=COLORS['accent'],
                       font=('Helvetica', 9, 'bold'))
        style.map('Treeview', background=[('selected', COLORS['accent'])])

        # Master-Fenster Hintergrund
        self.master.configure(bg=COLORS['bg_dark'])

    def _create_widgets(self) -> None:
        """Erstellt alle GUI-Widgets."""
        tk = self._tk
        ttk = self._ttk

        # === Linkes Control Panel ===
        self.control_frame = ttk.LabelFrame(
            self.master,
            text=" Steuerung ",
            padding=15
        )

        # Daten-Buttons Frame
        self.data_frame = ttk.LabelFrame(
            self.control_frame,
            text=" Daten ",
            padding=8
        )

        self.btn_generate = ttk.Button(
            self.data_frame,
            text="⟳  Neu generieren",
            command=self._generate_new,
            style='Accent.TButton'
        )

        self.btn_load = ttk.Button(
            self.data_frame,
            text="📂  Laden...",
            command=self._load_data
        )

        self.btn_export = ttk.Button(
            self.data_frame,
            text="💾  Exportieren (Zarr)",
            command=self._export_zarr
        )

        # Analyse-Parameter Frame
        self.param_frame = ttk.LabelFrame(
            self.control_frame,
            text=" Parameter ",
            padding=8
        )

        # Threshold Slider mit Wertanzeige
        self.threshold_label_title = ttk.Label(
            self.param_frame,
            text="Segmentierungs-Schwellwert:"
        )

        self.threshold_value_frame = ttk.Frame(self.param_frame)

        self.threshold_slider = ttk.Scale(
            self.threshold_value_frame,
            from_=0.1,
            to=0.9,
            variable=self.threshold_var,
            orient=tk.HORIZONTAL,
            length=120,
            command=self._on_threshold_change
        )

        self.threshold_label = ttk.Label(
            self.threshold_value_frame,
            text=f"{DEFAULT_THRESHOLD:.2f}",
            font=('Helvetica', 12, 'bold'),
            width=5
        )

        # Anzeige-Optionen Frame
        self.display_frame = ttk.LabelFrame(
            self.control_frame,
            text=" Anzeige ",
            padding=8
        )

        self.chk_overlay = ttk.Checkbutton(
            self.display_frame,
            text="Partikel-Markierungen",
            variable=self.show_overlay,
            command=self._update_plot
        )

        self.chk_preprocessed = ttk.Checkbutton(
            self.display_frame,
            text="Vorverarbeitet anzeigen",
            variable=self.show_preprocessed,
            command=self._update_plot
        )

        self.btn_histogram = ttk.Button(
            self.display_frame,
            text="📊  Größenverteilung",
            command=self._show_histogram
        )

        # === Statistik-Zusammenfassung ===
        self.stats_frame = ttk.LabelFrame(
            self.control_frame,
            text=" Statistik ",
            padding=8
        )

        self.stats_labels = {}
        stats_items = [
            ('n_particles', 'Partikel:'),
            ('mean_diameter', 'Ø Durchmesser:'),
            ('total_area', 'Gesamtfläche:'),
            ('mean_intensity', 'Ø Intensität:'),
        ]

        for key, label_text in stats_items:
            frame = ttk.Frame(self.stats_frame)
            ttk.Label(frame, text=label_text, width=14, anchor='w').pack(side=tk.LEFT)
            value_label = ttk.Label(frame, text="—", font=('Helvetica', 10, 'bold'))
            value_label.pack(side=tk.RIGHT)
            self.stats_labels[key] = value_label
            frame.pack(fill=tk.X, pady=2)

        # === Status ===
        self.status_frame = ttk.Frame(self.control_frame)
        self.status_label = ttk.Label(
            self.status_frame,
            text="● Bereit",
            font=('Helvetica', 9),
            foreground=COLORS['success']
        )

        # === Rechtes Panel: Visualisierung ===
        self.viz_frame = ttk.Frame(self.master)

        # Matplotlib Figure mit dunklem Hintergrund
        self.figure = self._Figure(figsize=(7, 6), dpi=FIGURE_DPI, facecolor=COLORS['bg_dark'])
        self.ax = self.figure.add_subplot(111)
        self.ax.set_facecolor(COLORS['bg_medium'])
        self.canvas = self._FigureCanvasTkAgg(self.figure, master=self.viz_frame)
        self.canvas.get_tk_widget().configure(bg=COLORS['bg_dark'])

        # Toolbar
        self.toolbar = self._NavigationToolbar2Tk(self.canvas, self.viz_frame)
        self.toolbar.update()

        # === Unteres Panel: Partikel-Tabelle ===
        self.table_frame = ttk.LabelFrame(
            self.master,
            text=" Erkannte Partikel ",
            padding=8
        )

        # Treeview für Tabelle
        columns = ("ID", "X [px]", "Y [px]", "Ø [px]", "Fläche [px²]", "Intensität")
        self.table = ttk.Treeview(
            self.table_frame,
            columns=columns,
            show="headings",
            height=5,
            selectmode='browse'
        )

        # Spaltenüberschriften mit Sortierung
        col_widths = [50, 80, 80, 80, 100, 90]
        for col, width in zip(columns, col_widths):
            self.table.heading(col, text=col)
            self.table.column(col, width=width, anchor=tk.CENTER, minwidth=50)

        # Scrollbar
        self.scrollbar = ttk.Scrollbar(
            self.table_frame,
            orient=tk.VERTICAL,
            command=self.table.yview
        )
        self.table.configure(yscrollcommand=self.scrollbar.set)

        # Klick-Event für Partikel-Highlight
        self.table.bind('<<TreeviewSelect>>', self._on_particle_select)

    def _create_layout(self) -> None:
        """Ordnet die Widgets im Layout an."""
        tk = self._tk
        ttk = self._ttk

        # Grid-Konfiguration für responsives Layout
        self.master.columnconfigure(0, weight=0, minsize=220)
        self.master.columnconfigure(1, weight=1)
        self.master.rowconfigure(0, weight=1)
        self.master.rowconfigure(1, weight=0)

        # === Control Panel (links) ===
        self.control_frame.grid(row=0, column=0, rowspan=2, sticky="nsew", padx=8, pady=8)

        # Daten-Buttons
        self.data_frame.pack(fill=tk.X, pady=(0, 10))
        self.btn_generate.pack(fill=tk.X, pady=3)
        self.btn_load.pack(fill=tk.X, pady=3)
        self.btn_export.pack(fill=tk.X, pady=3)

        # Parameter
        self.param_frame.pack(fill=tk.X, pady=10)
        self.threshold_label_title.pack(anchor='w', pady=(0, 5))
        self.threshold_value_frame.pack(fill=tk.X)
        self.threshold_slider.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.threshold_label.pack(side=tk.RIGHT, padx=(10, 0))

        # Anzeige-Optionen
        self.display_frame.pack(fill=tk.X, pady=10)
        self.chk_overlay.pack(anchor='w', pady=2)
        self.chk_preprocessed.pack(anchor='w', pady=2)
        self.btn_histogram.pack(fill=tk.X, pady=(10, 0))

        # Statistik
        self.stats_frame.pack(fill=tk.X, pady=10)

        # Status (unten im Control Panel)
        self.status_frame.pack(fill=tk.X, side=tk.BOTTOM, pady=(10, 0))
        self.status_label.pack(anchor='w')

        # === Visualisierung (rechts oben) ===
        self.viz_frame.grid(row=0, column=1, sticky="nsew", padx=(0, 8), pady=8)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.toolbar.pack(fill=tk.X)

        # === Tabelle (rechts unten) ===
        self.table_frame.grid(row=1, column=1, sticky="nsew", padx=(0, 8), pady=(0, 8))
        self.table.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

    def _on_particle_select(self, event) -> None:
        """Hebt ausgewähltes Partikel im Plot hervor."""
        selection = self.table.selection()
        if selection and self.current_result:
            item = self.table.item(selection[0])
            particle_id = int(item['values'][0])
            self._update_plot(highlight_id=particle_id)

    def _generate_new(self) -> None:
        """Generiert ein neues synthetisches Hologramm."""
        self._set_status("⟳ Generiere...", success=True)
        self.master.update()
        self.current_hologram = self.simulator.generate()
        self._analyze()

    def _load_data(self) -> None:
        """Lädt Hologramm-Daten aus einer NumPy-Datei."""
        filepath = self._filedialog.askopenfilename(
            title="Hologramm laden",
            filetypes=[
                ("NumPy Arrays", "*.npy"),
                ("Zarr Arrays", "*.zarr"),
                ("Alle Dateien", "*.*")
            ]
        )

        if not filepath:
            return

        try:
            path = Path(filepath)
            if path.suffix == ".npy":
                self.current_hologram = np.load(filepath).astype(np.float64)
            elif path.suffix == ".zarr" or path.is_dir():
                zarr_array = zarr.open(filepath, mode='r')
                self.current_hologram = np.array(zarr_array).astype(np.float64)
            else:
                raise ValueError(f"Unbekanntes Format: {path.suffix}")

            # Normalisieren falls nötig
            if self.current_hologram.max() > 1.0:
                self.current_hologram = self.current_hologram / self.current_hologram.max()

            self._set_status(f"Geladen: {path.name}")
            self._analyze()

        except Exception as e:
            self._messagebox.showerror("Fehler", f"Laden fehlgeschlagen:\n{e}")

    def _analyze(self) -> None:
        """Führt die Analyse auf dem aktuellen Hologramm aus."""
        if self.current_hologram is None:
            self._set_status("Keine Daten geladen", success=False)
            return

        # Aktualisiere Schwellwert
        self.analyzer.threshold = self.threshold_var.get()

        # Führe Analyse durch
        self.current_result = self.analyzer.analyze(self.current_hologram)

        # Aktualisiere GUI
        n_particles = len(self.current_result.particles)
        self._set_status(f"● {n_particles} Partikel erkannt", success=True)

        self._update_table()
        self._update_stats()
        self._update_plot()

    def _update_stats(self) -> None:
        """Aktualisiert die Statistik-Zusammenfassung."""
        if not self.current_result or not self.current_result.particles:
            for label in self.stats_labels.values():
                label.config(text="—")
            return

        particles = self.current_result.particles
        diameters = [p.equivalent_diameter for p in particles]
        areas = [p.area for p in particles]
        intensities = [p.mean_intensity for p in particles]

        self.stats_labels['n_particles'].config(text=str(len(particles)))
        self.stats_labels['mean_diameter'].config(text=f"{np.mean(diameters):.1f} px")
        self.stats_labels['total_area'].config(text=f"{sum(areas):,} px²")
        self.stats_labels['mean_intensity'].config(text=f"{np.mean(intensities):.3f}")

    def _update_plot(self, highlight_id: int | None = None) -> None:
        """Aktualisiert die Matplotlib-Visualisierung."""
        if self.current_result is None:
            return

        # Komplett neu zeichnen um Colorbar-Probleme zu vermeiden
        self.figure.clear()
        self.ax = self.figure.add_subplot(111)
        self.ax.set_facecolor(COLORS['bg_medium'])

        # Wähle Daten basierend auf Checkbox
        if self.show_preprocessed.get():
            data = self.current_result.preprocessed
            title = "Vorverarbeitetes Hologramm"
        else:
            data = self.current_result.original
            title = "Hologramm-Analyse"

        # Hologramm als Heatmap mit Colorbar
        im = self.ax.imshow(
            data,
            cmap=COLORMAP,
            origin='upper',
            vmin=0,
            vmax=1,
            aspect='equal'
        )

        # Colorbar hinzufügen
        cbar = self.figure.colorbar(im, ax=self.ax, shrink=0.8, pad=0.02)
        cbar.set_label('Intensität', color=COLORS['text'])
        cbar.ax.yaxis.set_tick_params(color=COLORS['text'])
        for label in cbar.ax.yaxis.get_ticklabels():
            label.set_color(COLORS['text'])

        # Partikel-Overlay
        if self.show_overlay.get() and self.current_result.particles:
            for particle in self.current_result.particles:
                is_highlighted = (highlight_id == particle.particle_id)

                # Farbe und Stil basierend auf Highlight
                if is_highlighted:
                    color = '#ff6b6b'
                    linewidth = 3
                    alpha = 1.0
                else:
                    color = '#00ff88'
                    linewidth = 1.5
                    alpha = OVERLAY_ALPHA

                # Zeichne Kreis um Partikel
                circle = self._Circle(
                    (particle.centroid_x, particle.centroid_y),
                    radius=particle.equivalent_diameter / 2,
                    fill=False,
                    color=color,
                    linewidth=linewidth,
                    alpha=alpha
                )
                self.ax.add_patch(circle)

                # Partikel-ID als Annotation
                self.ax.annotate(
                    str(particle.particle_id),
                    (particle.centroid_x, particle.centroid_y - particle.equivalent_diameter / 2 - 5),
                    color='white' if is_highlighted else '#ffdd44',
                    fontsize=9 if is_highlighted else 8,
                    fontweight='bold' if is_highlighted else 'normal',
                    ha='center',
                    va='bottom'
                )

        # Styling
        n_particles = len(self.current_result.particles) if self.current_result.particles else 0
        self.ax.set_title(f"{title}  |  {n_particles} Partikel erkannt",
                         color=COLORS['text'], fontsize=11, fontweight='bold', pad=10)
        self.ax.set_xlabel("X [Pixel]", color=COLORS['text'])
        self.ax.set_ylabel("Y [Pixel]", color=COLORS['text'])
        self.ax.tick_params(colors=COLORS['text'])

        # Rahmenfarbe
        for spine in self.ax.spines.values():
            spine.set_color(COLORS['text_dim'])

        self.figure.tight_layout()
        self.canvas.draw()

    def _update_table(self) -> None:
        """Aktualisiert die Partikel-Statistik-Tabelle."""
        tk = self._tk

        # Lösche alte Einträge
        for item in self.table.get_children():
            self.table.delete(item)

        if self.current_result is None:
            return

        # Füge neue Einträge hinzu
        for p in self.current_result.particles:
            self.table.insert("", tk.END, values=(
                p.particle_id,
                f"{p.centroid_x:.1f}",
                f"{p.centroid_y:.1f}",
                f"{p.equivalent_diameter:.1f}",
                p.area,
                f"{p.mean_intensity:.3f}"
            ))

    def _show_histogram(self) -> None:
        """Zeigt ein Histogramm der Partikelgrößen-Verteilung."""
        tk = self._tk
        ttk = self._ttk
        Figure = self._Figure

        if not self.current_result or not self.current_result.particles:
            self._messagebox.showinfo("Info", "Keine Partikel für Histogramm vorhanden")
            return

        # Extrahiere Durchmesser
        diameters = [p.equivalent_diameter for p in self.current_result.particles]

        # Erstelle neues Fenster mit dunklem Theme
        hist_window = tk.Toplevel(self.master)
        hist_window.title("Partikel-Größenverteilung")
        hist_window.geometry("600x500")
        hist_window.configure(bg=COLORS['bg_dark'])

        # Matplotlib Figure mit dunklem Theme
        fig = Figure(figsize=(6, 5), dpi=FIGURE_DPI, facecolor=COLORS['bg_dark'])
        ax = fig.add_subplot(111)
        ax.set_facecolor(COLORS['bg_medium'])

        # Histogramm mit schönerem Styling
        n, bins, patches = ax.hist(
            diameters,
            bins=min(15, len(diameters)),
            edgecolor=COLORS['bg_dark'],
            alpha=0.85,
            color=COLORS['accent'],
            linewidth=1.5
        )

        # Statistik
        mean_d = np.mean(diameters)
        std_d = np.std(diameters)
        median_d = np.median(diameters)

        # Vertikale Linien für Statistik
        ax.axvline(mean_d, color='#ff6b6b', linestyle='-', linewidth=2,
                  label=f'Mittelwert: {mean_d:.1f} px')
        ax.axvline(median_d, color='#ffd93d', linestyle='--', linewidth=2,
                  label=f'Median: {median_d:.1f} px')

        # Styling
        ax.set_xlabel("Äquivalenter Durchmesser [Pixel]", color=COLORS['text'], fontsize=10)
        ax.set_ylabel("Anzahl Partikel", color=COLORS['text'], fontsize=10)
        ax.set_title(f"Größenverteilung  |  n = {len(diameters)}  |  σ = {std_d:.1f} px",
                    color=COLORS['text'], fontsize=11, fontweight='bold', pad=10)

        ax.tick_params(colors=COLORS['text'])
        for spine in ax.spines.values():
            spine.set_color(COLORS['text_dim'])

        ax.legend(loc='upper right', facecolor=COLORS['bg_medium'],
                 edgecolor=COLORS['text_dim'], labelcolor=COLORS['text'])
        ax.grid(True, alpha=0.2, color=COLORS['text_dim'])

        fig.tight_layout()

        # Canvas
        canvas = self._FigureCanvasTkAgg(fig, master=hist_window)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

    def _export_zarr(self) -> None:
        """Exportiert Daten und Metadaten im Zarr-Format."""
        if self.current_result is None:
            self._messagebox.showinfo("Info", "Keine Daten zum Exportieren")
            return

        filepath = self._filedialog.asksaveasfilename(
            title="Zarr-Export",
            defaultextension=".zarr",
            filetypes=[("Zarr Arrays", "*.zarr")]
        )

        if not filepath:
            return

        try:
            # Erstelle Zarr-Gruppe für hierarchische Speicherung
            root = zarr.open_group(filepath, mode='w')

            # Speichere Hologramm-Arrays
            # Zarr 2.x: create_dataset, Zarr 3.x: create_array
            create_fn = getattr(root, 'create_dataset', root.create_array)

            create_fn(
                'hologram',
                data=self.current_result.original,
                chunks=(64, 64),  # Chunk-Größe für effizientes I/O
            )

            create_fn(
                'preprocessed',
                data=self.current_result.preprocessed,
                chunks=(64, 64),
            )

            create_fn(
                'labels',
                data=self.current_result.labels,
                chunks=(64, 64),
            )

            # Speichere Partikel-Daten als strukturiertes Array
            if self.current_result.particles:
                particle_data = np.array([
                    (p.particle_id, p.centroid_x, p.centroid_y,
                     p.area, p.equivalent_diameter, p.mean_intensity)
                    for p in self.current_result.particles
                ], dtype=[
                    ('id', 'i4'),
                    ('centroid_x', 'f8'),
                    ('centroid_y', 'f8'),
                    ('area', 'i4'),
                    ('diameter', 'f8'),
                    ('intensity', 'f8')
                ])
                create_fn('particles', data=particle_data)

            # Metadaten
            root.attrs['threshold'] = self.current_result.threshold
            root.attrs['n_particles'] = len(self.current_result.particles)
            root.attrs['software'] = 'HoloPython Demo'
            root.attrs['version'] = '1.0.0'

            self._set_status(f"✓ Exportiert: {Path(filepath).name}")
            self._messagebox.showinfo("Erfolg", f"Daten exportiert nach:\n{filepath}")

        except Exception as e:
            self._messagebox.showerror("Fehler", f"Export fehlgeschlagen:\n{e}")

    def _on_threshold_change(self, value: str) -> None:
        """Callback für Schwellwert-Änderungen."""
        threshold = float(value)
        self.threshold_label.config(text=f"{threshold:.2f}")
        # Automatische Neuanalyse bei Änderung
        self._analyze()

    def _set_status(self, message: str, success: bool = True) -> None:
        """Aktualisiert die Statusanzeige."""
        color = COLORS['success'] if success else COLORS['warning']
        self.status_label.config(text=message, foreground=color)


# =============================================================================
# HAUPTPROGRAMM
# =============================================================================

def main() -> None:
    """Startet die HoloPython Demo-Anwendung."""
    import tkinter as tk
    from tkinter import ttk

    root = tk.Tk()

    # Style konfigurieren
    style = ttk.Style()
    style.theme_use('clam')  # Modernes Theme

    # Anwendung starten
    app = HoloAnalyzerGUI(root)

    # Fenster schließen Handler
    root.protocol("WM_DELETE_WINDOW", root.quit)

    # Event-Loop starten
    root.mainloop()


if __name__ == "__main__":
    main()
