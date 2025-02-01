import pyvista as pv
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from natsort import natsorted
import imageio
import re
import math
import os
import natsort  # Per ordinare i file in modo naturale (es. 1, 2, 10 invece di 1, 10, 2)


import os
import math
import numpy as np
import pyvista as pv
import natsort


plt.rc('xtick', labelsize=15) 
plt.rc('ytick', labelsize=15) 
plt.rc('legend',fontsize=20) # using a size in points
plt.rc('legend',fontsize='large') # using a named size
plt.rc('axes', titlesize=20)
plt.rc('axes', labelsize=18)

vtk_folder = "/home/sarse/Desktop/UniCode/UniversityCODE/CFD/Assignment/5-final/VTK/"
vtk_files = ["prova2Daikx_5427.vtk", "prova2Daikx_16366.vtk", "prova2Daikx_31234.vtk"]


def plot_diameter_histograms(folder, files, bins, save_path, timeToPlot):
    """
    Plotta gli istogrammi dei diametri delle particelle per diversi tempi da file VTK.

    Args:
        folder (str): Cartella contenente i file VTK.
        files (list of str): Lista dei file VTK per i diversi tempi.
        bins (int): Numero di bin per l'istogramma.
        save_path (str): Percorso per salvare il grafico.
        timeToPlot (list of str): Lista dei tempi corrispondenti ai file VTK per i titoli.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import pyvista as pv

    # Setup del grafico
    fig, axs = plt.subplots(1, len(files), figsize=(18, 6), sharey=True)

    for i, file in enumerate(files):
        vtk_file = folder + file
        mesh = pv.read(vtk_file)
        
        # Controlla i campi disponibili
        if "d" not in mesh.array_names:
            print(f"Il campo 'd' non è disponibile nel file VTK: {file}")
            continue

        # Estrai il campo 'd' (diametro delle particelle)
        diameters = mesh["d"] * 1e6  # Converti in micrometri

        # Crea l'istogramma
        counts, edges = np.histogram(diameters, bins=bins, density=False)

        # Plot dell'istogramma (numero di particelle)
        axs[i].bar(edges[:-1], counts, width=np.diff(edges), edgecolor="black", alpha=0.7)
        axs[i].set_xlabel("Parcel Diameter [microm]")
        axs[i].set_title(f"Time = {timeToPlot[i]}s")
        axs[i].grid(True, linestyle="--", alpha=0.6)

    # Configurazione dell'asse y condiviso
    axs[0].set_ylabel("Number of Parcel")

    # Salva e mostra il grafico
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()



def plot_residuals(file_name, save_filename):
    # Carica il file in un DataFrame (file .dat)
    data = pd.read_csv(file_name, sep=r'\s+', skiprows=4, header=None, 
                       names=["Time", "p", "Ux", "Uy", "Uz","h", "k", "epsilon"])

    # Sostituisci 'N/A' con NaN per poter trattare i valori mancanti
    data.replace('N/A', float('nan'), inplace=True)
    
    # Converti tutte le colonne numeriche
    data = data.apply(pd.to_numeric, errors='ignore')
    
    # Creare il grafico
    plt.figure(figsize=(10, 6))
    
    # Tracciare i residui per ogni variabile
    plt.plot(data['Time'], data['p'], label='p', linestyle='-', color='blue', linewidth=2)
    #plt.plot(data['Time'], data['Ux'], label='Ux', linestyle='--', color='red', linewidth=2)
    #plt.plot(data['Time'], data['Uy'], label='Uy', linestyle='-.', color='green', linewidth=2)
    #plt.plot(data['Time'], data['k'], label='k', linestyle=':', color='orange', linewidth=2)
    plt.plot(data['Time'], data['epsilon'], label='epsilon', linestyle='--', color='purple', linewidth=2)
    plt.plot(data['Time'], data['h'], label='h', linestyle='-.', color='black', linewidth=2)
    plt.plot(data['Time'], data['k'], label='k', linestyle=':', color='orange', linewidth=2)
    
    plt.yscale('log')
    
    # Aggiungere etichette e legenda
    plt.xlabel('Time, t [s]')
    plt.ylabel('Residuals [-]')
    plt.legend(loc='upper right')
    plt.grid(True, linestyle='--', alpha=0.6)
   
    # Verifica del percorso e salvataggio
    save_folder = os.path.dirname(save_filename)
    if save_folder and not os.path.exists(save_folder):
        os.makedirs(save_folder)  # Crea la directory se non esiste

    try:
        plt.savefig(save_filename, format='png', dpi=300)
        print(f"Grafico salvato correttamente in: {save_filename}")
    except Exception as e:
        print(f"Errore durante il salvataggio del grafico: {e}")
    
    plt.show(block = False)
    plt.close()



# Define the function f(x) = sin(pi * x / 0.1)
def generate_half_sine_curve(duration=0.1, num_points=100):
    """
    Generates the (x, y) pairs for the half-sine curve f(x) = sin(pi * x / duration).

    Parameters:
    - duration (float): The duration over which the half-sine curve is defined.
    - num_points (int): Number of points to sample.

    Returns:
    - list of tuples: Each tuple contains an (x, y) pair.
    """
    x = np.linspace(0, duration, num_points)  # Generate equally spaced points in the range [0, duration]
    y = np.sin(np.pi * x / duration)         # Compute the corresponding y values
    return list(zip(x, y))                   # Return the list of (x, y) pairs










save_folder = "/home/sarse/Desktop/UniCode/UniversityCODE/CFD/Assignment/5-final/"

#######################################################################################


#####################################################################################


'''

import pyvista as pv
import numpy as np

# Percorso del file VTK
vtk_folder = "/home/sarse/OpenFOAM/sarse-12/run/Assignment/prova2/VTK/"
file = "prova2_0.vtk"

# Caricare il file VTK
mesh = pv.read(vtk_folder + file)

# Verificare la presenza del campo "cellLevel"
print("Campi disponibili:", mesh.array_names)
if "cellLevel" not in mesh.cell_data:
    raise ValueError("Il campo 'cellLevel' non è presente nei dati della mesh!")

# Creare uno slice nel piano XY (z=0)
slice_mesh = mesh.slice(normal=(0, 0, 1), origin=(0, 0, 0))

# Creare il plotter
plotter = pv.Plotter(off_screen=True)

sargs = dict(n_labels=6)

# Aggiungere la mesh e la legenda centrata
plotter.add_mesh(slice_mesh, scalars="cellLevel", cmap="coolwarm", show_scalar_bar=True, scalar_bar_args=sargs)
plotter.add_mesh(slice_mesh, scalars="cellLevel", show_edges=True, opacity=0.3, show_scalar_bar=False)
plotter.scalar_bar.SetPosition(0.25, 0.15)  # Posizionare la barra della legenda al centro in basso
plotter.scalar_bar.SetWidth(0.5)  # Impostare la larghezza della barra
plotter.scalar_bar.SetHeight(0.1)  # Impostare l'altezza della barra

# Impostare la vista XY e ruotare di 90° attorno a Z
plotter.view_yx()
plotter.zoom_camera(1.3)

plotter.show(screenshot=save_folder + "cellLevel.png")


import pyvista as pv
import numpy as np

# Percorso del file VTK
vtk_folder = "/home/sarse/OpenFOAM/sarse-12/run/Assignment/prova2/VTK/"
file = "prova2_0.vtk"

# Caricare il file VTK
mesh = pv.read(vtk_folder + file)

# Creare il plotter
plotter = pv.Plotter(off_screen=True)

sargs = dict(n_labels=6)

# Aggiungere la mesh e la legenda centrata
plotter.add_mesh(mesh, style="wireframe", color="black", show_edges=True)

# Impostare la vista XY e ruotare di 90° attorno a Z
plotter.zoom_camera(1.3)
plotter.camera_position = (1, -1, 0.6)

plotter.add_axes()


plotter.show(screenshot=save_folder + "meshWireframe.png")


import pyvista as pv
import numpy as np

# Percorso del file VTK
vtk_folder = "/home/sarse/OpenFOAM/sarse-12/run/Assignment/prova2/VTK/"
file = "prova2_0.vtk"

# Caricare il file VTK
mesh = pv.read(vtk_folder + file)

# Creare il plotter
plotter = pv.Plotter(off_screen = True)  # Due pannelli per le clip

# Parametri di clipping
clip_x = mesh.clip(normal=(-1, 0, 0), invert=False, origin=(0.01, 0, 0))
clip_z = clip_x.clip(normal=(0, 0, -1), invert=False, origin = (0, 0, 0.01))

# Aggiungere la clip lungo X
plotter.add_mesh(clip_z, show_edges=True, color = "white")
#plotter.add_text("Clip Normal X", font_size=10)
plotter.camera_position= (0.6, -0.4, 0.4)
#plotter.add_axes()

plotter.zoom_camera(1.3)


plotter.show(screenshot=save_folder + "meshclip.png")




# Generate points
points = generate_half_sine_curve(duration=0.1, num_points=100)

# Save results in the specified text format
with open("/home/sarse/Desktop/UniCode/UniversityCODE/CFD/Assignment/3-lagrangian/half_sine_curve.txt", "w") as file:
    for x, y in points:
        file.write(f"({x:.5f} {y:.5f})\n")

# Extract x and y for plotting
x_vals, y_vals = zip(*points)

# Plot the curve
plt.figure(figsize=(8, 6))
plt.plot(x_vals, y_vals, label="Half-Sine Injection Profile", linewidth=3)

# Add a vertical dashed red line at x = 0.08 s
plt.axvline(x=0.08, color="red", linestyle="--", linewidth=2, label="Simulation End Time")

plt.xlabel("Time, t [s]")
plt.ylabel("Normalized Injected Mass [-]")
#plt.title("Half-Sine Injection Profile")
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()

save_filename = "injector.png"

save_filename = save_folder + save_filename

# Save the plot
plt.savefig(save_filename)
plt.show()


'''

'''

import pyvista as pv
import numpy as np
import matplotlib.pyplot as plt

# Percorso del file VTK
vtk_folder = "/home/sarse/OpenFOAM/sarse-12/run/Assignment/prova2/VTK/"
file = "prova2_3888.vtk"

# Caricare il file VTK
mesh = pv.read(vtk_folder + file)

# Creare subplots per pressione, temperatura e velocità
pl = pv.Plotter(off_screen=True, shape=(1, 3))

# Creare uno slice parallelo al piano Z all'origine
slice_z_pressure = mesh.slice(normal=(0, 0, 1), origin=(0, 0, 0))
slice_z_velocity = mesh.slice(normal=(0, 0, 1), origin=(0, 0, 0))
slice_z_temp = mesh.slice(normal=(0, 0, 1), origin=(0, 0, 0))

# Parametri di visualizzazione
sargs = dict(
            title_font_size=18,
            label_font_size=20,
            n_labels=3,
            position_x=0.22,  # Posizionamento specifico
            position_y=0.05,
            vertical=False,
            height=0.1,
            fmt="%.f"
        )

# Plot pressione
pl.subplot(0, 0)
pl.view_xy()
pl.camera.zoom(0.5)

# Ottieni la posizione attuale della camera e il punto focale
current_position = pl.camera.GetPosition()  # Posizione attuale della camera
current_focal_point = pl.camera.GetFocalPoint()  # Punto focale attuale

# Definisci la traslazione verticale lungo l'asse Y
delta_y = +0.1  # Traslazione verso il basso; usa un valore positivo per alzare

# Calcola la nuova posizione e il nuovo punto focale
new_position = (
    current_position[0],  # x rimane invariato
    current_position[1] + delta_y,  # Traslazione su y
    current_position[2]   # z rimane invariato
)

new_focal_point = (
    current_focal_point[0],  # x rimane invariato
    current_focal_point[1] + delta_y,  # Traslazione su y
    current_focal_point[2]   # z rimane invariato
)

# Imposta la nuova posizione e il nuovo punto focale
pl.camera.SetPosition(new_position)
pl.camera.SetFocalPoint(new_focal_point)
        
pl.add_mesh(slice_z_pressure, scalars="p", cmap="coolwarm", show_edges=False, scalar_bar_args=sargs)
pl.add_title("Pressure")
pl.add_axes()

# Parametri di visualizzazione
sargs = dict(
            title_font_size=18,
            label_font_size=20,
            n_labels=5,
            position_x=0.22,  # Posizionamento specifico
            position_y=0.05,
            vertical=False,
            height=0.1,
            fmt="%.f"
        )

# Plot temperatura
pl.subplot(0, 1)
pl.view_xy()
pl.camera.zoom(0.5)
# Imposta la nuova posizione e il nuovo punto focale
pl.camera.SetPosition(new_position)
pl.camera.SetFocalPoint(new_focal_point)
pl.add_mesh(slice_z_temp, scalars="T", cmap="coolwarm", show_edges=False, scalar_bar_args=sargs)
pl.add_title("Temperature")
pl.add_axes()


sargs = dict(
            title_font_size=18,
            label_font_size=20,
            n_labels=5,
            position_x=0.22,  # Posizionamento specifico
            position_y=0.05,
            vertical=False,
            height=0.1,
            fmt="%.f"
        )

# Plot velocità
pl.subplot(0, 2)
pl.view_xy()
pl.camera.zoom(0.5)
# Imposta la nuova posizione e il nuovo punto focale
pl.camera.SetPosition(new_position)
pl.camera.SetFocalPoint(new_focal_point)
pl.add_mesh(slice_z_velocity, scalars="U", cmap="coolwarm", show_edges=False, scalar_bar_args=sargs)
pl.add_title("Velocity")
pl.add_axes()

save_filename = "fields_visualizationXY.png"
pl.show(screenshot=save_folder + save_filename)

'''


'''

import pyvista as pv
import numpy as np
import matplotlib.pyplot as plt

# Percorso del file VTK
vtk_folder = "/home/sarse/OpenFOAM/sarse-12/run/Assignment/prova2/VTK/"
file = "prova2_3888.vtk"

# Caricare il file VTK
mesh = pv.read(vtk_folder + file)
print("Campi disponibili:", mesh.array_names)

# Creare subplots per pressione, temperatura e velocità
pl = pv.Plotter(off_screen=True, shape=(3, 1))

# Creare uno slice parallelo al piano X all'origine
slice_x_pressure = mesh.slice(normal=(1, 0, 0), origin=(0, 0, 0))
slice_x_velocity = mesh.slice(normal=(1, 0, 0), origin=(0, 0, 0))
slice_x_temp = mesh.slice(normal=(1, 0, 0), origin=(0, 0, 0))

# Parametri di visualizzazione
sargs = dict(
    title_font_size=18,
    label_font_size=18,
    n_labels=5,
    position_x=0.9,  # Posizionamento specifico
    position_y=0.03,
    vertical=True,
    height=0.8,
    fmt="%.f"
)

# Plot pressione
pl.subplot(0, 0)
pl.view_yz()
pl.camera.zoom(2)
current_position = pl.camera.GetPosition()  # Posizione attuale della camera
current_focal_point = pl.camera.GetFocalPoint()  # Punto focale attuale

# Definisci la traslazione verticale lungo l'asse Y
delta_y = +0.1  # Traslazione verso il basso; usa un valore positivo per alzare

# Calcola la nuova posizione e il nuovo punto focale
new_position = (
    current_position[0],  # x rimane invariato
    current_position[1] + delta_y,  # Traslazione su y
    current_position[2]   # z rimane invariato
)

new_focal_point = (
    current_focal_point[0],  # x rimane invariato
    current_focal_point[1] + delta_y,  # Traslazione su y
    current_focal_point[2]   # z rimane invariato
)

# Imposta la nuova posizione e il nuovo punto focale
pl.camera.SetPosition(new_position)
pl.camera.SetFocalPoint(new_focal_point)

pl.add_mesh(slice_x_pressure, scalars="p", cmap="coolwarm", n_colors=20, show_edges=False, scalar_bar_args=sargs)
#pl.add_title("Pressure")
pl.add_axes()

# Plot temperatura
pl.subplot(1, 0)
pl.view_yz()
pl.camera.zoom(2)

# Imposta la nuova posizione e il nuovo punto focale
pl.camera.SetPosition(new_position)
pl.camera.SetFocalPoint(new_focal_point)

pl.add_mesh(slice_x_temp, scalars="T", cmap="coolwarm", n_colors=20, show_edges=False, scalar_bar_args=sargs)
#pl.add_title("Temperature")
pl.add_axes()

# Plot velocità
pl.subplot(2, 0)
pl.view_yz()
pl.camera.zoom(2)

# Imposta la nuova posizione e il nuovo punto focale
pl.camera.SetPosition(new_position)
pl.camera.SetFocalPoint(new_focal_point)

pl.add_mesh(slice_x_velocity, scalars="U", cmap="coolwarm", n_colors=20, show_edges=False, scalar_bar_args=sargs)
#pl.add_title("Velocity")
pl.add_axes()


save_filename = "fields_visualizationZY.png"
pl.show(screenshot=save_folder + save_filename)

'''





##################################################################################
#                               RESIDUAL
#################################################################################

file_name = "/home/sarse/Desktop/UniCode/UniversityCODE/CFD/Assignment/5-final/postProcessing/residuals(p,U,h,k,epsilon)/0/residuals.dat"
save_filename = "/home/sarse/Desktop/UniCode/UniversityCODE/CFD/Assignment/5-final/residual.png"

#plot_residuals(file_name, save_filename)


######################################################################
# PATCHES
#######################################################################


import pandas as pd
import matplotlib.pyplot as plt

# Specifica il nome del file .dat
file_path = "/home/sarse/Desktop/UniCode/UniversityCODE/CFD/Assignment/5-final/postProcessing/patchAverageOutlet/0/surfaceFieldValue.dat"  # Modifica con il percorso corretto

# Leggere il file .dat, saltando eventuali righe di intestazione
columns = ["Time", "Pressure", "Temperature"]
dataOutlet = pd.read_csv(file_path, delim_whitespace=True, names=columns, skiprows=4)

file_path = "/home/sarse/Desktop/UniCode/UniversityCODE/CFD/Assignment/5-final/postProcessing/patchAverageInlet/0/surfaceFieldValue.dat"
dataInlet = pd.read_csv(file_path, delim_whitespace=True, names=columns, skiprows=4)


plt.ioff()
# Creazione del primo grafico per la pressione e salvataggio
plt.figure(figsize=(15, 9))
plt.plot(dataOutlet["Time"], dataOutlet["Pressure"], label="Outlet", color='b')
plt.plot(dataInlet["Time"], dataInlet["Pressure"], label="Inlet", color='r')
plt.xlabel("Time [s]")
plt.ylabel("Pressure [Pa]")
#plt.title("Time vs Pressure")
plt.legend()
plt.grid()
plt.savefig(save_folder+"time_vs_pressure.png", dpi=300)  # Salvataggio immagine
plt.close()  # Chiusura della figura per evitare sovrapposizioni

# Creazione del secondo grafico per la temperatura e salvataggio
plt.figure(figsize=(15, 9))
plt.plot(dataOutlet["Time"], dataOutlet["Temperature"], label="Outlet", color='b')
plt.plot(dataInlet["Time"], dataInlet["Temperature"], label="Inlet", color='r')
plt.xlabel("Time [s]")
plt.ylabel("Temperature [K]")
#plt.title("Time vs Temperature")
plt.legend()
plt.grid()
plt.savefig(save_folder+"time_vs_temperature.png", dpi=300)  # Salvataggio immagine
plt.close()  # Chiusura della figura





###############################################################
''' LAGRANGIAN SPRAY
import pyvista as pv
import pandas as pd
import matplotlib.pyplot as plt

# Percorso dei file VTK
vtk_folder = "/home/sarse/Desktop/UniCode/UniversityCODE/CFD/Assignment/5-final/VTK/lagrangian/cloud/"
vtk_files = ("cloud_5381.vtk", "cloud_13442.vtk", "cloud_21471.vtk")

# Caricare la mesh principale per la geometria
geometry_mesh = pv.read("/home/sarse/Desktop/UniCode/UniversityCODE/CFD/Assignment/5-final/VTK/geometry.vtk")

# Caricare il file VTK contenente le particelle
particles_mesh = pv.read(vtk_folder + "cloud_13442.vtk")
print("Campi disponibili:", particles_mesh.array_names)

# Creare il plotter
pl = pv.Plotter()

# Aggiungere la mesh di contorno molto opaca
pl.add_mesh(geometry_mesh, opacity=0.1, color='gray')

# Aggiungere le particelle come sfere nere
particles_mesh = particles_mesh.glyph(scale="d", orient="U", factor=10)

pl.add_mesh

# Mostrare la visualizzazione
pl.show()

'''



###############################################################

import pyvista as pv

import pyvista as pv

import pyvista as pv

def plot_scalar_field(vtk_folder, vtk_files, save_folder, scalar, titleBar, labelScale = "%.f", cmap = "coolwarm"):
    timestamps = {"0.02s": 0, "0.05s": 1, "0.08s": 2}  # Indici dei file corrispondenti
    
    # Creare il plotter con un unico subplot verticale (YZ view)
    pl = pv.Plotter(shape=(len(timestamps), 1), off_screen=True)
    # Vista YZ con zoom
    pl.view_yz()
        
    # Ottieni la posizione attuale della camera
    current_position = pl.camera.GetPosition()
    current_focal_point = pl.camera.GetFocalPoint()

    # Definisci la traslazione verticale lungo l'asse Y
    delta_y = 0.1  # Traslazione verso il basso; usa un valore positivo per alzare

    # Calcola la nuova posizione e il nuovo punto focale
    new_position = (
        current_position[0],
        current_position[1] + delta_y,
        current_position[2]
    )

    new_focal_point = (
        current_focal_point[0],
        current_focal_point[1] + delta_y,
        current_focal_point[2]
    )

        
    for timestamp, index in timestamps.items():
        file = vtk_files[index]
        mesh = pv.read(vtk_folder + file)
        
        # Selezionare il subplot corrispondente
        pl.subplot(index, 0)
        
        # Imposta la nuova posizione e il nuovo punto focale
        pl.camera.SetPosition(new_position)
        pl.camera.SetFocalPoint(new_focal_point)
        
        pl.camera.zoom(1.5)
        
        # Creare uno slice parallelo al piano X all'origine
        slice_x = mesh.slice(normal=(1, 0, 0), origin=(0, 0, 0))
        
        sargs = dict(
            title=f"                           {titleBar}                             {index + 1}",
            title_font_size=18,
            label_font_size=18,
            n_labels=5,
            position_x=0.85,
            position_y=0.15,
            vertical=True,
            height=0.7,
            fmt=labelScale
        )
        
        # Aggiungere il campo scalare al plot
        pl.add_mesh(slice_x, scalars=scalar, cmap=cmap, show_edges=False, scalar_bar_args=sargs)
        pl.add_text(f"Time: {timestamp}", position="upper_left", font_size=14)
    
    pl.add_axes()
    
    # Salvataggio dell'immagine per vista ZX
    save_filename = scalar + "_zxView.png"
    pl.show(screenshot=save_folder + save_filename)
    
    # Creare il plotter con un unico subplot orizzontale (XY view)
    pl = pv.Plotter(shape=(len(timestamps), 1), off_screen=True)  # Cambiato da (1,4) a (1,3) per gestire il numero corretto di subplot
    # Vista YZ con zoom
    pl.view_yx()
        
    # Ottieni la posizione attuale della camera
    current_position = pl.camera.GetPosition()
    current_focal_point = pl.camera.GetFocalPoint()

    # Definisci la traslazione verticale lungo l'asse Y
    delta_y = 0.1 # Traslazione verso il basso; usa un valore positivo per alzare

    # Calcola la nuova posizione e il nuovo punto focale
    new_position = (
        current_position[0],
        current_position[1] + delta_y,
        current_position[2]
    )

    new_focal_point = (
        current_focal_point[0],
        current_focal_point[1] + delta_y,
        current_focal_point[2]
    )
    
    for timestamp, index in timestamps.items():
        file = vtk_files[index]
        mesh = pv.read(vtk_folder + file)
        
        # Selezionare il subplot corrispondente
        pl.subplot(index, 0)
        pl.view_yx()
        
        # Imposta la nuova posizione e il nuovo punto focale
        pl.camera.SetPosition(new_position)
        pl.camera.SetFocalPoint(new_focal_point)
        
        pl.camera.zoom(1.5)
        # Creare uno slice parallelo al piano Z all'origine
        slice_z = mesh.slice(normal=(0, 0, 1), origin=(0, 0, 0))
        
        '''
        sargs = dict(
            title=f"                           {titleBar}                           {index + 1}",
            title_font_size=18,
            label_font_size=18,
            n_labels=5,
            position_x=0.22,
            position_y=0.05,
            vertical=False,
            height=0.1,
            fmt=labelScale
        )
        '''
        sargs = dict(
            title=f"                           {titleBar}                             {index + 1}",
            title_font_size=18,
            label_font_size=18,
            n_labels=5,
            position_x=0.85,
            position_y=0.15,
            vertical=True,
            height=0.7,
            fmt=labelScale
        )
        
        # Aggiungere il campo scalare al plot
        pl.add_mesh(slice_z, scalars=scalar, cmap=cmap, show_edges=False, scalar_bar_args=sargs)
        pl.add_text(f"Time: {timestamp}", position="upper_left", font_size=14)

    pl.add_axes()
    
    # Salvataggio dell'immagine per vista XY
    save_filename = scalar + "_xyView.png"
    pl.show(screenshot=save_folder + save_filename)
    

# Chiamare la funzione
plot_scalar_field(vtk_folder, vtk_files, save_folder, scalar="U", titleBar="Velocity (m/s)")

plot_scalar_field(vtk_folder, vtk_files, save_folder, scalar="p", titleBar="Pressure (Pa)", labelScale="%.f")

plot_scalar_field(vtk_folder, vtk_files, save_folder, scalar="T", titleBar="Temperature (K)", cmap = "coolwarm"  )

plot_scalar_field(vtk_folder, vtk_files, save_folder, scalar="Ma", titleBar="Mach (-)", labelScale="%.2f")

plot_scalar_field(vtk_folder, vtk_files, save_folder, scalar="k", titleBar="TKE (m^2/s^2)")

plot_scalar_field(vtk_folder, vtk_files, save_folder, scalar="Co", titleBar="Courant Number (-)", labelScale = "%.2f")

plot_scalar_field(vtk_folder, vtk_files, save_folder, scalar="Qdot", titleBar="Heat Transfer (W)", labelScale="%.2e", cmap = "coolwarm")

plot_scalar_field(vtk_folder, vtk_files, save_folder, scalar="C7H16", titleBar="C7H16", labelScale = "%.3f")

plot_scalar_field(vtk_folder, vtk_files, save_folder, scalar="CO2", titleBar="CO2", labelScale = "%.3f")

plot_scalar_field(vtk_folder, vtk_files, save_folder, scalar="H2O", titleBar="H2O", labelScale = "%.3f")

plot_scalar_field(vtk_folder, vtk_files, save_folder, scalar="N2", titleBar="N2", labelScale = "%.3f")

'''


#####################################################
#                         SPRAY
#######################################################
'''
'''
# Path to the log file
log_file = "/home/sarse/Desktop/UniCode/UniversityCODE/CFD/Assignment/5-final/log.foamRun"

# Initialize lists to store extracted data
times = []
penetrations = []

# Read the log file and extract relevant data
with open(log_file, "r") as file:
    for line in file:
        # Extract time information
        time_match = re.search(r"Time = ([\d\.e\-]+)s", line)
        if time_match:
            current_time = float(time_match.group(1))
        
        # Extract penetration information
        penetration_match = re.search(r"Liquid penetration 95% mass \(m\) = ([\d\.e\-]+)", line)
        if penetration_match:
            penetration = float(penetration_match.group(1))
            times.append(current_time)
            penetrations.append(penetration)

# Save the data to a CSV file
data = pd.DataFrame({"Time [s]": times, "Penetration [m]": penetrations})
data.to_csv(save_folder + "spray_penetration.csv", index=False)

# Plot the data
plt.figure(figsize=(8, 6))
plt.plot(data["Time [s]"], data["Penetration [m]"], label="95% Spray Penetration")
plt.xlabel("Time [s]")
plt.ylabel("Penetration [m]")
#plt.title("Spray Penetration Over Time (95% Mass)")
plt.grid(True)
plt.legend()
plt.savefig(save_folder + "spray_penetration_plot.png")
plt.show()
'''



####################################################################################################
#                               ISOSURFACE
#############################################################################################

import numpy as np
import pyvista as pv

def generate_isosurface(folder, files_time, times, save_folder, base_filename, geometry_vtk, stoichiometric_value=15.06, delta=1):
    """
    Genera e salva un'immagine di isosuperfici per il rapporto di miscelazione SF_RATIO
    nei file VTK, salvando un'immagine separata per ogni tempo.

    Parametri:
    - folder: Percorso della cartella contenente i file VTK.
    - files_time: Lista dei nomi dei file VTK.
    - times: Lista dei tempi corrispondenti ai file.
    - save_folder: Cartella di destinazione per le immagini.
    - base_filename: Nome base del file immagine da salvare.
    - stoichiometric_value: Valore stechiometrico di riferimento.
    - delta: Intervallo per le isosuperfici.
    """

    # Lista di colori ciclici per distinguere le isosuperfici
    colors = ["red", "blue", "green", "orange", "purple", "cyan", "yellow", "pink", "brown", "gray"]
    
    # Caricare la mesh principale per la geometria
    geometry_mesh = pv.read(geometry_vtk)


    # Itera su tutti i file per creare e salvare un'immagine per ogni tempo
    for i, (file_name, time) in enumerate(zip(files_time, times)):
        # Costruisce il nome del file di output con il tempo
        save_filename = f"{base_filename}_T={time:.2f}s.png"
        save_path = save_folder + save_filename

        # Inizializza il plotter in modalità off-screen
        pl = pv.Plotter(off_screen=True)
        pl.camera_position= (1, -0.6, 1)
        
        pl.add_mesh(geometry_mesh, color = "gray", opacity  = 0.1)

        # Carica il file VTK
        mesh = pv.read(folder + file_name)

        # Calcola il campo scalare SF_RATIO
        Y_O2 = mesh.point_data["O2"]
        Y_N2 = mesh.point_data["N2"]
        Y_fuel = mesh.point_data["C7H16"]
        SF_RATIO = np.zeros_like(Y_fuel)

        # Evita divisioni per zero e assegna valori validi
        non_zero_fuel = Y_fuel > 0.00001
        SF_RATIO[non_zero_fuel] = (Y_O2[non_zero_fuel] + Y_N2[non_zero_fuel]) / Y_fuel[non_zero_fuel]
        SF_RATIO[~non_zero_fuel] = np.nan
        mesh.point_data["SF_RATIO"] = SF_RATIO

        # Genera l'isosuperficie basata su SF_RATIO
        isosurface = mesh.contour(
            isosurfaces=[stoichiometric_value],  # Isosuperficie al valore stechiometrico
            scalars="SF_RATIO"
        )

        # Verifica se l'isosuperficie è vuota
        if isosurface.n_points == 0:
            print(f"Nessuna isosuperficie trovata per il file: {file_name} (T = {time:.2f}s)")
            continue

        # Seleziona un colore ciclicamente
        color = colors[i % len(colors)]

        # Aggiunge l'isosuperficie alla scena
        pl.add_mesh(isosurface, color=color, opacity=0.6, label=f"T = {time:.2f} s")

        # Aggiunge la legenda e gli assi
        pl.add_axes()
        #pl.add_legend()

        # Salva l'immagine per il tempo attuale
        pl.show(screenshot=save_path)
        print(f"Isosuperficie salvata in: {save_path}")

# Cartella in cui si trovano i file VTK
save_folder = "/home/sarse/Desktop/UniCode/UniversityCODE/CFD/Assignment/5-final/"

# Caricare la mesh principale per la geometria
geometry_mesh = pv.read("/home/sarse/Desktop/UniCode/UniversityCODE/CFD/Assignment/5-final/VTK/geometry.vtk")

# Nome base del file immagine
base_filename = "isosurface"

# Lista di file VTK e tempi corrispondenti
times = [0.02, 0.05, 0.08]  # Tempi simulati

generate_isosurface(vtk_folder, vtk_files, times, save_folder, base_filename, geometry_mesh, stoichiometric_value=15.08)


############ PARCEL diameter
# Percorso dei file VTK
vtk_folderLag = "/home/sarse/Desktop/UniCode/UniversityCODE/CFD/Assignment/5-final/VTK/lagrangian/cloud/"
vtk_filesLag = ("cloud_5427.vtk", "cloud_16366.vtk", "cloud_31234.vtk")

file_name = save_folder  + "diameter_histograms.png"

plot_diameter_histograms(vtk_folderLag, vtk_filesLag, 20, file_name, timeToPlot = [0.02, 0.05, 0.08])




def plot_species_concentrations(folder, files, save_path, timeToPlot):
    """
    Plotta le concentrazioni in percentuale di H₂O, C₇H₁₆, CO₂, O₂ e N₂
    sull'intero dominio ai diversi tempi da file VTK.

    Args:
        folder (str): Cartella contenente i file VTK.
        files (list of str): Lista dei file VTK per i diversi tempi.
        save_path (str): Percorso per salvare il grafico.
        timeToPlot (list of str): Lista dei tempi corrispondenti ai file VTK per i titoli.
    """
    species = ["H2O", "C7H16", "CO2", "O2", "N2"]
    num_species = len(species)
    num_times = len(files)
    
    # Setup del grafico
    fig, axs = plt.subplots(1, num_times, figsize=(5 * num_times, 6), sharey=True)
    
    for i, file in enumerate(files):
        vtk_file = folder + file
        mesh = pv.read(vtk_file)
        
        # Controlla se tutte le specie sono disponibili
        missing_species = [sp for sp in species if sp not in mesh.array_names]
        if missing_species:
            print(f"Le seguenti specie non sono disponibili nel file {file}: {missing_species}")
            continue
        
        # Estrai i campi delle concentrazioni
        concentrations = np.array([mesh[sp] for sp in species])  # Matrice (num_species, num_cells)
        total_concentration = np.sum(concentrations, axis=0)  # Somma su tutte le specie
        
        # Calcola le percentuali
        percentages = (concentrations / total_concentration) * 100  # Normalizza in percentuale
        mean_percentages = np.mean(percentages, axis=1)  # Media sul dominio
        
        # Plot delle concentrazioni in percentuale
        axs[i].bar(species, mean_percentages, color=["blue", "red", "green", "orange", "purple"], alpha=0.7)
        axs[i].set_xlabel("Species")
        axs[i].set_title(f"Time = {timeToPlot[i]} s")
        axs[i].grid(True, linestyle="--", alpha=0.6)
    
    # Configurazione dell'asse y
    axs[0].set_ylabel("Concentration [%]")
    
    # Salva e mostra il grafico
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()
    
    
    
    timeToPlot = [0.02, 0.05, 0.08]
    
    
    save_path = save_folder + "concentrationOverTime.png"
    plot_species_concentrations(vtk_folder, vtk_files, save_path, timeToPlot)
    
    
    




















