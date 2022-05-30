#%%
"""
Universidad Nacional del Sur (UNS)
Departamento de Ingenieria electrica y de computadoras
Laboratorio de ciencias de las imagenes (LCI)
@authors: Steven Martinez, Nicolas Parma
Version: 30 Mayo 2022
Environment: Python 3.9.7 (miniconda)
"""
#%%
# Librerias
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from skimage.color import rgb2gray
from skimage.filters import gaussian
from skimage.segmentation import active_contour
import imageio
import glob
import os
import re
#%%
def key_func(x):
    pat = re.compile("(\d+)\D*$")
    mat = pat.search(os.path.split(x)[-1]) # match last group of digits
    if mat is None:
        return x
    return "{:>10}".format(mat.group(1)) # right align to 10 digits.

# Implementación de la formula de Shoelace para obtener el area de un conjunto de puntos en el plano
def PolyArea(x,y):
    return 0.5*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))

# Centros y radios del contorno de inicialización (unidades en pixeles)
def center_0(url):
    img = imageio.imread(url)
    img = rgb2gray(img)
    shape = img.shape    
    cx = shape[1]//2 + 10
    cy = shape[0]//2 - 15
    return cx, cy

def draw_circle(url):
    cx, cy = center_0(url)
    radInit = 210
    #Alpha depende de la cantidad de puntos, se crea la circunferencia inicial
    s = np.linspace(0, 2*np.pi, 800)
    r = cy + radInit * np.sin(s)
    c = cx + radInit * np.cos(s)
    init = np.array([r, c]).T
    #Radio de la probeta en pixeles (distinto al de inicialización para obtener mejores resultados)
    radTest = 240
    rprob = cy + radTest * np.sin(s)
    cprob = cx + radTest * np.cos(s)
    prob = np.array([rprob, cprob]).T
    #Area en pixeles de la probeta
    areaTestTube = np.pi * (radTest)**2
    return init, prob, areaTestTube

# Obtener contorno y area del poligono encontrado
def contour(url, image, alpha, beta, gamma):
    
    #Condiciones reales de la probeta
    cx, cy = center_0(url)
    MradTestTube = 350
    MareaTestTube = np.pi * (MradTestTube)**2 #Area de la probeta

    init,_,areaTestTube = draw_circle(url)
    snake = active_contour(image,init, alpha=alpha, beta=beta, gamma = gamma,convergence=0.000001,max_num_iter=5000)
    areaSnake = PolyArea(snake[:,0],snake[:,1])
    radEquiv = np.sqrt(areaSnake/np.pi) 
    s = np.linspace(0, 2 * np.pi, 800)
    er = cy + radEquiv * np.sin(s)
    ec = cx + radEquiv * np.cos(s)
    equivCircle = np.array([er, ec]).T

    MareaSnake = areaSnake * MareaTestTube/areaTestTube #Ajuste a escala real de la probeta
    MradEquiv = np.sqrt(MareaSnake/np.pi)   
    return snake, areaSnake, equivCircle, MareaSnake, MradEquiv

# alppha, beta, gamma y smoothing, son parametros de ajuste del algoritmo 
# Sus valores varian en base a pruebas empiricas con las imagenes

def main(folder_input, name_im, folder_output, alpha_1, alpha_2, beta_1, beta_2, gamma_1, gamma_2, smoothing, figure=True):

    url = folder_input + '/' + name_im
    init, prob, _ = draw_circle(url)
    cx, cy = center_0(url)

    #Las unidades se encuentran en micrómetros
    MradTestTube = 350
    MareaTestTube = np.pi*(MradTestTube)**2
    MareaTestTube = np.round(MareaTestTube,2)
    print(f"Área de la probeta (um): {MareaTestTube}")

    imglist = []
    arealist = []
    radEquivlist = []
    path = folder_input + '/*.*' #Batch
    img_number = 1 
    #Iteración por nombre consecutivo de las imagenes
    for file in sorted(glob.glob(path),key = key_func):

        if img_number < 10:
            alpha = alpha_1
            beta = beta_1
            gamma = gamma_1
        else:
            alpha = alpha_2
            beta = beta_2
            gamma = gamma_2

        img = imageio.imread(file)
        img = rgb2gray(img)

        gimage = gaussian(img, smoothing)  
        snake, areaSnake, equivCircle, MareaSnake, MradEquiv = contour(url, gimage, alpha, beta, gamma)     
        # Si el contorno anterior es mayor o menor a un 50% se corrige el valor de alpha
        if (img_number > 1) and (MareaSnake < 0.5*arealist[-1]) and (MareaSnake > 3000):
            alpha -= 0.1
           # print(f"Area nueva menor al 50% de la anterior, alpha: {alpha} para {file}")
            snake, areaSnake, equivCircle, MareaSnake, MradEquiv = contour(url, gimage, alpha, beta, gamma)
        elif MareaSnake <= 3000:
            alpha = alpha/2
          #  print(f"Area anterior menor a 3000, alpha: {alpha} para {file}")
            snake, areaSnake, equivCircle, MareaSnake, MradEquiv = contour(url, gimage, alpha, beta, gamma)          
        elif (img_number > 1) and (MareaSnake > 1.5*arealist[-1]):
            alpha += 0.1
          #  print(f"Area nueva mayor al 150% de la anterior, alpha: {alpha} para {file}")
            snake, areaSnake, equivCircle, MareaSnake, MradEquiv = contour(url, gimage, alpha, beta, gamma)
       
        MareaSnake = np.round(MareaSnake, 2)
        MradEquiv = np.round(MradEquiv, 2)
        print(f'Área del contorno (um): {MareaSnake} para ' 
                                                        + 'deteccion' + str(img_number)+'.jpg')       
        
        # Figuras    
        if figure: 

            font = {'family': 'serif',
            'color':  'white',
            'weight': 'bold',
            'size': 10,}

            #plt.ioff()
            fig, ax = plt.subplots(figsize=(7, 7), dpi=90)
            ax.scatter(cx,cy, lw = 1,c = '#ffffff')
            ax.imshow(gimage, cmap=plt.cm.gray)
            ax.plot(init[:, 1], init[:, 0], '--r', lw=3)
            ax.plot(snake[:, 1], snake[:, 0], '-b', lw=3)
            ax.plot(equivCircle[:, 1], equivCircle[:, 0], '--g', lw=3)
            ax.plot(prob[:, 1], prob[:, 0], 'go--', lw=3)
            ax.text(7, 20, 'Radio um = '+ str(MradEquiv), fontdict=font)
            ax.text(7, img.shape[0] - 20, 'Área um2 = '+str(MareaSnake), fontdict=font)
            ax.set_xticks([]), ax.set_yticks([])
            ax.axis([0, img.shape[1], img.shape[0], 0])
            fig.savefig(folder_output + '/deteccion'+str(img_number)+'.jpg',bbox_inches="tight")
            #plt.show()

        imglist.append(file)
        arealist.append(MareaSnake)
        radEquivlist.append(MradEquiv)
        img_number += 1

    # Exportar datos a excel
        data = pd.DataFrame({'Nombre': imglist,'Area um^2':arealist,'radio eq um':radEquivlist})
        file_name = folder_output + '/' + 'may_30.xlsx'
        data.to_excel(file_name)

#%%
if __name__ == '__main__':
    #Ingresar la ruta de las imagenes en name_folder_images
    #Ingresar la ruta donde se guardaran las imágenes segmentadas y el archivo de datos en  name_folder_output
    name_folder_images = '/Users/stevenmartine/Drive smarvar/Proyectos_extra/Segmentación Celular/Detector_Parma/2_ensayo'
    name_folder_output = '/Users/stevenmartine/Drive smarvar/Proyectos_extra/Segmentación Celular/Detector_Parma/Output_may30'
    name_image_1 = 'ENSAYO1_1.jpg'
    #Parametros para 2_ensayo
    main(name_folder_images, name_image_1, name_folder_output, 
        alpha_1=0.53, beta_1=0.4, gamma_1=0.05,
        alpha_2=0.37,beta_2=0.1, gamma_2=0.005,
        smoothing=1.1, figure=True)


# %%
