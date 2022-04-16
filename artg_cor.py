import cv2
import glob
import numpy as np
from utils import draw_3d, detector, superimpose as si

# Função utilizada para carregar e passar cada imagem da pasta por vez
path = glob.glob("imgs\*.jpeg")
for image in path:

    # Lê a imagem e cria uma cópia cortada (Apenas a região da simulação do Gazebo)  
    img = cv2.imread(image)
    copia_img = img.copy()

    # Aplica filtros para identificar o contorno dos ARTAGs na imagem
    blueLow = np.array([60, 50, 40])
    blueHigh = np.array([130, 255, 255])
    imgHSV = cv2.cvtColor(copia_img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(imgHSV, blueLow, blueHigh)
    imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    kernelOpen = np.ones((5,5))
    kernelClose = np.ones((20,20))
    maskOpen = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernelOpen)
    maskClose = cv2.morphologyEx(maskOpen, cv2.MORPH_CLOSE, kernelClose)

    imgray = cv2.GaussianBlur(imgray, (5,5), 0)
    imgray = cv2.bilateralFilter(imgray,9,75,75)

    edge = cv2.Canny(mask, 50, 200)
    edge = cv2.dilate(edge, None, iterations=1)
    edge = cv2.erode(edge, None, iterations=1)
    contours, hierarchy = cv2.findContours(edge, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) 

    # Loop para identificar os contornos
    for c in contours:
     perimeter = cv2.arcLength(c,True)

     if perimeter > 32 and perimeter < 70:  # Condição criada para otimizar a identificação do ARTAG
         #cv2.drawContours(copia_cortada, [c], 0, (0,0,255), 3) #Caso queira desenhar os contornos

         # Corta o ARTAG
         x,y,w,h = cv2.boundingRect(c)
         crop1 = copia_img[y:y+h, x:x+w]
         copia_crop = crop1.copy()

         # Função criada para pegar os pixels cinzas do ARTAG e converter para Branco, a fim de facilitar a leitura
         copia_crop[np.where((copia_crop==[41,41,41]).all(axis=2))] = [255,255,255]

         cv2.imshow('Imagem', copia_crop)
         cv2.waitKey()
         cv2.destroyAllWindows()
    