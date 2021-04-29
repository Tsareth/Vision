import numpy as np
import cv2
from vidstab import VidStab

def main():
    video = cv2.VideoCapture('VideoEjemploRecortado.mp4')       # Apertura del video a analizar
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))                 # Se obtiene la resolucion del video de entrada
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

    #stabilizer = VidStab()                                       # Estabilizacion del video
    #stabilizer.stabilize(input_path='VideoEjemploRecortado.mp4',output_path='estabilizado.mp4',max_frames=100, playback=True)
    #estabilizado = cv2.VideoCapture('estabilizado.mp4')
    
    fgbg = cv2.createBackgroundSubtractorMOG2(history = 500,varThreshold = 16,detectShadows = False)        #Creacion de operando de algoritmo de Gaussianas mixtas para separacion de fondo
    while(video.isOpened()):                                    #REvisa si el video se encuentra abierto       
        ret, frame = video.read()                               #Obtiene la informacion del video
        fgmask = fgbg.apply(frame)                              #Crea una mascara con el operando de MOG2 aplicado
        
        fotograma_filtrado = IdentificarCarros(frame, fgmask, width - 920, height - 740, width, height)          # Filtrar y etiquetar carros en fotograma
        fotograma_filtrado = cv2.resize(fotograma_filtrado, (1280, 720), interpolation=cv2.INTER_AREA)          # Convertir a 720p (NO ESTA FUNCIONANDO)
        
        cv2.imshow('Resultado',fotograma_filtrado)                      #Abre en una pantalla el video con el algoritmo MOG2 aplicado
        k = cv2.waitKey(30) & 0xff                              #Crea un delay de 30 milisegundos para que la pantalla sea capaz de realizar las operaciones 
        if k == 27:
            break
        if cv2.waitKey(30) & 0xFF == ord('q'):                  #Si la tecla q es presionada, se detiene la presentacion de la mascara creada
            break
    
    #estabilzado.release()                                      #Se libera el video para evitar llenado inecesario de memoria
    video.release()                                             
    cv2.destroyAllWindows()                                     #Se destruyen las pantallas creadas por la aplicacion
    print('Borrado exitoso')



def IdentificarCarros(fotograma, mascara, x0, y0, x1, y1):

    crop = mascara[y0:y1, x0:x1]                  # Se corta la zona de RDI de la mascara

    kernel = np.ones((5,5),np.uint8)
    filtro1 = cv2.morphologyEx(crop, cv2.MORPH_OPEN, kernel)    #Operacion conjunta de erosion + dilatacion
    filtro2 = cv2.dilate(filtro1,kernel,iterations = 1)         # Dilatacion

    contours, hierarchy = cv2.findContours(filtro2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) # Extraccion de contornos

    filtro3 = filtro2
    for cnt in contours:                                        # Demarcacion de contornos
        if cv2.contourArea(cnt) > 200:                          # Area de contorno debe ser mas grande que 200
            x,y,w,h = cv2.boundingRect(cnt)
            cv2.rectangle(fotograma,(x0+x,y0+y),(x0+x+w,y0+y+h),(0,255,0),2)  # Dibujar rectangulo
    
    cv2.imshow('Mascara RDI', filtro2)
    
    ident = fotograma                           # imagen base
    ident = cv2.rectangle(fotograma, (x0, y0), (x1, y1), (255, 0, 0), 2) # dibujar rectangulo de RDI
    
    return ident
    
    

    
