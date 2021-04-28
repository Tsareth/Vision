import numpy as np
import cv2





def Main ():
    video = cv2.VideoCapture('VideoEjemploRecortado.mp4')        # Apertura del video a analizar
    
    SeparadorFondo(video)                                       #Llamada a la fucion de separacion del fondo
    
    
    
    
    video.release()                                             #Se libera el video para evitar llenado inecesario de memoria
    cv2.destroyAllWindows()                                     #Se destruyen las pantallas creadas por la aplicacion
    print('Borrado exitoso')

def SeparadorFondo(video):

    fgbg = cv2.createBackgroundSubtractorMOG2(history = 500,varThreshold = 16,detectShadows = False)        #Creacion de operando de algoritmo de Gaussianas mixtas para separacion de fondo
    
    while(video.isOpened()):                                    #REvisa si el video se encuentra abierto       
        ret, frame = video.read()                               #Obtiene la informacion del video
        fgmask = fgbg.apply(frame)                              #Crea una mascara con el operando de MOG2 aplicado

        cv2.imshow('FondoSeparado',fgmask)                      #Abre en una pantalla el video con el algoritmo MOG2 aplicado
        k = cv2.waitKey(30) & 0xff                              #Crea un delay de 30 milisegundos para que la pantalla sea capaz de realizar las operaciones 
        if k == 27:
            break
        if cv2.waitKey(30) & 0xFF == ord('q'):                  #Si la tecla q es presionada, se detiene la presentacion de la mascara creada
            break

    
    
