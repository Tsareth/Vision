import numpy as np                          # Importar librerias necesarias
import cv2
from vidgear.gears import VideoGear
from vidgear.gears import WriteGear

global prev                                 # Definicion de variables
prev = []

def main():
 
    video = VideoGear(source='VideoEjemploRecortado.mp4', stabilize = True).start()     # Abre stream de video y lo estabiliza
    width = 1280                                                                        # Dimensiones de video (1080p)                         
    height = 720
    
    fgbg = cv2.createBackgroundSubtractorMOG2(history = 1000,varThreshold = 16,detectShadows = False)        #Creacion de operando de algoritmo de Gaussianas mixtas para separacion de fondo

    while(True):                                                                #Revisa si el video se encuentra abierto       
        frame = cv2.convertScaleAbs(video.read())                               #Obtiene la informacion del video
        if frame is None:
            break
        frame = cv2.resize(frame,(1280,720),fx=0,fy=0, interpolation = cv2.INTER_CUBIC) # convertir a 720p
        cv2.rectangle(frame,(width - 570,height - 400),(width,height),(255,0,0),2)
        fgmask = fgbg.apply(frame)                              #Crea una mascara con el operando de MOG2 aplicado
        
        rdi_filtrado = filtrar(fgmask, width - 570, height - 400, width, height)                        # Filtrar mascara
        fotograma_etiquetado = etiquetar(frame, rdi_filtrado, width - 570, height - 400, width, height) # Equiquetar vehiculos en el fotograma
        
        cv2.imshow('Resultado',fotograma_etiquetado)            # Abre en una pantalla el video procesado
        k = cv2.waitKey(30) & 0xff                              # Crea un delay de 30 milisegundos para que la pantalla sea capaz de realizar las operaciones 
        if k == 27:
            break
        if cv2.waitKey(30) & 0xFF == ord('q'):                  # Si la tecla q es presionada, se detiene la presentacion de la mascara creada
            break
        
    video.stop()                                             
    cv2.destroyAllWindows()                                     # Se destruyen las pantallas creadas por la aplicacion
    print('Borrado exitoso')



def filtrar(mascara, x0, y0, x1, y1):
    
    crop = mascara[y0:y1, x0:x1]                                # Se corta la zona de RDI de la mascara
    
    filtro1 = cv2.erode(crop,np.ones((1,1),np.uint8),iterations = 6)        # Primer filtro: erosion
    filtro2 = cv2.dilate(filtro1,np.ones((1,1),np.uint8),iterations = 12)   # Segundo filtro: dilatacion
    
    filtro_final = filtro2
    
    cv2.imshow('Mascara RDI', filtro_final)                                 # Se muestra la mascara procesada
    
    return filtro_final


def etiquetar(fotograma, mascara, x0, y0, x1, y1):
    
    contours, hierarchy = cv2.findContours(mascara, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) # Extraccion de contornos
    out = fotograma
    objetos = []
    
    for cnt in contours:                                        # Demarcacion de contornos
        if cv2.contourArea(cnt) > 700:                          # Area de contorno debe ser mas grande que 700
            x,y,w,h = cv2.boundingRect(cnt)
            cv2.rectangle(out,(x0+x,y0+y),(x0+x+w,y0+y+h),(0,255,0),2)  # Dibujar rectangulo de indicacion
            objetos.append([x,y,w,h])

    objetos = identificar(objetos)
    
    for obj in objetos:
        x, y, w, h, etiqueta = obj
        cv2.putText(out,                                                # Se coloca la etiqueta
            'V#' + str(etiqueta),
            (x0 + x + int(w/2), y0 + y - 15),
            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)
    
    return out
    

def identificar(objetos):                       # Identifica los contornos por medio de la evaluacion de su distancia euclideana
    
    global prev
    out = []
    out = objetos
    usadas = []                                 # Etiquetas ya utilizadas en esta iteracion
    tol = 20                                    # Tolerancia de movimiento en pixeles
    
    if len(prev) == 0 and len(objetos) > 0:     # Si no hay nada en previo le asigna un numero a cada objeto nuevo
        for i in range(len(objetos)):
            out[i].append(i)

    elif len(objetos) > 0 and len(prev) > 0:                      # Si existen objetos se comienza a evaluar
        for obj in out:
            x, y, w, h = obj
            flag = False
            for pre in prev:
                x_prev, y_prev, w_prev, h_prev, etiqueta = pre

                if (x_prev - tol < x < x_prev + tol) and (y_prev - tol < y < y_prev + tol):
                    obj.append(etiqueta)                # Si la esquina del rectangulo tiene una posicion parecida a uno anterior se le da la misma etiqueta
                    usadas.append(etiqueta)
                    flag = True
                    break

            if flag ==  False:
                for i in range(100):
                    if i not in usadas:             # Le asigna un numero que no este usado y sale del ciclo
                        obj.append(i)
                        usadas.append(i)
                        break
                

    prev = out
    return out

main()
