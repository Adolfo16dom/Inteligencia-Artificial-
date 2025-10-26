# Ejercicio de convolución 3x3 con filtro promedio de 1/9
 
 Aplicar un kernel 3×3 con todos sus elementos iguales a 1/9 sobre la matriz de entrada usando stride=1 y
 padding=1 (relleno de ceros).

* Matriz de entrada (5×4):

![Imagen 15](/Imagenes/Imagen_15.png)

* Imagen con padding de ceros (7×6):

![Imagen 16](/Imagenes/Imagen_16.png)

* Celda de salida (0, 0) — Ventana 3×3 centrada en la posición del pivote.
* Ventana 3×3 tomada de la imagen con padding:

![Imagen 17](/Imagenes/Imagen_17.png)

* Multiplicación elemento a elemento (cada celda: (1/9)*valor)
* Lo cual la suma de todas las multiplicaciones para la celda (0,0) es 19.55556

![Imagen 18](/Imagenes/Imagen_18.png)

* Celda de salida (0, 1) — Ventana 3×3 centrada en la posición del pivote.
* Ventana 3×3 tomada de la imagen con padding:
 
![Imagen 19](/Imagenes/Imagen_19.png)

* Multiplicación elemento a elemento (cada celda: (1/9)*valor)
* Lo cual la suma de todas las multiplicaciones para la celda (0,1) es 27.55556

![Imagen 20](/Imagenes/Imagen_20.png)

* Este procedimiento se haria con cada celda de la matriz.
* La matriz de salida (5×4) seria esta con valores flotantes:

![Imagen 21](/Imagenes/Imagen_21.png)




