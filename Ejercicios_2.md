# Fundamentos neurona artificial 

Esta hoja explica cómo funciona una neurona artificial perceptrón.

Gráfica de puntos con línea divisoria: 
Muestra cómo un perceptrón aprende a separar dos clases (por ejemplo, puntos rojos y azules) mediante una línea recta.
Esa línea es la frontera de decisión definida por la ecuación:

donde w1, w2 son los pesos y bias es el término de desplazamiento.

Función de activación sigmoide o escalón:
Muestra cómo la suma ponderada de las entradas pasa por una función que devuelve 0 o 1.

* Si la salida es ≥ 0, se activa (1).

* Si es < 0, no se activa (0).

Ejemplo con entradas y salidas:
La tabla con d, v y salidas (0, 1) representa valores de entrada (distancias o datos) y cómo la neurona responde dependiendo del peso asignado.

Dibujo con el muñeco y líneas:
Representa un ejemplo geométrico: el muñeco está en una posición
(x,y) y las líneas dividen regiones dependiendo de los pesos.
Es una forma visual de mostrar cómo el perceptrón clasifica puntos en el plano.
![Imagen 8](/Imagenes/Imagen_8.jpeg)
![Imagen 9](/Imagenes/Imagen_9.jpeg)

# Redes Neuronales y Funciones de Activación

Esta hoja expande el concepto a redes neuronales multicapa.

Diagrama con varias capas de neuronas:
Representa una red neuronal con capa de entrada, oculta y salida.
Cada neurona de una capa se conecta con todas las de la siguiente.

Tablas con X1, X2 y target:
Son ejemplos de conjuntos de entrenamiento, posiblemente para funciones lógicas como AND, OR o XOR.
Muestra cómo las combinaciones de entrada producen una salida deseada target.

Funciones de activación (relu, sigmoide, tangente hiperbólica, escalón):
Muestran diferentes tipos de funciones usadas según la red:

* ReLU: f(x)=max(0,x)

* Sigmoide: f(x)=1/(1+e−x)

* Tang: salida entre -1 y 1

* Escalón: salida binaria (0 o 1)

Fórmula (x1 w1 + x2 w2 − bias):
Es la suma ponderada de entradas antes de aplicar la función de activación.

Curvas de error:
Indican el proceso de aprendizaje supervisado, donde la red ajusta los pesos para minimizar el error entre la salida real y la deseada.

![Imagen 10](/Imagenes/Imagen_10.jpeg)
![Imagen 11](/Imagenes/Imagen_11.jpeg)

# Procesamiento de Imágenes y CNN

Esta hoja explica cómo las redes neuronales convolucionales (CNN) procesan imágenes.

Cuadrículas con R, G, B:
Representan los canales de color de una imagen (Rojo, Verde, Azul).
Cada canal tiene valores de intensidad entre 0 y 255.

28×28 = 784:
Se refiere al tamaño típico de imágenes en datasets como MNIST (dígitos escritos a mano).
Cada imagen se convierte en una matriz de 784 píxeles.

Bloques 3×3 y convolución:
Muestran el proceso de aplicar un filtro (kernel) a la imagen.
El filtro (matriz pequeña) se desliza sobre la imagen multiplicando y sumando valores.
Ejemplo:

1/9 (x1 + x2 + ... + x9)

que corresponde a un filtro de suavizado promedio.

Resultado:
Los valores resultantes forman una nueva matriz que resalta bordes, colores o texturas.
Es la base del funcionamiento de las CNNs.

![Imagen 12](/Imagenes/Imagen_12.jpeg)


# Arbol de decisión del juego de disparos dentro de un cuadro

Descripción del juego

Un muñeco (jugador) que se encuentra en el centro de un cuadro cerrado. Dentro del cuadro hay una o varias balas que se mueven constantemente rebotando contra las paredes. El objetivo del muñeco es sobrevivir el mayor tiempo posible esquivando las balas.

Árbol de decisión 

El árbol de decisión del juego comienza en el momento en que el sistema inicia y el muñeco se encuentra en el centro del cuadro. En este punto, el modelo analiza el entorno observando la posición de las balas y su dirección de movimiento. Si el modelo detecta que no hay balas cercanas, el muñeco no realiza ningún movimiento, permaneciendo quieto en su posición actual. Este comportamiento permite evitar desplazamientos innecesarios. Sin embargo, si el modelo entrenado identifica que una bala se aproxima, el árbol de decisión pasa al siguiente nivel, donde se evalúa la dirección desde la cual viene la bala.

Cuando la bala se mueve directamente hacia el muñeco, el modelo determina la trayectoria de impacto. Si la bala proviene del lado izquierdo, el muñeco se moverá hacia la derecha; si viene desde la derecha, se desplazará hacia la izquierda. En el caso de que la amenaza provenga desde arriba, el muñeco optará por moverse hacia abajo, y si la bala se acerca desde abajo, subirá para evitarla. Estos movimientos son el resultado de una decisión lógica que busca alejar al muñeco de la dirección del peligro.

Después de tomar la decisión de moverse, el modelo realiza una verificación adicional para asegurarse de que la nueva posición no implique una colisión con otra bala. Si se detecta que el movimiento elegido podría poner al muñeco en riesgo de chocar con otra bala, se busca una ruta alternativa, ya sea desplazándose en diagonal o permaneciendo inmóvil momentáneamente hasta que el camino sea seguro.

Finalmente, una vez ejecutada la acción o decisión correspondiente, el modelo vuelve al punto inicial del árbol para repetir el proceso. Esto ocurre de manera constante, generando un comportamiento dinámico y reactivo.

![Imagen 13](/Imagenes/Imagen_13.jpeg)

# Tabla con posibles valores 

Reglas usadas

El área del juego es una cuadrícula con coordenadas en el rango 0..10 en ambos ejes si la bala llegara a <0 o >10 rebota invirtiendo su componente de velocidad.

La bala se mueve cada paso sumando (Vel_X_bala, Vel_Y_bala) a su posición. Si el siguiente movimiento sale del rango, esa componente de velocidad se invierte (rebote).

El muñeco se mueve 1 unidad por paso en la dirección que decide. Para decidir la acción se usó una regla simple: comparar la diferencia en x y y entre la bala y el muñeco; el muñeco se mueve en la dirección opuesta en el eje con mayor diferencia absoluta por ejemplo si la bala está mayormente a la izquierda, el muñeco se mueve a la derecha. Si las diferencias son 0, permanece quieto. El muñeco también respeta los límites 0..10.

En el paso 9 se observa un ejemplo de rebote: la bala llegó a X=10 que es el límite, por lo que su Vel_X_bala se invirtió a -1 y Vel_Y_bala a 1 por la lógica de rebote en los límites.

![Imagen 14](/Imagenes/Imagen_14.png)