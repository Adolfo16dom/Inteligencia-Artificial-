# Acertijos 
* ## Los tres monjes y los tres canivales 
Tres misioneros representados por el circulo y tres caníbales representados por un palito  están en la orilla izquierda de un río. Tienen una barca que puede llevar 1 o 2 personas. Deben pasar todos a la orilla derecha. Regla crítica: en ninguna orilla los caníbales pueden ser más que los misioneros (si hay misioneros allí), porque entonces se los comerían.

* ## Personas que cruzan el rio en menos de 17 minutos 
Cuatro personas necesitan cruzar un puente nocturno con una sola linterna. Máximo 2 personas cruzan a la vez y deben llevar la linterna. Cada persona tiene un tiempo distinto para cruzar: por ejemplo 1 min, 2 min, 5 min y 10 min. Si dos cruzan juntos, van al ritmo del más lento. Pueden cruzar todos en no más de 17 minutos.

* ## No cruzar la reina en un tablero de 4x4
Colocar 4 reinas en un tablero de ajedrez 4×4 de forma que ninguna pueda atacar a otra.

![Imagen 1](/Imagenes/Imagen_1.jpeg)

# Algoritmo A*
En qué consiste

A* busca un camino desde un nodo inicial hasta un nodo objetivo, explorando nodos intermedios. Para decidir qué nodo expandir primero, combina:

Función de evaluación:

f(n)=g(n)+h(n)

g(n): lo que ya llevamos recorrido.

h(n): lo que falta según la heurística.

f(n): es la suma de g(n) mas h(n).

A* siempre expande primero el nodo con el menor 

f(n).

* Pasos básicos del algoritmo

    * Poner el nodo inicial en una lista abierta (nodos por explorar).

    * Repetir mientras la lista no esté vacía:

    * Sacar el nodo con menor 

    * Si es el objetivo hemos encontrado el camino.

    * Si no, expandir sus vecinos:

    * Calcular g,h,f de cada uno.

    * Añadirlos a la lista abierta si no estaban antes, o actualizar si ahora se obtiene un mejor costo.

    * Mover el nodo actual a la lista cerrada (ya explorados).

    * Reconstruir el camino óptimo siguiendo los predecesores.

* ## Ejemplo 1
![Imagen 2](/Imagenes/Imagen_2.jpeg)
* ## Ejemplo 2
![Imagen 3](/Imagenes/Imagen_3.jpeg)
* ## Ejemplo 3
![Imagen 4](/Imagenes/Imagen_4.jpeg)

# Problema del recorrido del caballo en el tablero de ajedrez
Llenar todos los espacios del tablero de ajedrez con movimientos de la pieza del caballo sin que vuelva a pasar por el mismo punto que habia pasado.
![Imagen 5](/Imagenes/Imagen_5.jpeg) 

## Cuadro magico 3x3 
l cuadrado mágico 3x3 consiste en colocar los números del 1 al 9 en una cuadrícula de 3 filas por 3 columnas, de manera que:

Cada fila, cada columna y las dos diagonales sumen lo mismo.
En el caso del 3x3, la suma mágica es 15.

## El Solitario triangular
Consiste en un tablero con hoyos en forma de triángulo generalmente 15 hoyos cada uno con una canica excepto uno, que queda vacío en este caso el ultimo inferior a la derecha o sea el numero 15.

* Reglas básicas:

Solo puedes mover una canica saltando sobre otra adyacente (como en las damas).

El salto debe terminar en un hoyo vacío, en línea recta.

La canica saltada se retira del tablero.

El objetivo es seguir haciendo saltos hasta que quede una sola canica en el tablero.

* Objetivo:

El reto es elegir la secuencia correcta de saltos para terminar con una sola pieza.
Si terminas con más de una ficha, el reto no está resuelto.
![Imagen 6](/Imagenes/Imagen_6.jpeg) 

