# Introducción a las redes neuronales

Notebooks del módulo de redes neuronales de *Fundamentos de la Inteligencia Artificial y Programación de APIs para LLMs*.

1. Funcionamiento de una neurona
2. Neurona con activación sigmoide
3. Descenso por el gradiente
4. Aprendizaje de la neurona
5. La red neuronal
6. Redes neuronales multiclase
7. Redes neuronales para texto: LLMs
8. Modelos locales: un LLM en tu propio ordenador
9. API: cómo hablan los programas con los modelos
10. Razonamiento: de la cadena de pensamiento a los modelos que razonan
11. Herramientas: cuando el modelo necesita actuar
12. MCP: un estándar para conectar modelos y herramientas
13. Agentes: modelos que trabajan paso a paso hacia un objetivo

## Leer en la web

La web se publica con GitHub Pages desde la raíz de la rama `main` (*Settings → Pages → Deploy from a branch → main / (root)*). La portada es `index.html` y cada notebook tiene su versión HTML, ya ejecutada, en `notebooks/`.

## Trabajar en Google Colab

Cada notebook tiene un botón **Abrir en Colab**, también disponible en la web. Desde Colab, *Archivo → Guardar una copia en Drive* crea una copia propia en la cuenta de Google de cada alumno. Las imágenes se cargan desde este repositorio y el notebook 06 incluye una celda inicial que clona el repositorio para disponer de `data/` y `libs/`. La última celda del 08 y el código de los notebooks 09 a 13 necesitan LM Studio en el ordenador local y no funcionan en Colab.

Para abrir un notebook en Colab: `https://colab.research.google.com/github/cayetanoguerra/redes-neuronales/blob/main/notebooks/<notebook>.ipynb`

## Ejecutar los notebooks

```bash
pip install -r requirements.txt
jupyter lab notebooks/
```

Ejecuta los notebooks con `notebooks/` como directorio de trabajo: necesitan `imgs/`, `data/` y `libs/`.

Los notebooks 09 a 13 necesitan [LM Studio](https://lmstudio.ai) con el servidor local activo y los modelos `qwen/qwen3-4b-2507` y `qwen/qwen3-4b-thinking-2507`. El 12 y el 13 crean al ejecutarse el archivo `servidor_cuaderno.py` y, el 13, la carpeta `informes/`.

## Regenerar el HTML

Si modificas un notebook, vuelve a ejecutarlo y a convertirlo:

```bash
cd notebooks
jupyter nbconvert --to notebook --execute --inplace 0*.ipynb 1*.ipynb
jupyter nbconvert --to html --template lab 0*.ipynb 1*.ipynb
```

Después revisa en el HTML generado que los enlaces entre notebooks apunten a `.html` en lugar de a `.ipynb`.
