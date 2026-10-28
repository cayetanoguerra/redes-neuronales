# Introducción a las redes neuronales

Notebooks del módulo de redes neuronales de *Fundamentos de la Inteligencia Artificial y Programación de APIs para LLMs*.

1. Funcionamiento de una neurona
2. Neurona con activación sigmoide
3. Descenso por el gradiente
4. Aprendizaje de la neurona
5. La red neuronal
6. Redes neuronales multiclase

## Leer en la web

La web se publica con GitHub Pages desde la raíz de la rama `main` (*Settings → Pages → Deploy from a branch → main / (root)*). La portada es `index.html` y cada notebook tiene su versión HTML, ya ejecutada, en `notebooks/`.

## Ejecutar los notebooks

```bash
pip install -r requirements.txt
jupyter lab notebooks/
```

Ejecuta los notebooks con `notebooks/` como directorio de trabajo: necesitan `imgs/`, `data/` y `libs/`.

## Regenerar el HTML

Si modificas un notebook, vuelve a ejecutarlo y a convertirlo:

```bash
cd notebooks
jupyter nbconvert --to notebook --execute --inplace 0*.ipynb
jupyter nbconvert --to html --template lab 0*.ipynb
```

Después revisa en el HTML generado que los enlaces entre notebooks apunten a `.html` en lugar de a `.ipynb`.
