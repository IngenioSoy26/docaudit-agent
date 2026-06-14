# [OPEN] Debug Session: image-ocr-fallback

## Contexto
- Sintoma reportado: las imagenes ahora se procesan con `ollama_vision` y no con `easyocr`.
- Comportamiento esperado: las imagenes deberian priorizar `EasyOCR` y solo usar vision como fallback.
- Observacion del usuario: ayer por la noche las pruebas funcionaban correctamente con `EasyOCR`; hoy, incluso tras volver a un commit anterior, el problema persiste.

## Hipotesis iniciales
1. `EasyOCR` ya no esta disponible o falla al importar en el entorno con el que se esta lanzando Streamlit.
2. `EasyOCR` sigue instalado, pero lanza una excepcion en tiempo de ejecucion y por eso el codigo cae silenciosamente a `ollama_vision`.
3. El entorno desde el que hoy se ejecuta la app no es el mismo que el de ayer, aunque el codigo sea el mismo.
4. Alguna dependencia transitiva de `EasyOCR` o `torch` cambio o se rompio fuera del repositorio.
5. El problema no esta en el OCR en si, sino en un archivo concreto o en una familia de imagenes, y por eso parece un fallo global.

## Plan
1. Verificar el commit actual y el estado del repositorio.
2. Revisar la ruta de OCR de imagen en `core/document_loader.py`.
3. Comprobar en tiempo de ejecucion si `easyocr` puede importarse y procesar una imagen.
4. Comparar el entorno actual con la hipotesis de cambio externo al codigo.
5. Confirmar la causa real antes de proponer cualquier cambio.

## Evidencia recogida
- El repositorio esta actualmente en `a9aa167`.
- En `a9aa167` y tambien en `4d52eaa`, `core/document_loader.py` contiene la misma llamada:
  - `results = reader.readtext(prepared, detail=1)`
- Esa ruta usa `prepared` como objeto `PIL.Image`, y el codigo cae a `ollama_vision` ante cualquier excepcion en EasyOCR:
  - `except ImportError: method = "ollama_vision"`
  - `except Exception: method = "ollama_vision"`
- El historial de `core/document_loader.py` muestra que los ultimos commits del archivo son:
  - `a9aa167`
  - `86b8fe9`
- No hay evidencia en los commits recientes de una version guardada en Git donde esa llamada ya estuviera corregida a `numpy array`.
- La salida de entorno mostro que se esta usando el Python de `.venv`, por lo que no hay evidencia inmediata de que hoy se este lanzando con otro ejecutable distinto.

## Analisis de hipotesis
- H1. `EasyOCR` no disponible: no confirmada; hay indicios de que el modulo arranca y usa CPU.
- H2. `EasyOCR` falla en runtime y el pipeline cae a `ollama_vision`: fuertemente respaldada por el codigo actual y por el comportamiento observado.
- H3. Se usa otro entorno distinto al de ayer: no respaldada de momento con la evidencia recogida.
- H4. Dependencia externa rota: posible, pero no necesaria para explicar el problema actual.
- H5. Falla de una imagen concreta: no respaldada; la causa principal ya aparece en la ruta general de imagen.

## Conclusión provisional
- El problema no parece ser un cambio "de hoy" dentro del repositorio.
- La version `a9aa167` a la que se volvio ya contiene la ruta que puede hacer caer silenciosamente a `ollama_vision`.
- Si ayer funcionaba con `EasyOCR`, lo mas probable es una de estas dos situaciones:
  1. estabas ejecutando una version local no guardada en Git;
  2. el proceso de Streamlit que funcionaba ayer tenia codigo cargado distinto al que ahora quedo tras el reset.

## Estado
- Pendiente de confirmacion del usuario para aplicar la correccion minima en la ruta `EasyOCR` y verificarla manualmente en la app.
