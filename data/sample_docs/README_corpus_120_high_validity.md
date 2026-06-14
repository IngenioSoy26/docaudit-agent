# Corpus 120 High Validity

Este repositorio mantiene el corpus de validacion dentro de `data/sample_docs`, organizado por caso de uso y por tipo de documento.

## Estructura actual

Cada carpeta de caso de uso conserva los archivos base originales en la raiz y, ademas, incorpora directamente los documentos sinteticos de alta validez en subcarpetas por formato:

```text
caso_uso_X/
  native_pdf/
    conforme/
    no_conforme/
  scanned_blurry_pdf/
    conforme/
    no_conforme/
  image_photo/
    conforme/
    no_conforme/
  image_handwritten/
    conforme/
    no_conforme/
```

No se utiliza una subcarpeta intermedia `corpus_120_high_validity`. Los documentos quedaron integrados directamente dentro de cada caso de uso para respetar la estructura final del proyecto.

## Cobertura del corpus

- `3` casos de uso:
  - `caso_uso_1_auditoria_hipotecaria`
  - `caso_uso_2_auditoria_fiscal`
  - `caso_uso_3_kyc_onboarding`
- `4` tipos de soporte por caso:
  - PDF nativo
  - PDF escaneado o borroso
  - imagen fotografiada
  - imagen con contenido manuscrito
- `40` documentos por caso de uso
- `120` documentos en total
- `120` ficheros `ground_truth` asociados

## Criterio de organizacion

La estructura actual busca:

- mantener un corpus util para pruebas desde la app y para experimentacion controlada;
- reflejar de forma clara la variabilidad documental evaluada en la tesis;
- dejar versionado en GitHub solo el corpus formal de `data/sample_docs`;
- evitar depender de carpetas auxiliares externas para la validacion principal.
