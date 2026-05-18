"""Agentes del sistema.

Cada agente encapsula una responsabilidad del pipeline:
- classifier: selección de caso de uso (schema)
- extractor: extracción estructurada (LLM + RAG)
- auditor: evaluación de reglas y generación de informe
"""
