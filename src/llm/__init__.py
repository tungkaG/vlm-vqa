"""LLM client layer. All VLM calls go through this package.

Providers: 'nvidia' (default), 'gemini', 'mock'.
Use :func:`llm.factory.build_llm_client` (or the alias
``build_gemini_client``) to get a client from an :class:`~config.AppConfig`.
"""
