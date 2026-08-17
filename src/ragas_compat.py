"""Ragas 0.4.x imports LangChain VertexAI symbols that were removed from langchain-community."""
from __future__ import annotations

import sys
import types


class _MissingLangChainModel:
    """Placeholder so isinstance() checks in Ragas do not crash on import."""


def apply() -> None:
    def ensure(name: str) -> types.ModuleType:
        mod = sys.modules.get(name)
        if mod is None:
            mod = types.ModuleType(name)
            sys.modules[name] = mod
        parent_name, _, child = name.rpartition(".")
        if parent_name:
            parent = ensure(parent_name)
            setattr(parent, child, mod)
        return mod

    vertex_chat = ensure("langchain_community.chat_models.vertexai")
    vertex_chat.ChatVertexAI = getattr(vertex_chat, "ChatVertexAI", _MissingLangChainModel)

    try:
        import langchain_community.llms as community_llms
    except Exception:
        community_llms = ensure("langchain_community.llms")
    if not hasattr(community_llms, "VertexAI"):
        community_llms.VertexAI = _MissingLangChainModel
