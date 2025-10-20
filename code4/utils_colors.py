#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Color palette utilities (project-wide).

Priority:
1) ENV MBIO_COLORS_FILE -> YAML
2) config/config.yml -> key 'colors_file' -> YAML
3) config/colors.yml
Fallback: safe defaults.

Expected YAML (colors file):
  group:
    Fecal_Crohn:  "#E76F51"
    Oral_Crohn:   "#516D99"
    Fecal_Healthy:"#3A455B"
    Oral_Healthy: "#83AAAC"
  synonyms:
    Fecal_Crohn:   ["Crohn-Fecal","Fecal-Crohn","CF"]
    Oral_Crohn:    ["Crohn-Oral","Oral-Crohn","CO"]
    Fecal_Healthy: ["Healthy-Fecal","Fecal-Healthy","HF"]
    Oral_Healthy:  ["Healthy-Oral","Oral-Healthy","HO"]

Returned mapping (standard keys for plots):
  "Crohn-Fecal", "Crohn-Oral", "Healthy-Fecal", "Healthy-Oral"
"""

import os, warnings, yaml

_STD_KEYS = ["Crohn-Fecal", "Crohn-Oral", "Healthy-Fecal", "Healthy-Oral"]

def _safe_defaults():
    return {
        "Crohn-Oral":    "#516D99",
        "Crohn-Fecal":   "#E76F51",
        "Healthy-Oral":  "#83AAAC",
        "Healthy-Fecal": "#3A455B",
    }

def _normalize_palette_from_yaml(cfg_colors: dict, synonyms: dict):
    # canonical (e.g., Fecal_Crohn) -> color
    canon2color = dict(cfg_colors or {})
    # alias -> color
    alias2color = {}
    for canon, aliases in (synonyms or {}).items():
        col = canon2color.get(canon)
        if not col: 
            continue
        for a in aliases:
            alias2color[a] = col

    out = {}
    for std in _STD_KEYS:
        if std in alias2color:
            out[std] = alias2color[std]
            continue
        # try direct canonical guess: "Crohn-Fecal" -> "Fecal_Crohn"
        p = std.split("-")  # ["Crohn","Fecal"]
        canon_guess = f"{p[1]}_{p[0]}"
        out[std] = canon2color.get(canon_guess, _safe_defaults().get(std))
    return out

def load_palette():
    """Return dict with keys: Crohn-Fecal, Crohn-Oral, Healthy-Fecal, Healthy-Oral."""
    tried = []

    # 1) explicit env
    env_p = os.environ.get("MBIO_COLORS_FILE")
    if env_p and os.path.isfile(env_p):
        tried.append(env_p)
        try:
            with open(env_p, "r") as f:
                cc = yaml.safe_load(f) or {}
            return _normalize_palette_from_yaml(cc.get("group"), cc.get("synonyms"))
        except Exception:
            pass

    # 2) config/config.yml -> colors_file
    cfg_p = os.path.join("config", "config.yml")
    colors_file = None
    if os.path.isfile(cfg_p):
        try:
            with open(cfg_p, "r") as f:
                cfg = yaml.safe_load(f) or {}
            colors_file = cfg.get("colors_file")
        except Exception:
            colors_file = None

    # candidate list in order
    cands = []
    if colors_file: cands.append(colors_file)
    cands.append(os.path.join("config", "colors.yml"))

    for p in cands:
        if p and os.path.isfile(p):
            tried.append(p)
            try:
                with open(p, "r") as f:
                    cc = yaml.safe_load(f) or {}
                return _normalize_palette_from_yaml(cc.get("group"), cc.get("synonyms"))
            except Exception:
                continue

    warnings.warn(f"[colors] Could not load from {tried}; using safe defaults.")
    return _safe_defaults()
