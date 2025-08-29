"""
This module contains the configuration for the asset analyst system.
"""

import os
from dotenv import load_dotenv

# from typing import Optional
from asset_analyst.configs.config_enum import ConfigEnum

load_dotenv()


class Config:
    _instance = None  # class-level singleton storage

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if not hasattr(self, "initialized"):  # run only once
            for enum_member in ConfigEnum:
                # attach using ENV VAR name (current behavior)
                setattr(self, enum_member.value, os.getenv(enum_member.value))
                # attach using ENUM attribute name (new behavior)
                setattr(self, enum_member.name, os.getenv(enum_member.value))
                # print(f"{enum_member.name}:{os.getenv(enum_member.value)}")
            self.initialized = True

    def validate(self):
        missing = []
        for key in ConfigEnum:
            if not getattr(self, key.value):
                missing.append(key.value)
        if missing:

            raise RuntimeError(f"Missing env vars: {', '.join(missing)}")
