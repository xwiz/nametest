#!/usr/bin/env python3
"""
SRM — Stochastic Resonance Memory
Entry point: delegates to srm.cli.main()

Usage
-----
    python main.py --seed                         load sample KB, enter REPL
    python main.py --seed -q "DNA replication"    single query and exit
    python main.py --seed --verbose               REPL with attractor details
    python main.py --load facts.txt               load custom knowledge base
    python main.py --stats                        show store statistics
    python main.py --help                         full argument reference
"""

from srm.cli import main

if __name__ == "__main__":
    main()
