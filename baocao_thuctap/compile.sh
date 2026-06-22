#!/bin/bash
# Usage: ./compile.sh [main|tung|phuc|all]

compile_one() {
    name="$1"
    echo "=== Compiling $name ==="
    xelatex -interaction=nonstopmode "$name" 2>&1 | tail -1
    bibtex "$name" 2>&1 | tail -1
    xelatex -interaction=nonstopmode "$name" 2>&1 | tail -1
    xelatex -interaction=nonstopmode "$name" 2>&1 | tail -1
    echo "$name.pdf done"
}

case "${1:-all}" in
    main|base)       compile_one main ;;
    tung)            compile_one baocao_tung ;;
    phuc)            compile_one baocao_phuc ;;
    all|*)           compile_one main && compile_one baocao_tung && compile_one baocao_phuc ;;
esac
