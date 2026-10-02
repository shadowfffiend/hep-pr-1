#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_FILE=""
SEED=42
OUT_ROOT="$ROOT_DIR/runs"

while (($#)); do
  case "$1" in
    -h|--help)
      echo "Использование: $0 --data ПУТЬ [--seed ЦЕЛОЕ] [--out-dir ПУТЬ]"
      exit 0 ;;
    --data|--seed|--out-dir)
      if (($# < 2)); then echo "Не указано значение для $1" >&2; exit 2; fi
      case "$1" in
        --data) DATA_FILE="$2" ;;
        --seed) SEED="$2" ;;
        --out-dir) OUT_ROOT="$2" ;;
      esac
      shift 2 ;;
    *) echo "Неизвестный аргумент: $1" >&2; exit 2 ;;
  esac
done

[[ -f "$DATA_FILE" ]] || { echo "Укажите существующий CSV-файл через --data ПУТЬ" >&2; exit 2; }
[[ "$SEED" =~ ^[0-9]+$ ]] || { echo "--seed должен быть неотрицательным целым числом" >&2; exit 2; }

DATA_FILE="$(realpath "$DATA_FILE")"
mkdir -p "$OUT_ROOT"
OUT_ROOT="$(realpath "$OUT_ROOT")"
RUN_DIR="$(mktemp -d "$OUT_ROOT/$(date +%Y%m%d_%H%M%S)_seed${SEED}_XXXXXX")"

cd "$ROOT_DIR"
echo "Каталог запуска: $RUN_DIR"
uv run --frozen python -m src.train --data "$DATA_FILE" --seed "$SEED" --out-dir "$RUN_DIR" 2>&1 | tee "$RUN_DIR/run.log"
echo "Готово: $RUN_DIR"
