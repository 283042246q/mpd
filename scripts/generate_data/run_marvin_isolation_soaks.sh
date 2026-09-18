#!/usr/bin/env bash
set -euo pipefail

repository="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
python_bin="${PYTHON_BIN:-python}"
duration_seconds="${MARVIN_SOAK_DURATION_SECONDS:-2400}"
cpu_list="${MARVIN_SOAK_CPU_LIST:-0-1,3-31}"
dataset="${MARVIN_SOAK_DATASET:-${repository}/data_public/data_trajectories/EnvWarehouse-RobotMarvinBimanual-independent-v3-res002-nosimplifier-150k-3/dataset_merged.hdf5}"
config="${MARVIN_SOAK_CONFIG:-${repository}/data_generation_cfgs/EnvWarehouse-RobotMarvinBimanual-independent.yaml}"
run_id="$(date +%Y%m%d-%H%M%S)"
output_root="${MARVIN_SOAK_OUTPUT_ROOT:-${repository}/diagnostics/marvin-isolation-${run_id}}"
runner="${repository}/scripts/generate_data/soak_marvin_pipeline_stage.py"
modes="${MARVIN_SOAK_MODES:-endpoint gpu pybullet}"

mkdir -p "${output_root}"
printf 'Marvin isolation output: %s\n' "${output_root}"
printf 'Each stage: %s seconds; CPU affinity: %s\n' "${duration_seconds}" "${cpu_list}"

for mode in ${modes}; do
    case "${mode}" in
        endpoint|gpu|pybullet) ;;
        *) printf 'Unsupported MARVIN_SOAK_MODES entry: %s\n' "${mode}" >&2; exit 2 ;;
    esac
    printf '\n[%s] starting\n' "${mode}"
    mkdir -p "${output_root}/${mode}"
    if [[ "${mode}" == "gpu" ]]; then
        taskset -c "${cpu_list}" "${python_bin}" "${runner}" \
            --mode "${mode}" \
            --config "${config}" \
            --dataset "${dataset}" \
            --output-dir "${output_root}/${mode}" \
            --duration-seconds "${duration_seconds}" \
            --excluded-cpu 2 \
            2>&1 | tee "${output_root}/${mode}/console.log"
    else
        CUDA_VISIBLE_DEVICES='' taskset -c "${cpu_list}" "${python_bin}" "${runner}" \
            --mode "${mode}" \
            --config "${config}" \
            --dataset "${dataset}" \
            --output-dir "${output_root}/${mode}" \
            --duration-seconds "${duration_seconds}" \
            --excluded-cpu 2 \
            2>&1 | tee "${output_root}/${mode}/console.log"
    fi
done

printf '\nAll isolation soaks completed: %s\n' "${output_root}"
