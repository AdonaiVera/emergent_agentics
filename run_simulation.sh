#!/bin/bash

for i in {0..10}
do
    echo "Running scenario $i..."
    ./run_backend_automatic.sh \
        --env_name simulacra \
        -o base_party \
        -t simulation_crosmodality_qwen2f$i \
        -s 600 \
        --ui True \
        --scenario_index $i

    echo "Finished scenario $i"
done

echo "All scenarios completed!"
