#!/bin/bash
TASK_NAME="$1"
MODE="$2"
EXTRA_ARG="$3"
printf "=================================\n"
printf "Run task $TASK_NAME in mode $MODE\n"
printf "=================================\n"

CONTAINER_NAME="genie_sim_benchmark"
START_SCRIPT="$PWD/scripts/start_gui.sh"
TERMINAL_ENV="autorun"
PROCESS_CLIENT="task_benchmark|teleop|replay|infer"
PROCESS_SERVER="isaac-sim|omni|omni_python|raise|ros"

SERVER_CONFIG=""


if [ "$TASK_NAME" = "clean" ]; then
    printf "\nEnter clean mode..."
    for i in $(seq 1 5); do
        docker exec -it $CONTAINER_NAME bash -c "pkill -9 -f '$PROCESS_CLIENT\|$PROCESS_SERVER'" || true
        sleep 1
    done
    printf "\nFinish cleaning env..."
    docker exec -it $CONTAINER_NAME bash -c "find output -type f -name "*.db3" -exec rm -v {} \;" || true
    exit 0
fi

if [ "$TASK_NAME" = "genie_task_home_pour_water" ]; then
    SERVER_CONFIG="--enable_gpu_dynamics --render_mode RealTimePathTracing"
elif [ "$TASK_NAME" = "genie_task_home_wipe_dirt" ]; then
    SERVER_CONFIG="--enable_gpu_dynamics --render_mode RealTimePathTracing"
elif [ "$TASK_NAME" = "iros_pack_moving_objects_from_conveyor" ]; then
    SERVER_CONFIG="--reset_fallen=True"
fi

printf "\nRun with server config: $SERVER_CONFIG\n"

if ! docker inspect --format='{{.State.Running}}' "$CONTAINER_NAME" 2>/dev/null | grep -q "true"; then
    printf "\nWarning: Contrainer $CONTAINER_NAME not running, try to start..."

    if [ -x "$START_SCRIPT" ]; then
        printf "Executing script: $START_SCRIPT"
        "$START_SCRIPT"
        sleep 5
        if docker inspect --format='{{.State.Running}}' "$CONTAINER_NAME" 2>/dev/null | grep -q "true"; then
            printf "Info: Container $CONTAINER_NAME started"
        else
            printf "Error: Container failed to start"
            exit 1
        fi
    else
        printf "Error:  Start script $START_SCRIPT not exist or not executable"
        exit 1
    fi
else
    printf "Info: Container $CONTAINER_NAME already running"
fi

if [ "$MODE" = "pico" ]; then
    printf "\n\nEnter pico teleop mode...\n\n"
    if [ -z "$3" ]; then
        printf "Error: IP address needed..."
        exit 1
    fi
    declare -a COMMANDS=(
        "docker exec -it $CONTAINER_NAME bash -ic 'run_server $SERVER_CONFIG'"
        "docker exec -it $CONTAINER_NAME bash -ic 'run_teleop $TASK_NAME --mode pico --host_ip $EXTRA_ARG --record'"
    )
    declare -a DELAYS=(0 3)
elif [ "$MODE" = "keyboard" ]; then
    printf "\n\nEnter keyboard teleop mode...\n\n"
    declare -a COMMANDS=(
        "docker exec -it $CONTAINER_NAME bash -ic 'run_server $SERVER_CONFIG'"
        "docker exec -it $CONTAINER_NAME bash -ic 'run_teleop $TASK_NAME --mode keyboard --record'"
    )
    declare -a DELAYS=(0 3)
elif [ "$MODE" = "infer" ]; then
    printf "\n\nEnter model infer mode...\n\n"

    if [ -z "$EXTRA_ARG" ]; then
        MODEL="User Defined Model"
        MODEL_PATH="AgiBot-World"
    else
        MODEL=$EXTRA_ARG
        MODEL_PATH="AgiBot-World/$MODEL"
    fi

    printf "Model type: $MODEL \n"
    printf "Model path: $MODEL_PATH \n"

    declare -a COMMANDS=(
        "docker exec -it $CONTAINER_NAME bash -ic 'cd $MODEL_PATH && python3 scripts/infer.py --task_name $TASK_NAME'"
        "docker exec -it $CONTAINER_NAME bash -ic 'run_server $SERVER_CONFIG'"
        "docker exec -it $CONTAINER_NAME bash -ic 'run_client $TASK_NAME --policy_class=BaselinePolicy > client.log'"
    )
    declare -a DELAYS=(0 10 13)
elif [ "$MODE" = "replay" ]; then
    printf "\n\nEnter replay mode...\n\n"
    declare -a COMMANDS=(
        "docker exec -it $CONTAINER_NAME bash -ic 'run_server --record_img --disable_physics --record_video $SERVER_CONFIG'"
        "docker exec -it $CONTAINER_NAME bash -ic 'run_replay --task_file /root/workspace/main/source/geniesim/benchmark/ader/eval_tasks/$TASK_NAME.json --state_file /root/workspace/main/$EXTRA_ARG --record'"
    )
    declare -a DELAYS=(0 3)
else
    printf "\n\nEnter empty benchmark mode...\n\n"
    declare -a COMMANDS=(
        "docker exec -it $CONTAINER_NAME bash -ic 'run_server $SERVER_CONFIG'"
        "docker exec -it $CONTAINER_NAME bash -ic 'run_client $TASK_NAME'"
    )
    declare -a DELAYS=(0 3)
fi

TERMINAL_CMD=""
for term in gnome-terminal konsole xterm terminator; do
    if command -v "$term" &>/dev/null; then
        case "$term" in
        gnome-terminal) TERMINAL_CMD="gnome-terminal -- bash -c" ;;
        konsole) TERMINAL_CMD="konsole -e bash -c" ;;
        xterm) TERMINAL_CMD="xterm -e" ;;
        terminator) TERMINAL_CMD="terminator -e" ;;
        esac
        break
    fi
done

if [ -z "$TERMINAL_CMD" ]; then
    printf "\nNo terminal emulator found. Please install one and try again."
    exit 1
fi

for i in "${!COMMANDS[@]}"; do
    sleep "${DELAYS[$i]}"
    if [[ "$TERMINAL_CMD" == "gnome-terminal"* ]]; then
        gnome-terminal -- bash -c "export TERMINAL_ENV=$TERMINAL_ENV; ${COMMANDS[$i]}; exec bash" &
    else
        $TERMINAL_CMD "${COMMANDS[$i]}" &
    fi
done


client_log="client.log"
timeout_cnt=0
timeout_thresh=60

printf "\nAll started"
while true; do
    read -t 1 -n 1 input
    if [[ "$input" == "n" || "$input" == "N" ]]; then
        printf "\nFinish task..."
        docker exec -it $CONTAINER_NAME bash -c "pkill -SIGINT -f '$PROCESS_CLIENT'" || true
        reset
        exit 1
    fi
    if [[ "$input" == "q" || "$input" == "Q" ]]; then
        printf "\nKill task..."
        docker exec -it $CONTAINER_NAME bash -c "pkill -9 -f '$PROCESS_CLIENT\|$PROCESS_SERVER'" || true
        reset
        exit 1
    fi
    if docker exec -it $CONTAINER_NAME ps aux | pgrep -f raise_standalone_sim > /dev/null; then
        printf "\rServer alive ...\n"

    else
        printf "\nServer dead..."
        reset
        exit 0
    fi

    if [ "$MODE" = "infer" ]; then
        initial_size=$(stat -c %s "$client_log")
        sleep 1
        current_size=$(stat -c %s "$client_log")

        if [ "$initial_size" -eq "$current_size" ]; then
            ((timeout_cnt++))
        else
            timeout_cnt=0
        fi

        if [ "$timeout_cnt" -gt "$timeout_thresh" ]; then
            printf "\n Time out..."
            docker exec -it $CONTAINER_NAME bash -c "pkill -9 -f '$PROCESS_CLIENT\|$PROCESS_SERVER'" || true
            reset
            exit 1
        fi
    else
        sleep 1
    fi
done
