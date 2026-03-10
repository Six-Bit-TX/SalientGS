set -e

datasets=(
    # Mip-NeRF 360
    ../data_fastmap/bicycle
    # ../data_fastmap/counter
    # ../data_fastmap/flowers
    # ../data_fastmap/kitchen
    # ../data_fastmap/room
    # ../data_fastmap/bonsai
    # ../data_fastmap/garden
    # ../data_fastmap/stump
    # ../data_fastmap/treehill
    # # Tanks and Temples
    # ../data_fastmap/train
    # ../data_fastmap/truck
    # # Deep Blending
    # ../data_fastmap/drjohnson
    # ../data_fastmap/playroom
)

for data_path in ${datasets[@]}; do
    rm -rf "$data_path/database.db"
    python global_corr/global_matching.py --image_dir $data_path/images --output_dir $data_path
    python fastmap/run.py --headless \
        --database $data_path/database.db \
        --image_dir $data_path/images \
        --output_dir $data_path
    rm -rf $data_path/gsplat/ 
    sgs-joint --data_path $data_path
done
