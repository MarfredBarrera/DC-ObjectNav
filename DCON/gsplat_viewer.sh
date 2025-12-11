CKPT_DIR=/workspace/DCON/output/current_scene/model.pt
OUTPUT_DIR=/workspace/DCON/output/current_scene/results

TORCH_CUDA_ARCH_LIST=8.9+PTX CUDA_VISIBLE_DEVICES=1 python /workspace/gsplat/examples/simple_viewer.py \
        --ckpt $CKPT_DIR \
        --output_dir $OUTPUT_DIR \
        --port 8080