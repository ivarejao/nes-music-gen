docker run --rm \
    --entrypoint bash \
    -v "/home/igor/Desktop/Repos/music-nes-lmf/datasets/nesmdb/nesmdb_vgm:/data" \
    nesmdb-env -c '
    find /data -type f -name "*.vgm" | while read -r file; do
        echo "Converting: $file"
        nesmdb_convert vgm_to_wav "$file"
    done
    echo "Batch conversion complete!"
'
