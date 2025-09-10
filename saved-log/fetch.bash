rsync -av \
    --include '*/' \
    --include 'log.txt' \
    --exclude '*' \
    "snellius-river:/home/rbetting/spatial_delays/saved/" .
