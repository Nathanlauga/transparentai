docker run -p 8888:8888 \
--mount type=bind,source="$(pwd)"/dev,target=/nb \
--mount type=bind,source="$(pwd)"/tests,target=/transparentai/tests \
--name transparentai transparentai-dev