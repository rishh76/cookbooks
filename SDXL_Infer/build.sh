set -e

docker build -t sdxlinfer -f sdxl.dockerfile .

docker tag sdxlinfer qblockrepo/staging:sdxl-infer-dev

docker push qblockrepo/staging:sdxl-infer-dev