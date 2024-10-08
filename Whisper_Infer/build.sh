set -e

docker build -t whisperinfer -f whisper.dockerfile .

docker tag whisperinfer qblockrepo/staging:sdxl-infer-dev

docker push qblockrepo/staging:whisper-infer-dev