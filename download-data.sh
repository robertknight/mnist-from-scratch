#!/bin/sh

FILES=$(echo train-images-idx3-ubyte.gz \
             train-labels-idx1-ubyte.gz \
             t10k-images-idx3-ubyte.gz \
             t10k-labels-idx1-ubyte.gz)

# The original MNIST dataset.
CLASSIC_MNIST=http://yann.lecun.com/exdb/mnist

# The harder Fashion-MNIST dataset.
# See https://github.com/zalandoresearch/fashion-mnist
FASHION_MNIST=http://fashion-mnist.s3-website.eu-central-1.amazonaws.com

fetch_dataset()
{
  NAME=$1
  URL=$2
  DEST_DIR=data/$NAME

  rm -rf $DEST_DIR
  mkdir -p $DEST_DIR

  for FILE in $FILES
  do
    echo "fetching URL $URL/$FILE"
    DEST=$(basename $FILE .gz)
    curl -s "$URL/$FILE" | gzip -d > $DEST_DIR/$DEST
  done
}

fetch_dataset classic "$CLASSIC_MNIST"
fetch_dataset fashion "$FASHION_MNIST"
