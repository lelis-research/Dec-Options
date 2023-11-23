FOURROOMS_EASY_DIR=fourrooms_easy/logs/
FOURROOMS_MEDIUM_DIR=fourrooms_medium/logs/
FOURROOMS_HARD_DIR=fourrooms_hard/logs/
mkdir -p $FOURROOMS_EASY_DIR
mkdir -p $FOURROOMS_MEDIUM_DIR
mkdir -p $FOURROOMS_HARD_DIR

CODE_LOCATION=/home/alikhasi/Dec-Options

sbatch ./fourrooms_easy_training.sh $CODE_LOCATION $FOURROOMS_EASY_DIR