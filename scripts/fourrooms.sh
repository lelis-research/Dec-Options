FOURROOMS_EASY_DIR=fourrooms_easy/logs/
FOURROOMS_MEDIUM_DIR=fourrooms_medium/logs/
FOURROOMS_HARD_DIR=fourrooms_hard/logs/
BASELINE_VANILLA=Vanilla

rm -rf nc*
mkdir -p $FOURROOMS_EASY_DIR
mkdir -p $FOURROOMS_MEDIUM_DIR
mkdir -p $FOURROOMS_HARD_DIR

CODE_LOCATION=/home/alikhasi/Dec-Options

# training tasks
sbatch ./fourrooms_training.sh $CODE_LOCATION $FOURROOMS_EASY_DIR
# Parameter search for vanilla
sbatch --time=0-3:00 ./fourrooms_parameter_search.sh $CODE_LOCATION $FOURROOMS_EASY_DIR $BASELINE_VANILLA