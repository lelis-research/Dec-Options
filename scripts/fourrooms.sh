FOURROOMS_EASY_DIR=fourrooms_easy/logs/
FOURROOMS_EASY_CONFIG=fourrooms_easy.json
FOURROOMS_MEDIUM_DIR=fourrooms_medium/logs/
FOURROOMS_MEDIUM_CONFIG=fourrooms_medium.json
FOURROOMS_HARD_DIR=fourrooms_hard/logs/
FOURROOMS_HARD_CONFIG=fourrooms_hard.json
BASELINE_VANILLA=Vanilla
BASELINE_NeuralAugmented=NeuralAugmented
BASELINE_DecOptionsWhole=DecOptionsWhole
BASELINE_DecOptions=DecOptions

rm -rf nc*
mkdir -p $FOURROOMS_EASY_DIR
mkdir -p $FOURROOMS_MEDIUM_DIR
mkdir -p $FOURROOMS_HARD_DIR

CODE_LOCATION=/home/alikhasi/Dec-Options

# training tasks
sbatch ./fourrooms_training.sh $CODE_LOCATION $FOURROOMS_EASY_DIR
cp -rf $FOURROOMS_EASY_DIR/task1*_MODEL* $FOURROOMS_MEDIUM_DIR
cp -rf $FOURROOMS_EASY_DIR/task2*_MODEL* $FOURROOMS_MEDIUM_DIR
cp -rf $FOURROOMS_EASY_DIR/task3*_MODEL* $FOURROOMS_MEDIUM_DIR
cp -rf $FOURROOMS_EASY_DIR/task1*_MODEL* $FOURROOMS_HARD_DIR
cp -rf $FOURROOMS_EASY_DIR/task2*_MODEL* $FOURROOMS_HARD_DIR
cp -rf $FOURROOMS_EASY_DIR/task3*_MODEL* $FOURROOMS_HARD_DIR

# Parameter search for vanilla
sbatch --time=0-3:00 ./fourrooms_parameter_search.sh $CODE_LOCATION $FOURROOMS_EASY_DIR $BASELINE_VANILLA $FOURROOMS_EASY_CONFIG
sbatch --time=0-5:00 ./fourrooms_parameter_search.sh $CODE_LOCATION $FOURROOMS_EASY_DIR $BASELINE_NeuralAugmented $FOURROOMS_EASY_CONFIG
sbatch --time=0-6:00 ./fourrooms_parameter_search.sh $CODE_LOCATION $FOURROOMS_EASY_DIR $BASELINE_DecOptionsWhole $FOURROOMS_EASY_CONFIG
sbatch --time=0-7:00 ./fourrooms_parameter_search.sh $CODE_LOCATION $FOURROOMS_EASY_DIR $BASELINE_DecOptions $FOURROOMS_EASY_CONFIG

sbatch --time=0-3:00 ./fourrooms_parameter_search.sh $CODE_LOCATION $FOURROOMS_MEDIUM_DIR $BASELINE_VANILLA $FOURROOMS_MEDIUM_CONFIG
sbatch --time=0-5:00 ./fourrooms_parameter_search.sh $CODE_LOCATION $FOURROOMS_MEDIUM_DIR $BASELINE_NeuralAugmented $FOURROOMS_MEDIUM_CONFIG
sbatch --time=0-6:00 ./fourrooms_parameter_search.sh $CODE_LOCATION $FOURROOMS_MEDIUM_DIR $BASELINE_DecOptionsWhole $FOURROOMS_MEDIUM_CONFIG
sbatch --time=0-7:00 ./fourrooms_parameter_search.sh $CODE_LOCATION $FOURROOMS_MEDIUM_DIR $BASELINE_DecOptions $FOURROOMS_MEDIUM_CONFIG

sbatch --time=0-3:00 ./fourrooms_parameter_search.sh $CODE_LOCATION $FOURROOMS_HARD_DIR $BASELINE_VANILLA $FOURROOMS_HARD_CONFIG
sbatch --time=0-5:00 ./fourrooms_parameter_search.sh $CODE_LOCATION $FOURROOMS_HARD_DIR $BASELINE_NeuralAugmented $FOURROOMS_HARD_CONFIG
sbatch --time=0-6:00 ./fourrooms_parameter_search.sh $CODE_LOCATION $FOURROOMS_HARD_DIR $BASELINE_DecOptionsWhole $FOURROOMS_HARD_CONFIG
sbatch --time=0-7:00 ./fourrooms_parameter_search.sh $CODE_LOCATION $FOURROOMS_HARD_DIR $BASELINE_DecOptions $FOURROOMS_HARD_CONFIG

# Train using best parameters