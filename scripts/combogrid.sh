COMBOGRID3X3_DIR=combogrid3x3/logs/
COMBOGRID3X3_CONFIG=combogrid3x3.json
COMBOGRID4X4_DIR=combogrid4x4/logs/
COMBOGRID4X4_CONFIG=combogrid4x4.json
COMBOGRID5X5_DIR=combogrid5x5/logs/
COMBOGRID5X5_CONFIG=combogrid5x5.json
COMBOGRID6X6_DIR=combogrid6x6/logs/
COMBOGRID6X6_CONFIG=combogrid6x6.json
BASELINE_VANILLA=Vanilla
BASELINE_NeuralAugmented=NeuralAugmented
BASELINE_DecOptionsWhole=DecOptionsWhole
BASELINE_DecOptions=DecOptions

rm -rf nc*
mkdir -p $COMBOGRID3X3_DIR
mkdir -p $COMBOGRID4X4_DIR
mkdir -p $COMBOGRID5X5_DIR
mkdir -p $COMBOGRID6X6_DIR

CODE_LOCATION=/home/alikhasi/Dec-Options

# training tasks - assuming perfect convergence in our training agents
sbatch ./combogrid_training.sh $CODE_LOCATION $COMBOGRID3X3_DIR $COMBOGRID3X3_CONFIG
sbatch ./combogrid_training.sh $CODE_LOCATION $COMBOGRID4X4_DIR $COMBOGRID4X4_CONFIG
sbatch ./combogrid_training.sh $CODE_LOCATION $COMBOGRID5X5_DIR $COMBOGRID5X5_CONFIG
sbatch ./combogrid_training.sh $CODE_LOCATION $COMBOGRID6X6_DIR $COMBOGRID6X6_CONFIG

cp -rf $COMBOGRID3X3_DIR/task1_seed1_MODEL.zip $COMBOGRID3X3_DIR/task1_seed0_MODEL.zip
cp -rf $COMBOGRID3X3_DIR/task2_seed1_MODEL.zip $COMBOGRID3X3_DIR/task2_seed3_MODEL.zip
cp -rf $COMBOGRID3X3_DIR/task3_seed1_MODEL.zip $COMBOGRID3X3_DIR/task3_seed24_MODEL.zip
cp -rf $COMBOGRID3X3_DIR/task4_seed1_MODEL.zip $COMBOGRID3X3_DIR/task4_seed26_MODEL.zip

cp -rf $COMBOGRID4X4_DIR/task2_seed0_MODEL.zip $COMBOGRID4X4_DIR/task2_seed14_MODEL.zip
cp -rf $COMBOGRID4X4_DIR/task4_seed0_MODEL.zip $COMBOGRID4X4_DIR/task4_seed10_MODEL.zip

cp -rf $COMBOGRID5X5_DIR/task1_seed26_MODEL.zip $COMBOGRID5X5_DIR/task1_seed22_MODEL.zip
cp -rf $COMBOGRID5X5_DIR/task1_seed29_MODEL.zip $COMBOGRID5X5_DIR/task1_seed27_MODEL.zip
cp -rf $COMBOGRID5X5_DIR/task2_seed2_MODEL.zip $COMBOGRID5X5_DIR/task2_seed0_MODEL.zip
cp -rf $COMBOGRID5X5_DIR/task2_seed2_MODEL.zip $COMBOGRID5X5_DIR/task2_seed1_MODEL.zip
cp -rf $COMBOGRID5X5_DIR/task2_seed7_MODEL.zip $COMBOGRID5X5_DIR/task2_seed3_MODEL.zip
cp -rf $COMBOGRID5X5_DIR/task2_seed11_MODEL.zip $COMBOGRID5X5_DIR/task2_seed8_MODEL.zip
cp -rf $COMBOGRID5X5_DIR/task2_seed25_MODEL.zip $COMBOGRID5X5_DIR/task2_seed12_MODEL.zip
cp -rf $COMBOGRID5X5_DIR/task2_seed29_MODEL.zip $COMBOGRID5X5_DIR/task2_seed26_MODEL.zip
cp -rf $COMBOGRID5X5_DIR/task4_seed25_MODEL.zip $COMBOGRID5X5_DIR/task4_seed18_MODEL.zip
cp -rf $COMBOGRID5X5_DIR/task4_seed29_MODEL.zip $COMBOGRID5X5_DIR/task4_seed26_MODEL.zip

cp -rf $COMBOGRID6X6_DIR/task2_seed0_MODEL.zip $COMBOGRID6X6_DIR/task2_seed28_MODEL.zip

# Parameter search
sbatch --time=0-3:00 ./combogrid_parameter_search.sh $CODE_LOCATION $COMBOGRID3X3_DIR $BASELINE_VANILLA $COMBOGRID3X3_CONFIG
sbatch --time=0-5:00 ./combogrid_parameter_search.sh $CODE_LOCATION $COMBOGRID3X3_DIR $BASELINE_NeuralAugmented $COMBOGRID3X3_CONFIG
sbatch --time=0-6:00 ./combogrid_parameter_search.sh $CODE_LOCATION $COMBOGRID3X3_DIR $BASELINE_DecOptionsWhole $COMBOGRID3X3_CONFIG
sbatch --time=0-7:00 ./combogrid_parameter_search.sh $CODE_LOCATION $COMBOGRID3X3_DIR $BASELINE_DecOptions $COMBOGRID3X3_CONFIG

sbatch --time=0-3:00 ./combogrid_parameter_search.sh $CODE_LOCATION $COMBOGRID4X4_DIR $BASELINE_VANILLA $COMBOGRID4X4_CONFIG
sbatch --time=0-5:00 ./combogrid_parameter_search.sh $CODE_LOCATION $COMBOGRID4X4_DIR $BASELINE_NeuralAugmented $COMBOGRID4X4_CONFIG
sbatch --time=0-6:00 ./combogrid_parameter_search.sh $CODE_LOCATION $COMBOGRID4X4_DIR $BASELINE_DecOptionsWhole $COMBOGRID4X4_CONFIG
sbatch --time=0-7:00 ./combogrid_parameter_search.sh $CODE_LOCATION $COMBOGRID4X4_DIR $BASELINE_DecOptions $COMBOGRID4X4_CONFIG

sbatch --time=0-3:00 ./combogrid_parameter_search.sh $CODE_LOCATION $COMBOGRID5X5_DIR $BASELINE_VANILLA $COMBOGRID5X5_CONFIG
sbatch --time=0-5:00 ./combogrid_parameter_search.sh $CODE_LOCATION $COMBOGRID5X5_DIR $BASELINE_NeuralAugmented $COMBOGRID5X5_CONFIG
sbatch --time=0-6:00 ./combogrid_parameter_search.sh $CODE_LOCATION $COMBOGRID5X5_DIR $BASELINE_DecOptionsWhole $COMBOGRID5X5_CONFIG
sbatch --time=0-7:00 ./combogrid_parameter_search.sh $CODE_LOCATION $COMBOGRID5X5_DIR $BASELINE_DecOptions $COMBOGRID5X5_CONFIG

sbatch --time=0-3:00 ./combogrid_parameter_search.sh $CODE_LOCATION $COMBOGRID6X6_DIR $BASELINE_VANILLA $COMBOGRID6X6_CONFIG
sbatch --time=0-5:00 ./combogrid_parameter_search.sh $CODE_LOCATION $COMBOGRID6X6_DIR $BASELINE_NeuralAugmented $COMBOGRID6X6_CONFIG
sbatch --time=0-6:00 ./combogrid_parameter_search.sh $CODE_LOCATION $COMBOGRID6X6_DIR $BASELINE_DecOptionsWhole $COMBOGRID6X6_CONFIG
sbatch --time=0-7:00 ./combogrid_parameter_search.sh $CODE_LOCATION $COMBOGRID6X6_DIR $BASELINE_DecOptions $COMBOGRID6X6_CONFIG