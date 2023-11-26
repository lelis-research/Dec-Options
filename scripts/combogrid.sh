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

# training tasks
sbatch ./combogrid_training.sh $CODE_LOCATION $COMBOGRID3X3_DIR $COMBOGRID3X3_CONFIG
sbatch ./combogrid_training.sh $CODE_LOCATION $COMBOGRID4X4_DIR $COMBOGRID4X4_CONFIG
sbatch ./combogrid_training.sh $CODE_LOCATION $COMBOGRID5X5_DIR $COMBOGRID5X5_CONFIG
sbatch ./combogrid_training.sh $CODE_LOCATION $COMBOGRID6X6_DIR $COMBOGRID6X6_CONFIG
