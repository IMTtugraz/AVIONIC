#!/bin/bash
# AVIONIC DEMO perfusion MRI DATA

# setting paths
FUNCTYPE="ICTGV2"
PATTERN="vista"
R=16

function usage()
{
    echo "AVIONIC Demo for cardiac perfusion MRI data"
    echo ""
    echo "-h                      Display help"
    echo "--functype=$FUNCTYPE:   Regularization functional: ICTGV2, TGV2, TV"
    echo "--pattern=$PATTERN:     Sampling pattern: vista, vd (variable density), uni"
    echo "--red=$R options:       Undersampling factor: 4 8 12 16"
    echo ""
}

while [ "$1" != "" ]; do
    PARAM=`echo $1 | awk -F= '{print $1}'`
    VALUE=`echo $1 | awk -F= '{print $2}'`
    case $PARAM in
        -h | --help)
            usage
            exit
            ;;
        --functype)
            FUNCTYPE=$VALUE
            ;;
        --pattern)
            PATTERN=$VALUE
            ;;
        --red)
            R=$VALUE
            ;;
          *)
            echo "ERROR: unknown parameter \"$PARAM\""
            usage
            exit 1
            ;;
    esac
    shift
done
###########################################
echo "==================================================================================="
echo "Running $FUNCTYPE Reconstruction on cardiac perfusion data with $PATTERN pattern and R=$R"
echo "==================================================================================="

DATAFILE="cardiac_perfusion_data.bin"
PATTERNFILE="cardiac_perfusion_${PATTERN}_acc${R}.bin"
RESULTSFILE="${FUNCTYPE}_recon_cardiac_perfusion_${PATTERN}_acc${R}"

echo "==================================================================================="
echo "Downloading Data"
echo "==================================================================================="
if [ ! -f $DATAFILE ]
then
  wget ftp://ftp.tugraz.at/outgoing/AVIONIC/avionic_testdata/cardiac_perfusion_data.bin
fi

if [ ! -f $PATTERNFILE ]
then
  wget ftp://ftp.tugraz.at/outgoing/AVIONIC/avionic_testdata/$PATTERNFILE
fi

echo "==================================================================================="
echo "Run Reconstruction"
echo "==================================================================================="
mkdir ./results_perfusion/
nENC=128;nRO=128;nFRAMES=40;nCOILS=12;
nX=$nRO;nY=$nENC;


if [ ! -f ./results_perfusion/${RESULTSFILE}.bin ]
then

  recon_cmd="./CUDA/bin/avionic -i 500 -m $FUNCTYPE -e -a \
   	    -p ./CUDA/config/default_perf.cfg -d $nX:$nY:0:$nRO:$nENC:0:$nCOILS:$nFRAMES \
 			  $DATAFILE $PATTERNFILE \
			  ./results_perfusion/${RESULTSFILE}.bin"
	echo "------------------------------------------------------------------------"
  echo "$recon_cmd"
  echo "------------------------------------------------------------------------"
  eval $recon_cmd
	mv ./results_perfusion/x3_component ./results_perfusion/${RESULTSFILE}_comp.bin
	mv ./results_perfusion/PDGap ./results_perfusion/${RESULTSFILE}_pdgap.bin
	mv ./results_perfusion/b1_reconstructed.bin ./results_perfusion/${RESULTSFILE}_b1.bin
	mv ./results_perfusion/u0_reconstructed.bin ./results_perfusion/${RESULTSFILE}_initguess.bin
fi

echo "==================================================================================="
echo "Display Results"
echo "==================================================================================="

./python/show_avionic.py -f "./results_perfusion/${RESULTSFILE}.bin" -nx $nENC -ny $nRO -nframes $nFRAMES

if [ -f ./results_perfusion/${RESULTSFILE}_comp.bin ]
then
 ./python/show_avionic.py -f "./results_perfusion/${RESULTSFILE}_comp.bin" -nx $nENC -ny $nRO -nframes $nFRAMES
fi
