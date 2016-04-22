#!/bin/bash
# AVIONIC DEMO CINE MRI DATA

# setting paths
FUNCTYPE="ICTGV2"
PATTERN="vista"
R=16

function usage()
{
    echo "AVIONIC Demo for CINE cardiac MRI data"
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

echo "====================================================================================="
echo "Running $FUNCTYPE Reconstruction on CINE cardiac data with $PATTERN pattern and R=$R"
echo "====================================================================================="

DATAFILE="cardiac_cine_data.bin"
PATTERNFILE="cardiac_cine_${PATTERN}_acc${R}.bin"
RESULTSFILE="${FUNCTYPE}_recon_cardiac_cine_${PATTERN}_acc${R}"
echo "$RESULTSFILE"

echo "==================================================================================="
echo "Downloading Data"
echo "==================================================================================="
if [ ! -f $DATAFILE ]
then
  wget ftp://ftp.tugraz.at/outgoing/AVIONIC/avionic_testdata/cardiac_cine_data.bin
fi

if [ ! -f $PATTERNFILE ]
then
  wget ftp://ftp.tugraz.at/outgoing/AVIONIC/avionic_testdata/$PATTERNFILE
fi

echo "==================================================================================="
echo "Run Reconstruction"
echo "==================================================================================="
mkdir ./results_cine/
nENC=168;nRO=416;nFRAMES=25;nCOILS=30;
nX=$nENC;nY=$nRO;


if [ ! -f ./results_cine/${RESULTSFILE}.bin ]
then

  recon_cmd="./CUDA/bin/avionic -i 500 -m ICTGV2 -e -a \
   	    -p ./CUDA/config/default_cine.cfg -d $nX:$nY:$nENC:$nRO:$nCOILS:$nFRAMES \
 			  $DATAFILE $PATTERNFILE \
			  ./results_cine/${RESULTSFILE}.bin"
	echo "------------------------------------------------------------------------"
  echo "$recon_cmd"
  echo "------------------------------------------------------------------------"
  eval $recon_cmd
	mv ./results_cine/x3_component ./results_cine/${RESULTSFILE}_comp.bin
	mv ./results_cine/PDGap ./results_cine/${RESULTSFILE}_pdgap.bin
	mv ./results_cine/b1_reconstructed.bin ./results_cine/${RESULTSFILE}_b1.bin
	mv ./results_cine/u0_reconstructed.bin ./results_cine/${RESULTSFILE}_initguess.bin
fi

echo "==================================================================================="
echo "Display Results"
echo "==================================================================================="

./python/show_avionic.py -f "./results_cine/${RESULTSFILE}.bin" -nx $nRO -ny $nENC -nframes $nFRAMES

if [ -f ./results_cine/${RESULTSFILE}_comp.bin ]
then
 ./python/show_avionic.py -f "./results_cine/${RESULTSFILE}_comp.bin" -nx $nRO -ny $nENC -nframes $nFRAMES
fi
