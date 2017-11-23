#!/bin/bash
# Reconstruct golden-angle radial dynamic MRI data

# setting paths
SPF=8
PHASES=100
LD=0.01
SCALE=0.5
OUTNAME="out"
TRAJFILE="./traj"
B1FILE="./b1"
KFILE="./kspace"
function usage()
{
    echo "Reconstruct golden-angle radial dynamic MRI data"
    echo ""
    echo "-h: Display help"
    echo "--spf=$SPF: Spokes-per-frame"
    echo "--phases=$PHASES: number of phases"
    echo "--lambda=$LD: reg par"
    echo "--scale=$SCALE: nufft scale"
    echo "--outfile=$OUTNAME: output filename"
    echo "--trajfile=$OF: output filename"
    echo "--b1file=$OF: output filename"
    echo "--kfile=$OF: output filename"
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
        --spf)
            SPF=$VALUE
            ;;
        --phases)
            PHASES=$VALUE
            ;;
        --scale)
            SCALE=$VALUE
            ;;
        --lambda)
            LD=$VALUE
            ;;
        --outfile)
            OUTNAME=$VALUE
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

set -e
set -x

OUTDIR=$(date +%Y%m%d_%H%M%S)
mkdir $OUTDIR


mv graspbart.* ./$OUTDIR/
mv graspbartsens.* ./$OUTDIR/
mv tmp1* ./$OUTDIR/

SCALE=$SCALE
SPOKES=$SPF
PHASES=$PHASES
KSPACE=./$OUTDIR/graspbart
SENS=./$OUTDIR/graspbartsens



echo "SPOKES=$SPF PHASES=$PHASES KSPACE=$KSPACE SENS=$SENS"
READ=$(/home/mr_recon/Workspace/bart/bart show -d0 $KSPACE)
LINES=$(/home/mr_recon/Workspace/bart/bart show -d1 $KSPACE)
PHASES=$(($LINES / $SPOKES))

# create Golden-ratio radial trajectory
#/home/mr_recon/Workspace/bart/bart traj -G -x$READ -y$LINES $OUTDIR/tmp1

# over-sampling x2
/home/mr_recon/Workspace/bart/bart scale $SCALE $OUTDIR/tmp1  $OUTDIR/tmp2

# split off time dimension into index 10
/home/mr_recon/Workspace/bart/bart reshape $(/home/mr_recon/Workspace/bart/bart bitmask 2 10) $SPOKES $PHASES  $OUTDIR/tmp2  $OUTDIR/traj

rm $OUTDIR/tmp1.*  $OUTDIR/tmp2.*

# split-off time dim
/home/mr_recon/Workspace/bart/bart reshape $(/home/mr_recon/Workspace/bart/bart bitmask 1 2) $SPOKES $PHASES $KSPACE  $OUTDIR/tmp1

# move time dimensions to dim 10 and reshape
/home/mr_recon/Workspace/bart/bart transpose 2 10  $OUTDIR/tmp1  $OUTDIR/tmp2

/home/mr_recon/Workspace/bart/bart reshape $(/home/mr_recon/Workspace/bart/bart bitmask 0 1 2) 1 $READ $SPOKES  $OUTDIR/tmp2  $OUTDIR/kspace

rm  $OUTDIR/tmp1.*  $OUTDIR/tmp2.*


# Reconstruct with total variation in time (GRASP)
/home/mr_recon/Workspace/bart/bart pics -S -d5 -u10. -i100 -RT:$(/home/mr_recon/Workspace/bart/bart bitmask 10):0:$LD -i50 -t $OUTDIR/traj $OUTDIR/kspace $SENS $OUTNAME

# Reconstruct with l1-wavelets in space and total variation in time
#/home/mr_recon/Workspace/bart/bart pics -S -d5 -i 100 -R T:$(/home/mr_recon/Workspace/bart/bart bitmask 10):0:$LD -R W:$(/home/mr_recon/Workspace/bart/bart bitmask 0 1 2):0:0.001 -t traj kspace $SENS img

# Reconstruct with multi-scale low rank across space and time
#/home/mr_recon/Workspace/bart/bart pics -S -d5 -i 100  -R M:$(/home/mr_recon/Workspace/bart/bart bitmask 0 1 2):$(/home/mr_recon/Workspace/bart/bart bitmask 0 1 2):0.05 -t traj kspace $SENSimg

rm -rf  $OUTDIR



