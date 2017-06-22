daint:/scratch/snx3000/lina/RBC/tanktreading/grid/links

/scratch/snx3000/lina/RBC/tanktreading/grid/links/RBCgammaC_0.0/RBCp_0.0039/RBCkb_100.0/RBCkbT_0.1/RBCgammaT_NaN
/scratch/snx3000/lina/RBC/tanktreading/grid/links/RBCgammaC_0.0/RBCp_0.0039/RBCkb_100.0/RBCkbT_0.1/RBCgammaT_NaN

rsync -avz --copy-links daint:/scratch/snx3000/lina/RBC/tanktreading/grid/links.daint/RBCgammaC_0.0/RBCp_0.0039/RBCkb_100.0/RBCkbT_0.1/RBCgammaT_NaN/ . --exclude '*/e/*' --exclude '*/test'

sh_1.0
sh_6.0
sh_12.0


for f in ply/*.ply; do v=`basename $f .ply`.vtk; nb=498 ply2vtk $f vtk/$v; done
