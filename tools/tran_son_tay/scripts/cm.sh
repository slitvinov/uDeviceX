mkdir -p cm

ls *.ply -1 | xargs -I {} -n1 echo 'boundply -m < {} > cm/{}' > s.sh
echo "Creating cm files"
sh s.sh
