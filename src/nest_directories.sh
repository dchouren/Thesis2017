DIR=$1

pushd $DIR
for x in $(ls); do
    mv $(echo $x) tmp
    mkdir $(echo $x)
    mv tmp $(echo $x)/$(echo $x)
done

popd
