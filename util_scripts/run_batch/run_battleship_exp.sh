# !/bin/bash
current_path=`pwd`
current_dir=`basename "$current_path"`

if [ "GAN_Harmonized_with_HMMs" != "$current_dir" ]; then
    echo "You should run this script in GAN_Harmonized_with_HMMs/ directory!!"
    exit 1
fi

# mkdir config dir
dir=`dirname $0`
config_dir=$dir/temp
mkdir -p $config_dir

# generate config and run.sh
for gan_gumbel in soft hard
do
    for intra_gumbel in soft
    do
        name='gan_gumbel_'${gan_gumbel}'_intra_gumbel_'${intra_gumbel}
        config_path=$config_dir/$name
        hrun -c 1 -m 4 python $dir/process.py --config ./src/GAN-based-model/config.yaml --gan_gumbel $gan_gumbel --intra_gumbel $intra_gumbel --output $config_path
        ./run_battleship.sh $config_path ${name}_ &
    done
done

rm -r $config_dir
