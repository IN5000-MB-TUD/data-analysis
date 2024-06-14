# Pull repos
versions=("0.3 0.7 1.0.2 1.1 1.4rc3 1.4rc4 1.5.1rc1 1.5.1 6.0.3 7.0.1 7.0.2 8.1.0 8.1.1 10.0.1 18.0.1 19.2.1 19.2.2 20.1.1 20.2 20.3.2 20.3.3 21.2.1 21.2.2 22.0.4 22.1 23.0 23.1 24.0")

for v in $versions
do
   v_no_backslash=${v//\//-}
   timestamp=$(date +%s%N)
   folder_name="$timestamp.$v_no_backslash"
   echo $folder_name

   git clone --depth 1 --branch $v https://github.com/pypa/pip $folder_name
done

