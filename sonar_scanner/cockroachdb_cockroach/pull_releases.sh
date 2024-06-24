# Pull repos
versions=("v1.0 v1.0.7 v1.1.0 v1.1.9 v2.0.0 v2.0.7 v2.1.0 v2.1.11 v19.1.0 v19.1.11 v19.2.0 v19.2.12 v20.1.0 v20.1.17 v20.2.0 v20.2.19 v21.1.0 v21.1.21 v21.2.0 v21.2.17 v22.1.0 v22.1.22 v22.2.8 v22.2.19 v23.1.0 v23.1.22 v24.1.0-alpha.1 v24.1.0")

for v in $versions
do
   v_no_backslash=${v//\//-}
   timestamp=$(date +%s%N)
   folder_name="$timestamp.$v_no_backslash"
   echo $folder_name

   git clone --depth 1 --branch $v https://github.com/cockroachdb/cockroach $folder_name
done

