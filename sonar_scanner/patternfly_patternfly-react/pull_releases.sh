# Pull repos
versions=("@patternfly/react-virtualized-extension@4.8.52 @patternfly/react-catalog-view-extension@4.26.15 @patternfly/react-catalog-view-extension@4.75.3 @patternfly/react-catalog-view-extension@4.81.1 @patternfly/react-catalog-view-extension@4.87.4 @patternfly/react-catalog-view-extension@4.92.23 @patternfly/react-docs@5.102.66 @patternfly/react-catalog-view-extension@4.93.3
 @patternfly/react-catalog-view-extension@4.93.17 @patternfly/react-docs@5.103.66 @patternfly/react-docs@5.103.72 @patternfly/react-docs@5.103.76 @patternfly/react-code-editor@4.82.115 @patternfly/react-docs@5.103.80 @patternfly/react-inline-edit-extension@4.86.123 @patternfly/react-icons@4.93.7 @patternfly/react-table@4.113.3 demo-app-ts@4.210.16 v5.1.1 @patternfly/react-inline-edit-extension@4.86.129 v5.3.3")

for v in $versions
do
   v_no_backslash=${v//\//-}
   timestamp=$(date +%s%N)
   folder_name="$timestamp.$v_no_backslash"
   echo $folder_name

   git clone --depth 1 --branch $v https://github.com/patternfly/patternfly-react $folder_name
done

