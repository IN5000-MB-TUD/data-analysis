# Create token variable
export SONAR_TOKEN=9f5803d35d1a19e490d0b4ab5d2a8b84cb99f380

# Loop through folders and run sonar command
echo "Start Sonar analysis"
for d in */ ; do
    echo $d
    cd $d
    /home/mattia-cs/sonar-scanner/bin/sonar-scanner -Dsonar.organization=in5000-fp -Dsonar.projectKey=in5000-fp_pip -Dsonar.sources=. -Dsonar.inclusions=**/*.py -Dsonar.host.url=https://sonarcloud.io
    cd ../
    echo "---------------------------------------------------------------------------------------------------------------------------"
done

echo "Done!"
