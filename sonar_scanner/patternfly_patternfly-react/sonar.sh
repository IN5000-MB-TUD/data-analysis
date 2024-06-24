# Create token variable
export SONAR_TOKEN=637440d7e41570e8cf14fb4c8ce5c7f807c5430f

# Loop through folders and run sonar command
echo "Start Sonar analysis"
for d in */ ; do
    echo $d
    cd $d
    /home/mattia-cs/sonar-scanner/bin/sonar-scanner -Dsonar.organization=in5000-fp -Dsonar.projectKey=in5000-fp_patternfly-react -Dsonar.sources=. -Dsonar.inclusions=**/*.js,**/*.ts -Dsonar.host.url=https://sonarcloud.io
    cd ../
    echo "---------------------------------------------------------------------------------------------------------------------------"
done

echo "Done!"
