# Create token variable
export SONAR_TOKEN=sonar_token_here

# Loop through folders and run sonar command
echo "Start Sonar analysis"
for d in */ ; do
    echo $d
    cd $d
    /home/mattia-cs/sonar-scanner/bin/sonar-scanner -Dsonar.organization=in5000-fp -Dsonar.projectKey=in5000-fp_cockroach -Dsonar.sources=. -Dsonar.inclusions=**/*.go -Dsonar.host.url=https://sonarcloud.io
    cd ../
    echo "---------------------------------------------------------------------------------------------------------------------------"
done

echo "Done!"
